"""Warehouses the environment can load into and grade against.

A destination says which config.yaml field holds the per-rollout namespace, how to create and drop
that namespace, and which ELT-Bench connector reads it back for grading. A snapshot is a saved copy
of a verified extract/load, so transform-only rollouts can start from loaded sources without
re-running Airbyte. Supporting another warehouse means adding one class here; the environment
itself does not change.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Protocol

import duckdb
from db_connectors import BaseConnector, SnowflakeConnector
from tinker_cookbook import cli_utils


class Destination(Protocol):
    name: str  # key of the destination block in config.yaml
    namespace_field: str  # field of that block's config holding the per-rollout namespace

    def connector(self, workspace: Path) -> BaseConnector: ...

    def create_namespace(self, namespace: str, workspace: Path, snapshot: str | None = None) -> dict[str, str]:
        """Create the rollout's namespace (cloned from `snapshot` if given); return its config.yaml values."""

    def drop_namespace(self, namespace: str) -> None: ...

    def reset_namespace(self, namespace: str, workspace: Path, tables: list[str], snapshot: str | None = None) -> None:
        """Rebuild the namespace with only `tables`, copied from `snapshot` or else from the namespace itself."""

    def save_snapshot(self, namespace: str, workspace: Path, snapshot: str, tables: list[str]) -> None: ...


class DuckDBConnector(BaseConnector):
    """ELT-Bench connector for a DuckDB file, where a schema plays the role of the task namespace."""

    def _connect(self):
        return duckdb.connect(self.config["path"], read_only=True)

    def _quote(self, ident: str) -> str:
        return f'"{ident}"'

    def _norm(self, ident: str) -> str:
        return ident.lower()

    def _schemata_sql(self, database: str) -> str:
        return "SELECT schema_name FROM information_schema.schemata"

    def _tables_sql(self, database: str, schema: str) -> str:
        return (
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE'"
        )

    def _fqn(self, database: str, schema: str, table: str) -> str:
        return f"{self._quote(schema)}.{self._quote(table)}"


class DuckDBDestination:
    """Local warehouse for credential-free runs: each rollout writes /workspace/warehouse.duckdb."""

    name = "duckdb"
    namespace_field = "schema"
    snapshot_dir = Path(cli_utils.runs_path("eltbench_snapshots"))

    def connector(self, workspace: Path) -> BaseConnector:
        return DuckDBConnector({"path": str(workspace / "warehouse.duckdb")})

    def create_namespace(self, namespace: str, workspace: Path, snapshot: str | None = None) -> dict[str, str]:
        # Every rollout has its own database file, which the agent creates unless we start from a snapshot.
        if snapshot:
            _copy_duckdb_tables(self.snapshot_dir / f"{snapshot}.duckdb", "main", workspace / "warehouse.duckdb", namespace)
        return {"schema": namespace}

    def drop_namespace(self, namespace: str) -> None:
        pass

    def reset_namespace(self, namespace: str, workspace: Path, tables: list[str], snapshot: str | None = None) -> None:
        path = workspace / "warehouse.duckdb"
        if snapshot:
            path.unlink()
            _copy_duckdb_tables(self.snapshot_dir / f"{snapshot}.duckdb", "main", path, namespace, tables)
        else:
            old = path.rename(workspace / "warehouse.old.duckdb")
            _copy_duckdb_tables(old, namespace, path, namespace, tables)
            old.unlink()

    def save_snapshot(self, namespace: str, workspace: Path, snapshot: str, tables: list[str]) -> None:
        path = self.snapshot_dir / f"{snapshot}.duckdb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        _copy_duckdb_tables(workspace / "warehouse.duckdb", namespace, path, "main", tables)


def _copy_duckdb_tables(
    source: Path, source_schema: str, target: Path, target_schema: str, tables: list[str] | None = None
) -> None:
    """Copy `tables` (default: all of them) from a schema of one DuckDB file into a schema of another."""
    con = duckdb.connect(str(target))
    try:
        con.execute(f"ATTACH '{source}' AS source (READ_ONLY)")
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"')
        if tables is None:
            query = f"SELECT table_name FROM duckdb_tables() WHERE database_name = 'source' AND schema_name = '{source_schema}'"
            tables = [table for (table,) in con.execute(query).fetchall()]
        for table in tables:
            con.execute(f'CREATE TABLE "{target_schema}"."{table}" AS FROM source."{source_schema}"."{table}"')
    finally:
        con.close()


WAREHOUSE = "AIRBYTE_WAREHOUSE"  # created by ELT-Bench's setup/destination/setup.sql


class SnowflakeDestination:
    """Snowflake laid out like ELT-Bench: one database per namespace, tables in AIRBYTE_SCHEMA."""

    name = "snowflake"
    namespace_field = "database"

    def __init__(self, credential_path: str):
        self.credential_path = credential_path

    def connector(self, workspace: Path | None = None) -> BaseConnector:
        return SnowflakeConnector(json.loads(Path(self.credential_path).read_text()))

    def create_namespace(self, namespace: str, workspace: Path, snapshot: str | None = None) -> dict[str, str]:
        # Each rollout gets its own role and user that own only this database, so rollouts can't read
        # or change each other's tables (ELT-Bench shares one AIRBYTE_ROLE between all tasks).
        role, user, password = f"{namespace}_ROLE", f"{namespace}_USER", secrets.token_urlsafe(24)
        self._execute(
            "USE ROLE SECURITYADMIN",
            f"CREATE ROLE {role}",
            f"CREATE USER {user} PASSWORD = '{password}' DEFAULT_ROLE = {role} TYPE = LEGACY_SERVICE",
            f"GRANT ROLE {role} TO USER {user}",
            f"GRANT ROLE {role} TO ROLE SYSADMIN",  # so the admin connection can grade and clean up
            "USE ROLE SYSADMIN",
            f"GRANT USAGE ON WAREHOUSE {WAREHOUSE} TO ROLE {role}",
            # A snapshot clone is zero-copy: instant, and independent of the snapshot afterwards.
            f"CREATE OR REPLACE DATABASE {namespace}" + (f" CLONE {snapshot}" if snapshot else ""),
            *self._hand_over(namespace),
        )
        return {"database": namespace, "username": user, "password": password, "role": role}

    def drop_namespace(self, namespace: str) -> None:
        self._execute(
            f"DROP DATABASE IF EXISTS {namespace}",
            "USE ROLE SECURITYADMIN",
            f"DROP USER IF EXISTS {namespace}_USER",
            f"DROP ROLE IF EXISTS {namespace}_ROLE",
        )

    def reset_namespace(self, namespace: str, workspace: Path, tables: list[str], snapshot: str | None = None) -> None:
        source = snapshot or f"{namespace}__OLD"
        self._execute(
            *([] if snapshot else [f"ALTER DATABASE {namespace} RENAME TO {source}"]),
            *self._copy_tables(source, namespace, tables),
            *self._hand_over(namespace),
            *([] if snapshot else [f"DROP DATABASE {source}"]),
        )

    def save_snapshot(self, namespace: str, workspace: Path, snapshot: str, tables: list[str]) -> None:
        # Copy only the source tables, so a snapshot never carries a previous rollout's dbt models.
        # The admin role owns it, so rollouts can't read or change it.
        self._execute(*self._copy_tables(namespace, snapshot, tables))

    @staticmethod
    def _copy_tables(source: str, target: str, tables: list[str]) -> list[str]:
        return [
            f"CREATE OR REPLACE DATABASE {target}",
            f"CREATE SCHEMA {target}.AIRBYTE_SCHEMA",
            *[f"CREATE TABLE {target}.AIRBYTE_SCHEMA.{t} CLONE {source}.AIRBYTE_SCHEMA.{t}" for t in tables],
        ]

    @staticmethod
    def _hand_over(namespace: str) -> list[str]:
        """Give the rollout's role ownership of everything in its database."""
        role = f"{namespace}_ROLE"
        return [
            f"GRANT OWNERSHIP ON DATABASE {namespace} TO ROLE {role} COPY CURRENT GRANTS",
            f"GRANT OWNERSHIP ON ALL SCHEMAS IN DATABASE {namespace} TO ROLE {role} COPY CURRENT GRANTS",
            f"GRANT OWNERSHIP ON ALL TABLES IN DATABASE {namespace} TO ROLE {role} COPY CURRENT GRANTS",
        ]

    def _execute(self, *statements: str) -> None:
        connector = self.connector()
        try:
            cursor = connector.conn().cursor()
            for statement in statements:
                cursor.execute(statement)
        finally:
            connector.close()


def get_destination(name: str, credential_path: str | None = None) -> Destination:
    if name == "duckdb":
        return DuckDBDestination()
    if name == "snowflake":
        if credential_path is None:
            raise ValueError("snowflake needs credential_path (ELT-Bench setup/destination/snowflake_credential.json)")
        return SnowflakeDestination(credential_path)
    raise ValueError(f"Unknown destination {name!r}; expected 'duckdb' or 'snowflake'")
