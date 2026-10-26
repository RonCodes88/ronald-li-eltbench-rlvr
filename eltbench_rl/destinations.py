"""Warehouses the environment can load into and grade against.

A destination says which config.yaml field holds the per-rollout namespace, how to create and drop
that namespace, and which ELT-Bench connector reads it back for grading. Supporting another
warehouse means adding one class here; the environment itself does not change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import duckdb
from db_connectors import BaseConnector, SnowflakeConnector


class Destination(Protocol):
    name: str  # key of the destination block in config.yaml
    namespace_field: str  # field of that block's config holding the per-rollout namespace

    def connector(self, workspace: Path) -> BaseConnector: ...

    def create_namespace(self, namespace: str) -> None: ...

    def drop_namespace(self, namespace: str) -> None: ...


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

    def connector(self, workspace: Path) -> BaseConnector:
        return DuckDBConnector({"path": str(workspace / "warehouse.duckdb")})

    def create_namespace(self, namespace: str) -> None:
        pass  # every rollout has its own database file

    def drop_namespace(self, namespace: str) -> None:
        pass


class SnowflakeDestination:
    """Snowflake laid out like ELT-Bench: one database per namespace, tables in AIRBYTE_SCHEMA."""

    name = "snowflake"
    namespace_field = "database"

    def __init__(self, credential_path: str):
        self.credential_path = credential_path

    def connector(self, workspace: Path | None = None) -> BaseConnector:
        return SnowflakeConnector(json.loads(Path(self.credential_path).read_text()))

    def create_namespace(self, namespace: str) -> None:
        # Same as ELT-Bench's example/create_snowflake_db.py: the Airbyte role owns the database.
        self._execute(
            f"CREATE OR REPLACE DATABASE {namespace}",
            f"GRANT OWNERSHIP ON DATABASE {namespace} TO ROLE AIRBYTE_ROLE",
        )

    def drop_namespace(self, namespace: str) -> None:
        self._execute(f"DROP DATABASE IF EXISTS {namespace}")

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
