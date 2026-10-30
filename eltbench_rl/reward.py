"""Execution-based reward: grade the warehouse state a rollout leaves behind, as ELT-Bench does."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml
from db_connectors import BaseConnector
from eva_stage2 import check_corretness, sort_by_keys
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxInterface

from eltbench_rl.destinations import Destination
from eltbench_rl.tasks import EltTask

logger = logging.getLogger(__name__)

EL_WEIGHT, BUILT_WEIGHT, COLUMN_WEIGHT = 0.2, 0.1, 0.7
AIRBYTE_COLUMN = "_airbyte_raw_id"
# Airbyte's public API as the host sees it (abctl local install --port 8001).
AIRBYTE_API_URL = "http://localhost:8001/api/public/v1"


@dataclass
class EltReward:
    """Grade a finished rollout: re-run the submitted dbt project, then `grade` the warehouse.

    Before the re-run, the namespace is rebuilt with only the source tables, so dbt has to compute
    every model from the sources: tables the agent made by hand, staged files and anything else it
    left in the warehouse are gone. Same shape as the cookbook's HarborReward: called once at
    episode end, before the sandbox goes away.
    """

    task: EltTask
    destination: Destination
    sandbox: SandboxInterface
    workspace: Path  # host directory mounted at /workspace
    namespace: str
    check_airbyte_jobs: bool  # False when the sources came from a snapshot instead of Airbyte
    restore_from: str | None = None  # snapshot the sources came from; the re-run starts from it again
    save_snapshot_as: str | None = None  # keep a copy of the sources once they pass stage 1
    dbt_timeout: int = 900

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        start = time.monotonic()
        try:
            airbyte_tables = await asyncio.to_thread(self._airbyte_tables) if self.check_airbyte_jobs else None
            built_models, has_seeds, dbt_ok = set(), False, False
            # Stage 2 only counts once stage 1 passes, so only then is the dbt re-run worth doing.
            if all(await asyncio.to_thread(self._with_connector, sources_loaded, airbyte_tables)):
                tables = list(self.task.source_tables)
                await asyncio.to_thread(
                    self.destination.reset_namespace, self.namespace, self.workspace, tables, self.restore_from
                )
                built_models, has_seeds, dbt_ok = await self._rerun_dbt()
            reward, metrics = await asyncio.to_thread(
                self._with_connector, grade, built_models, has_seeds, airbyte_tables
            )
            if self.save_snapshot_as and metrics["srdel"] == 1.0:
                await asyncio.to_thread(
                    self.destination.save_snapshot,
                    self.namespace, self.workspace, self.save_snapshot_as, list(self.task.source_tables),
                )  # fmt: skip
        except Exception as e:
            logger.error("ELT-Bench grading failed for %s: %s", self.namespace, e)
            return 0.0, {"grading_error": 1.0}
        metrics["dbt_rerun_ok"] = float(dbt_ok)
        metrics["time/grade_s"] = time.monotonic() - start
        return reward, metrics

    async def _rerun_dbt(self) -> tuple[set[str], bool, bool]:
        """Run the agent's dbt project again from its final state; return (built models, has seeds, ok)."""
        found = await self.sandbox.run_command(
            "find /workspace/elt -name dbt_project.yml -not -path '*/dbt_packages/*'"
        )
        if not found.stdout.strip():
            return set(), False, False
        project = PurePosixPath(found.stdout.split()[0]).parent
        # Delete the artifacts the agent left, so only our own run is trusted.
        result = await self.sandbox.run_command(
            f"cd {project} && rm -f target/run_results.json target/manifest.json && dbt run",
            timeout=self.dbt_timeout,
        )

        target = self.workspace / project.relative_to("/workspace") / "target"
        built = set()
        if (target / "run_results.json").exists():
            for node in json.loads((target / "run_results.json").read_text())["results"]:
                if node["unique_id"].startswith("model.") and node["status"] == "success":
                    built.add(node["unique_id"].split(".", 2)[2].lower())
        # Seeds would let the agent hardcode results instead of computing them from the sources.
        # Ask dbt's own manifest, which finds seeds wherever `seed-paths` points.
        has_seeds = (target / "manifest.json").exists() and any(
            node["resource_type"] == "seed"
            for node in json.loads((target / "manifest.json").read_text())["nodes"].values()
        )
        return built, has_seeds, result.exit_code == 0

    def _with_connector(self, fn, *args):
        connector = self.destination.connector(self.workspace)
        try:
            return fn(self.task, connector, self.namespace, *args)
        finally:
            connector.close()

    def _airbyte_tables(self) -> set[str]:
        workspace_id = yaml.safe_load((self.workspace / "config.yaml").read_text())["Airbyte"]["config"]["workspace_id"]
        return airbyte_synced_tables(workspace_id, self.destination.namespace_field, self.namespace)


def airbyte_synced_tables(workspace_id: str, namespace_field: str, namespace: str) -> set[str]:
    """Streams that Airbyte itself reports as successfully synced into `namespace`.

    Asks Airbyte directly instead of reading anything the agent wrote (like terraform.tfstate),
    so tables loaded around Airbyte, or synced into a sibling rollout's namespace, don't count.
    """

    def get(path: str) -> dict:
        with urllib.request.urlopen(f"{AIRBYTE_API_URL}/{path}", timeout=30) as response:
            return json.load(response)

    tables = set()
    for connection in get(f"connections?workspaceIds={workspace_id}&limit=100")["data"]:
        destination = get(f"destinations/{connection['destinationId']}")
        if str(destination["configuration"].get(namespace_field, "")).lower() != namespace.lower():
            continue
        if get(f"jobs?connectionId={connection['connectionId']}&jobType=sync&status=succeeded")["data"]:
            tables |= {stream["name"].lower() for stream in connection["configurations"]["streams"]}
    return tables


def grade(
    task: EltTask,
    connector: BaseConnector,
    namespace: str,
    built_models: set[str],
    has_seeds: bool,
    airbyte_tables: set[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Score the tables in `namespace` against the task's hidden evaluation data.

    Stage 1 checks every source table was loaded with the right row count (and, when
    `airbyte_tables` is given, by a successful Airbyte sync). Stage 2 only counts
    once stage 1 passes (ELT-Bench's rule), and only for tables the re-executed dbt project built,
    so hand-made or stale tables earn nothing.
    """
    loaded = sources_loaded(task, connector, namespace, airbyte_tables)
    el_frac = sum(loaded) / len(loaded)
    srdel = float(all(loaded))

    gt_tables = sorted(path.stem for path in task.gt_dir.glob("*.csv"))
    built = [table.lower() in built_models for table in gt_tables]
    if srdel and not has_seeds:
        built_frac = sum(built) / len(built)
        col_fracs = [
            _column_match(task, connector, namespace, table) if is_built else 0.0
            for table, is_built in zip(gt_tables, built)
        ]
    else:
        built_frac = 0.0
        col_fracs = [0.0] * len(gt_tables)

    col_frac = sum(col_fracs) / len(col_fracs)
    srdt = sum(frac == 1.0 for frac in col_fracs) / len(col_fracs)
    reward = EL_WEIGHT * el_frac + BUILT_WEIGHT * built_frac + COLUMN_WEIGHT * col_frac
    return reward, {
        "el_frac": el_frac,
        "srdel": srdel,
        "built_frac": built_frac,
        "col_frac": col_frac,
        "srdt": srdt,
        "pipeline_success": float(srdel == 1.0 and srdt == 1.0),
    }


def sources_loaded(
    task: EltTask, connector: BaseConnector, namespace: str, airbyte_tables: set[str] | None = None
) -> list[bool]:
    """Stage 1, per source table: loaded with the right row count (by Airbyte, when it's checked)."""
    return [
        _source_table_ok(task, connector, namespace, table, rows)
        and (airbyte_tables is None or table.lower() in airbyte_tables)
        for table, rows in task.source_tables.items()
    ]


def rewrite_namespace(sql: str, task_name: str, namespace: str) -> str:
    """Point ELT-Bench's `<task>.<table>` references at the rollout's own namespace."""
    return re.sub(rf"\b{re.escape(task_name)}\.", f"{namespace}.", sql, flags=re.IGNORECASE)


def _source_table_ok(
    task: EltTask, connector: BaseConnector, namespace: str, table: str, expected_rows: int
) -> bool:
    # A missing table, schema or database file raises; that just means the table was not loaded.
    try:
        if connector.table_size(namespace, namespace, table) != expected_rows:
            return False
        if task.uses_airbyte:
            # Tables loaded by Airbyte carry its metadata columns; loading around Airbyte does not count.
            fqn = connector._fqn(namespace, namespace, table)
            columns = connector.query(f"SELECT * FROM {fqn} LIMIT 0").columns
            return AIRBYTE_COLUMN in {column.lower() for column in columns}
        return True
    except Exception:
        return False


def _column_match(task: EltTask, connector: BaseConnector, namespace: str, table: str) -> float:
    """Fraction of ground-truth columns the table reproduces, using ELT-Bench's comparator."""
    sql_path = task.sql_dir / f"{table}.sql" if task.sql_dir else None
    sql = sql_path.read_text() if sql_path and sql_path.exists() else f"SELECT * FROM {task.name}.{table}"
    try:
        df = connector.query_evaluation_sql(rewrite_namespace(sql, task.name, namespace), namespace)
    except Exception:
        return 0.0
    # Same CSV round trip as ELT-Bench's eva_stage2, so values compare exactly like the benchmark.
    df = pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=True)
    df_gt = _read_ground_truth(task.gt_dir / f"{table}.csv")
    keys = task.sort_keys.get(table, [])
    result = check_corretness(sort_by_keys(df_gt, keys), sort_by_keys(df, keys))
    return len(result["matched"]) / len(df_gt.columns)


@functools.cache
def _read_ground_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True)
