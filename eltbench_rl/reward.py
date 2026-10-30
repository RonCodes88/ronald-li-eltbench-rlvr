"""Execution-based reward: grade the warehouse state a rollout leaves behind, as ELT-Bench does."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
from db_connectors import BaseConnector
from eva_stage2 import check_corretness, sort_by_keys
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxInterface

from eltbench_rl import airbyte
from eltbench_rl.destinations import Destination, rollout_snapshot
from eltbench_rl.tasks import EltTask

logger = logging.getLogger(__name__)

EL_WEIGHT, BUILT_WEIGHT, COLUMN_WEIGHT = 0.2, 0.1, 0.7
AIRBYTE_COLUMN = "_airbyte_raw_id"


@dataclass
class EltReward:
    """Grade a finished rollout: re-run the submitted dbt project, then `score` what it built.

    Before each re-run, the namespace is rebuilt from a snapshot of just the source tables, so dbt
    has to compute every model from the sources: tables the agent made by hand, staged files and
    anything else it left in the warehouse are gone. The snapshot is the task's reference load when
    there is one, so source values the agent edited after loading don't count either. If any table
    matches the ground truth, dbt runs once more on empty copies of the sources: a model that
    returns the same rows without any data has its answers hardcoded.

    Same shape as the cookbook's HarborReward: called once at episode end, before the sandbox goes away.
    """

    task: EltTask
    destination: Destination
    sandbox: SandboxInterface
    workspace: Path  # host directory mounted at /workspace
    namespace: str
    airbyte_workspace: str | None  # check its sync jobs in stage 1; None when there is no Airbyte stage
    snapshot: str | None  # the task's reference load of its sources; None grades from this rollout's own load
    save_snapshot: bool = False  # make this rollout's load the reference (only for a trusted run, like the oracle)
    dbt_timeout: int = 900

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        start = time.monotonic()
        try:
            airbyte_tables = None
            if self.airbyte_workspace:
                airbyte_tables = await asyncio.to_thread(
                    airbyte.synced_tables, self.airbyte_workspace, self.destination.namespace_field, self.namespace
                )
            loaded = await asyncio.to_thread(self._with_connector, sources_loaded, airbyte_tables)
            built_models, has_seeds, dbt_ok, outputs = set(), False, False, {}
            # Stage 2 only counts once stage 1 passes, so only then are the dbt re-runs worth doing.
            if all(loaded):
                snapshot = await asyncio.to_thread(self._snapshot_sources)
                await self._reset(snapshot)
                built_models, has_seeds, dbt_ok = await self._rerun_dbt()
                outputs = await asyncio.to_thread(self._with_connector, model_outputs, built_models)
            reward, metrics = score(self.task, loaded, built_models, has_seeds, outputs)
            if metrics["col_frac"] > 0:
                await self._reset(snapshot, empty=True)
                empty_built, _, _ = await self._rerun_dbt()
                empty_outputs = await asyncio.to_thread(self._with_connector, model_outputs, empty_built)
                reward, metrics = score(self.task, loaded, built_models, has_seeds, outputs, empty_outputs)
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

    def _snapshot_sources(self) -> str:
        """The snapshot the re-runs start from: the task's reference load if there is one, else this rollout's."""
        tables = list(self.task.source_tables)
        if self.save_snapshot:
            self.destination.save_snapshot(self.namespace, self.workspace, self.snapshot, tables)
        if self.snapshot:
            return self.snapshot
        own = rollout_snapshot(self.namespace)
        self.destination.save_snapshot(self.namespace, self.workspace, own, tables)
        return own

    async def _reset(self, snapshot: str, empty: bool = False) -> None:
        tables = list(self.task.source_tables)
        await asyncio.to_thread(self.destination.reset_namespace, self.namespace, self.workspace, snapshot, tables, empty)

    def _with_connector(self, fn, *args):
        connector = self.destination.connector(self.workspace)
        try:
            return fn(self.task, connector, self.namespace, *args)
        finally:
            connector.close()


def grade(
    task: EltTask,
    connector: BaseConnector,
    namespace: str,
    built_models: set[str],
    has_seeds: bool,
    airbyte_tables: set[str] | None = None,
    empty_outputs: dict[str, pd.DataFrame] | None = None,
) -> tuple[float, dict[str, float]]:
    """Check stage 1 and score the tables in `namespace` in one go (see `score`)."""
    loaded = sources_loaded(task, connector, namespace, airbyte_tables)
    outputs = model_outputs(task, connector, namespace, built_models) if all(loaded) else {}
    return score(task, loaded, built_models, has_seeds, outputs, empty_outputs)


def score(
    task: EltTask,
    loaded: list[bool],
    built_models: set[str],
    has_seeds: bool,
    outputs: dict[str, pd.DataFrame],
    empty_outputs: dict[str, pd.DataFrame] | None = None,
) -> tuple[float, dict[str, float]]:
    """Score a rollout against the task's hidden evaluation data.

    `loaded` is stage 1 per source table (`sources_loaded`). Stage 2 only counts once stage 1
    passes (ELT-Bench's rule), and only for tables the re-executed dbt project built, so hand-made
    or stale tables earn nothing. `outputs` are those tables as `model_outputs` read them. A table
    whose rows are also in `empty_outputs`, what the same model returned when every source table
    was empty, doesn't depend on the sources: it is hardcoded and counts as not built.
    """
    el_frac = sum(loaded) / len(loaded)
    srdel = float(all(loaded))

    gt_tables = ground_truth_tables(task)
    if srdel and not has_seeds:
        empty_outputs = empty_outputs or {}
        hardcoded = {
            table
            for table, df in outputs.items()
            if len(df) and table in empty_outputs and _rows(df) == _rows(empty_outputs[table])
        }
        built = [table.lower() in built_models and table not in hardcoded for table in gt_tables]
        col_fracs = [
            _column_match(task, table, outputs[table]) if ok and table in outputs else 0.0
            for table, ok in zip(gt_tables, built)
        ]
    else:
        hardcoded, built, col_fracs = set(), [False] * len(gt_tables), [0.0] * len(gt_tables)

    built_frac = sum(built) / len(built)
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
        "hardcoded_frac": len(hardcoded) / len(gt_tables),
    }


def model_outputs(
    task: EltTask, connector: BaseConnector, namespace: str, built_models: set[str]
) -> dict[str, pd.DataFrame]:
    """Read each ground-truth table that dbt built with ELT-Bench's evaluation SQL; skip ones that fail."""
    outputs = {}
    for table in ground_truth_tables(task):
        if table.lower() not in built_models:
            continue
        sql_path = task.sql_dir / f"{table}.sql" if task.sql_dir else None
        sql = sql_path.read_text() if sql_path and sql_path.exists() else f"SELECT * FROM {task.name}.{table}"
        try:
            df = connector.query_evaluation_sql(rewrite_namespace(sql, task.name, namespace), namespace)
        except Exception:
            continue
        # Same CSV round trip as ELT-Bench's eva_stage2, so values compare exactly like the benchmark.
        outputs[table] = pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=True)
    return outputs


def ground_truth_tables(task: EltTask) -> list[str]:
    return sorted(path.stem for path in task.gt_dir.glob("*.csv"))


def sources_loaded(
    task: EltTask, connector: BaseConnector, namespace: str, airbyte_tables: set[str] | None = None
) -> list[bool]:
    """Stage 1, per source table: loaded with the right rows and columns (by Airbyte, when it's checked)."""
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
        fqn = connector._fqn(namespace, namespace, table)
        columns = {column.lower() for column in connector.query(f"SELECT * FROM {fqn} LIMIT 0").columns}
    except Exception:
        return False
    # Tables loaded by Airbyte carry its metadata columns; loading around Airbyte does not count.
    if task.uses_airbyte and AIRBYTE_COLUMN not in columns:
        return False
    # Apart from Airbyte's own columns, exactly the ones in the task's schema file, so a source can't
    # be swapped for another file with the same row count (like the public ground truth).
    expected = _schema_columns(task.input_dir / "schemas" / f"{table}.csv")
    data_columns = {column for column in columns if not column.startswith(("_airbyte_", "_ab_"))}
    return expected is None or data_columns == expected


def _rows(df: pd.DataFrame) -> list[tuple]:
    return sorted(df.astype(str).itertuples(index=False, name=None))


def _column_match(task: EltTask, table: str, df: pd.DataFrame) -> float:
    """Fraction of ground-truth columns the table reproduces, using ELT-Bench's comparator."""
    df_gt = _read_ground_truth(task.gt_dir / f"{table}.csv")
    keys = task.sort_keys.get(table, [])
    result = check_corretness(sort_by_keys(df_gt, keys), sort_by_keys(df, keys))
    return len(result["matched"]) / len(df_gt.columns)


@functools.cache
def _schema_columns(path: Path) -> frozenset[str] | None:
    if not path.exists():
        return None
    return frozenset(pd.read_csv(path, dtype=str)["column_name"].str.lower())


@functools.cache
def _read_ground_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True)
