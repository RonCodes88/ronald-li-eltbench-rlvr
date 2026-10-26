"""Execution-based reward: grade the warehouse state a rollout leaves behind, as ELT-Bench does."""

from __future__ import annotations

import functools
import io
import re
from pathlib import Path

import pandas as pd
from db_connectors import BaseConnector
from eva_stage2 import check_corretness, sort_by_keys

from eltbench_rl.tasks import EltTask

EL_WEIGHT, BUILT_WEIGHT, COLUMN_WEIGHT = 0.2, 0.1, 0.7
AIRBYTE_COLUMN = "_airbyte_raw_id"


def grade(
    task: EltTask,
    connector: BaseConnector,
    namespace: str,
    built_models: set[str],
    has_seeds: bool,
) -> tuple[float, dict[str, float]]:
    """Score the tables in `namespace` against the task's hidden evaluation data.

    Stage 1 checks every source table was loaded with the right row count. Stage 2 only counts
    once stage 1 passes (ELT-Bench's rule), and only for tables the re-executed dbt project built,
    so hand-made or stale tables earn nothing.
    """
    loaded = [
        _source_table_ok(task, connector, namespace, table, rows)
        for table, rows in task.source_tables.items()
    ]
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
