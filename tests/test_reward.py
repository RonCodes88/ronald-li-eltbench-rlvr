import dataclasses
from pathlib import Path

import duckdb
import pytest

from eltbench_rl.destinations import DuckDBDestination
from eltbench_rl.reward import grade, model_outputs, rewrite_namespace
from eltbench_rl.tasks import load_tasks

TOY_BENCH = Path(__file__).parent / "fixtures" / "toy_bench"
SOLUTION = TOY_BENCH / "solution"
NAMESPACE = "shop__test"
MODELS = {"customer_orders", "daily_revenue"}
[TASK] = load_tasks(TOY_BENCH, "duckdb")


def render(sql: str) -> str:
    return sql.replace("{{ target.schema }}", NAMESPACE).replace("/workspace/", f"{TASK.input_dir}/")


def build_warehouse(workspace: Path, models: dict[str, str] | None = None, after_load: str = "") -> None:
    """Leave the warehouse the way a rollout would: sources loaded, then one table per model."""
    models = models or {name: (SOLUTION / f"{name}.sql").read_text() for name in MODELS}
    con = duckdb.connect(str(workspace / "warehouse.duckdb"))
    con.execute(render((SOLUTION / "load.sql").read_text()))
    if after_load:
        con.execute(render(after_load))
    for name, sql in models.items():
        con.execute(f"create table {NAMESPACE}.{name} as {render(sql)}")
    con.close()


def run_grade(workspace: Path, task=TASK, built_models=MODELS, has_seeds=False, airbyte_tables=None):
    return grade(task, DuckDBDestination().connector(workspace), NAMESPACE, built_models, has_seeds, airbyte_tables)


def test_reference_solution_scores_one(tmp_path):
    build_warehouse(tmp_path)
    reward, metrics = run_grade(tmp_path)

    assert reward == pytest.approx(1.0)
    assert metrics["srdel"] == 1.0 and metrics["srdt"] == 1.0 and metrics["pipeline_success"] == 1.0


def test_wrong_column_gets_partial_credit(tmp_path):
    sql = (SOLUTION / "customer_orders.sql").read_text().replace(" and o.status <> 'cancelled'", "")
    build_warehouse(tmp_path, {"customer_orders": sql, "daily_revenue": (SOLUTION / "daily_revenue.sql").read_text()})
    reward, metrics = run_grade(tmp_path)

    assert 0.2 < reward < 1.0
    assert metrics["srdt"] == 0.5


def test_missing_rows_fail_stage_one(tmp_path):
    build_warehouse(tmp_path, after_load="delete from {{ target.schema }}.orders where order_id = 12;")
    reward, metrics = run_grade(tmp_path)

    assert metrics["el_frac"] == 0.5 and metrics["srdel"] == 0.0
    assert reward == pytest.approx(0.2 * 0.5)


def test_tables_not_built_by_dbt_earn_nothing(tmp_path):
    build_warehouse(tmp_path)  # daily_revenue exists, but dbt did not build it
    reward, metrics = run_grade(tmp_path, built_models={"customer_orders"})

    assert metrics["built_frac"] == 0.5 and metrics["col_frac"] == 0.5
    assert reward == pytest.approx(0.2 + 0.1 * 0.5 + 0.7 * 0.5)


def test_seeds_get_no_stage_two_credit(tmp_path):
    build_warehouse(tmp_path)
    reward, _ = run_grade(tmp_path, has_seeds=True)

    assert reward == pytest.approx(0.2)


def test_airbyte_tasks_require_airbyte_columns(tmp_path):
    airbyte_task = dataclasses.replace(TASK, uses_airbyte=True)
    build_warehouse(tmp_path)
    _, metrics = run_grade(tmp_path, task=airbyte_task)
    assert metrics["srdel"] == 0.0

    con = duckdb.connect(str(tmp_path / "warehouse.duckdb"))
    for table in TASK.source_tables:
        con.execute(f"alter table {NAMESPACE}.{table} add column _airbyte_raw_id varchar")
    con.close()
    _, metrics = run_grade(tmp_path, task=airbyte_task)
    assert metrics["srdel"] == 1.0


def test_source_tables_need_exactly_their_schema_columns(tmp_path):
    build_warehouse(tmp_path, after_load="alter table {{ target.schema }}.orders add column answer varchar;")
    _, metrics = run_grade(tmp_path)

    assert metrics["el_frac"] == 0.5 and metrics["srdel"] == 0.0


def test_tables_airbyte_did_not_sync_fail_stage_one(tmp_path):
    build_warehouse(tmp_path)
    _, metrics = run_grade(tmp_path, airbyte_tables={"customers", "orders"})
    assert metrics["srdel"] == 1.0

    _, metrics = run_grade(tmp_path, airbyte_tables={"customers"})
    assert metrics["el_frac"] == 0.5 and metrics["srdel"] == 0.0


def test_tables_that_ignore_their_sources_count_as_hardcoded(tmp_path):
    build_warehouse(tmp_path)
    connector = DuckDBDestination().connector(tmp_path)
    outputs = model_outputs(TASK, connector, NAMESPACE, MODELS)

    # As if the run on empty sources had returned exactly these rows too.
    reward, metrics = grade(TASK, connector, NAMESPACE, MODELS, False, empty_outputs=outputs)
    assert reward == pytest.approx(0.2)
    assert metrics["hardcoded_frac"] == 1.0 and metrics["built_frac"] == 0.0

    no_rows = {name: df.iloc[:0] for name, df in outputs.items()}
    reward, metrics = grade(TASK, connector, NAMESPACE, MODELS, False, empty_outputs=no_rows)
    assert reward == pytest.approx(1.0) and metrics["hardcoded_frac"] == 0.0


def test_empty_warehouse_scores_zero(tmp_path):
    reward, metrics = run_grade(tmp_path, built_models=set())

    assert reward == 0.0 and metrics["el_frac"] == 0.0


def test_rewrite_namespace():
    sql = "SELECT * FROM shop.daily_revenue JOIN shopping.x ON 1=1"
    assert rewrite_namespace(sql, "shop", NAMESPACE) == (
        f"SELECT * FROM {NAMESPACE}.daily_revenue JOIN shopping.x ON 1=1"
    )
