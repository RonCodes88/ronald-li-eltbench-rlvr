from pathlib import Path

import pandas as pd

from eltbench_rl import ELT_BENCH_DIR
from eltbench_rl.tasks import elt_bench_instruction, ground_truth_dir, load_tasks

TOY_BENCH = Path(__file__).parent / "fixtures" / "toy_bench"


def test_load_toy_task():
    [task] = load_tasks(TOY_BENCH, "duckdb")

    assert task.name == "shop"
    assert task.source_tables == {"customers": 6, "orders": 12}
    assert task.sort_keys == {"customer_orders": ["customer_id"], "daily_revenue": ["order_date"]}
    assert task.uses_airbyte is False
    assert task.instruction.startswith("We're building a small ELT pipeline")
    assert sorted(p.name for p in task.gt_dir.glob("*.csv")) == [
        "customer_orders.csv",
        "daily_revenue.csv",
    ]


def test_toy_row_counts_match_data():
    [task] = load_tasks(TOY_BENCH, "duckdb")
    data_dir = task.input_dir / "data"

    assert len(pd.read_csv(data_dir / "customers.csv")) == task.source_tables["customers"]
    assert len(pd.read_json(data_dir / "orders.jsonl", lines=True)) == task.source_tables["orders"]


def test_elt_bench_instruction():
    text = elt_bench_instruction("trains", "snowflake")

    assert text.startswith("Build the ELT pipeline for the `trains` now.")
    assert "Airbyte" in text and "Snowflake" in text


def test_official_ground_truth_download():
    gt = pd.read_csv(ground_truth_dir(ELT_BENCH_DIR, "snowflake", "trains") / "trains.csv")

    assert list(gt.columns)[0] == "TRAIN_ID"
    assert len(gt) == 20
