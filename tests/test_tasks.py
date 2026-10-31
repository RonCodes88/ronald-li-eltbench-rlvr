import asyncio
from pathlib import Path

import pandas as pd

from eltbench_rl import ELT_BENCH_DIR
from eltbench_rl.env import EltDatasetBuilder
from eltbench_rl.tasks import elt_bench_instruction, ground_truth_dir, load_tasks, transform_instruction

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


def test_transform_instruction_keeps_only_stage_two():
    [task] = load_tasks(TOY_BENCH, "duckdb")
    for text in (task.instruction, elt_bench_instruction("trains", "snowflake")):
        stage_two = transform_instruction(text)

        assert "# Stage 1" not in stage_two and "terraform init" not in stage_two
        assert "# Stage 2" in stage_two and "dbt run" in stage_two.lower()


def test_official_ground_truth_download():
    gt = pd.read_csv(ground_truth_dir(ELT_BENCH_DIR, "snowflake", "trains") / "trains.csv")

    assert list(gt.columns)[0] == "TRAIN_ID"
    assert len(gt) == 20


def test_each_epoch_gets_its_own_builders():
    builder = EltDatasetBuilder(
        bench_dir=str(TOY_BENCH), destination="duckdb", batch_size=2, group_size=4,
        model_name="Qwen/Qwen3.6-35B-A3B", num_epochs=2,
    )  # fmt: skip
    dataset, _ = asyncio.run(builder())
    [batch] = [dataset.get_batch(i) for i in range(len(dataset))]

    # Both groups of `shop` run at once, so each needs its own builder to clean up after it.
    assert len(batch) == 2 and batch[0] is not batch[1]
