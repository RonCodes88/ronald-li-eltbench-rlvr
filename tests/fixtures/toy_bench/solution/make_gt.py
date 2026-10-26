"""Regenerate the toy ground truth by running the reference solution in DuckDB."""

from pathlib import Path

import duckdb

BENCH_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BENCH_DIR / "inputs_duckdb" / "shop"
GT_DIR = BENCH_DIR / "ground_truth" / "gt_duckdb" / "shop"
MODELS = ["customer_orders", "daily_revenue"]


def render(sql: str) -> str:
    return sql.replace("{{ target.schema }}", "shop").replace("/workspace/", f"{INPUT_DIR}/")


def main() -> None:
    con = duckdb.connect()
    con.execute(render((BENCH_DIR / "solution" / "load.sql").read_text()))
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        model_sql = render((BENCH_DIR / "solution" / f"{model}.sql").read_text())
        con.execute(f"create table shop.{model} as {model_sql}")
        eval_sql = (BENCH_DIR / "evaluation" / "sql" / "shop" / f"{model}.sql").read_text()
        con.execute(eval_sql).df().to_csv(GT_DIR / f"{model}.csv", index=False)


if __name__ == "__main__":
    main()
