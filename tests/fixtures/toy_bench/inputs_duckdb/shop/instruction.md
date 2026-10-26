We're building a small ELT pipeline into a local DuckDB warehouse. Here's the detailed instruction:

# Stage 1: Data Extraction and Loading #
1. Read /workspace/config.yaml. Each entry under `flat_files` is a source file, and the `duckdb` section gives the warehouse file (`path`) and the schema (`schema`) to load into.
2. Load every source into that schema as a table named after its `table` field, for example with the `duckdb` Python package (`read_csv_auto` / `read_json_auto`). Load each source exactly once.

# Stage 2: Data Transformation #
1. Create a DBT project under /workspace/elt that uses the duckdb adapter, with its profile pointing at the `path` and `schema` from /workspace/config.yaml.
2. Read /workspace/data_model.yaml and /workspace/schemas, then write one model per data model, materialized as a table, with exactly the listed columns.
3. Run `dbt run` and fix any errors it reports.

When the final tables are built and verified, reply without calling a tool.
