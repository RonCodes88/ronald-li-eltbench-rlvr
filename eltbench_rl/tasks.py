"""ELT-Bench tasks: the input bundle the agent sees plus the hidden evaluation data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from common import warehouse
from eva_stage2 import load_sort_keys
from huggingface_hub import snapshot_download

GT_DATASET = "tttjjj/elt_bench"

# Prompt text adapted from ELT-Bench agents/codex_agent/run.py.
ELT_BENCH_GOAL = """Your goal is to build an ELT pipeline by extracting data from multiple sources, such as custom APIs, PostgreSQL, MongoDB, flat files, and the cloud service S3.
The extracted data will be loaded into a target system, Snowflake, followed by writing transformation queries to construct final tables for downstream use.
This task is divided into two stages:
1. Data Extraction and Loading – Using Airbyte's Terraform provider.
2. Data Transformation – Using the DBT Project workflow for Snowflake.
"""

TASK_INSTRUCTION = """We're currently building the ELT pipeline using Airbyte Terraform and DBT. Here's the detailed instruction:

INSTRUCTIONS:
 # Stage 1: Data Extraction and Loading Hints#
1. Initialize the Airbyte Provider: Use /workspace/config.yaml to configure the username, password, and server URL in /workspace/elt/main.tf. You can refer to /workspace/documentation/airbyte_Provider.md, then run `terraform init`.
• Important: You must not modify any provided code in /workspace/elt/main.tf.
2. Configure Sources and Destinations: Use /workspace/config.yaml to set up the listed sources and destinations in in Terraform file located in /workspace/elt. The values in config.yaml represent the actual configurations required for the project.
3. Refer to Documentation for Configuration Guidance: You must consult the Airbyte documentation located in /workspace/documentation to understand how to configure sources and destinations.
• Important: The values in the documentation are for reference only. DO NOT use them directly. And you must not fill the fields with random values or placeholders. Instead, fill in the fields based on the field names in /workspace/documentation and the realistic values specified in /workspace/config.yaml. The field names in /workspace/documentation and /workspace/config.yaml may not match exactly. In such cases, you need to infer values for certain fields. If you forget the realistic values for configuring sources and destinations in Terraform file located in /workspace/elt, you must go back and read /workspace/config.yaml to retrieve them.
4. Establish Connections: After creating sources and destinations, establish connections between them by following /workspace/documentation/connection.md. For each connection, include the name and sync_mode fields in configuration.streams to streamline only the tables defined in /workspace/config.yaml.
5. Apply the Configuration: Once all necessary configurations are written, run `terraform apply` to create the resources.
6. Retrieve Connection IDs: After successfully creating sources, destinations and connections, obtain the connection IDs for all newly created connections from the terraform.tfstate file, as these will be needed to trigger the jobs.
7. Trigger Sync Jobs: For each connection ID, trigger the job using the Airbyte API (refer to /workspace/documentation/trigger_job.md and use /workspace/config.yaml to retrieve the required values).
8. Monitor Job Status: After all jobs have been triggered, check the status of all jobs by running `python /workspace/check_job_status.py --server <server> --username <username> --password <password>` (replace <server>, <username>, and <password> with the actual values). Keep checking until all jobs have completed. Proceed to the next stage only if all data extraction and loading jobs are successful. If any job fails, fix it and retry before moving on.
9. Error Handling: If an error occurs, DO NOT modify any provided files. Instead, verify the configuration against the provided documentation and review previous steps to identify and resolve the issue. If the issue persists after multiple retries, terminate the task.

#  Stage 2: Data Transformation Hints#
1. Initialize the DBT Project: Set up a new DBT project by configuring it with /workspace/config.yaml, and remove the example directory under the models directory.
2. Understand the Data Model: Review data_model.yaml in /workspace to understand the required data models and their column descriptions. Then, write SQL queries to generate these defined data models, referring to the files in the /workspace/schemas directory to understand the schemas of source tables.
• Important: Write a separate query for each data model, and if using any DBT project variables, ensure they have already been declared.
3. Validate Table Locations: Ensure all SQL queries reference the correct database and schema names for source tables. If you encounter a "table not found" error, refer to /workspace/config.yaml to obtain the correct configuration. If the table does not exist, the issue is likely due to a failure in data extraction and loading in Stage 1, and you should return to Stage 1 to resolve it.
4. Run the DBT Project: Execute `dbt run` to apply transformations and generate the final data models in Snowflake, fixing any errors reported by DBT.
5. Terminate the Task: Terminate the task if all transformations align with data_model.yaml and the final tables in Snowflake are accurate and verified. Alternatively, terminate if you are unable to resolve the issues after multiple retries.
"""


@dataclass(frozen=True)
class EltTask:
    name: str
    input_dir: Path  # bundle copied into /workspace: config.yaml, data_model.yaml, schemas/, ...
    gt_dir: Path  # <table>.csv ground truth
    sql_dir: Path | None  # evaluation queries; None means SELECT * FROM <table>
    source_tables: dict[str, int]  # expected row count of every loaded source table
    sort_keys: dict[str, list[str]]
    instruction: str
    uses_airbyte: bool


def elt_bench_instruction(name: str, destination: str) -> str:
    """The user prompt ELT-Bench's own agent runners send, adapted to the destination."""
    display_name = warehouse.get_destination(destination).display_name
    return (
        f"Build the ELT pipeline for the `{name}` now. All configuration values "
        f"(including the {display_name} namespace) must come from /workspace/config.yaml.\n\n"
        + warehouse.adapt_prompt(ELT_BENCH_GOAL + "\n" + TASK_INSTRUCTION, destination)
    )


def load_tasks(bench_dir: Path, destination: str, names: list[str] | None = None) -> list[EltTask]:
    """Load tasks from an ELT-Bench style directory (the ELT-Bench checkout or the toy bench)."""
    bench_dir = Path(bench_dir)
    inputs_dir = bench_dir / ("inputs" if destination == "snowflake" else f"inputs_{destination}")
    row_counts = json.loads((bench_dir / "evaluation" / "table.json").read_text())
    sort_keys = load_sort_keys(str(bench_dir / "evaluation" / "sort_key.json"))
    names = names or sorted(p.name for p in inputs_dir.iterdir() if p.is_dir())

    tasks = []
    for name in names:
        input_dir = inputs_dir / name
        config = yaml.safe_load((input_dir / "config.yaml").read_text())
        instruction_path = input_dir / "instruction.md"
        sql_dir = bench_dir / "evaluation" / "sql" / name
        tasks.append(
            EltTask(
                name=name,
                input_dir=input_dir,
                gt_dir=ground_truth_dir(bench_dir, destination, name),
                sql_dir=sql_dir if sql_dir.is_dir() else None,
                source_tables=row_counts[name],
                sort_keys=sort_keys.get(name, {}),
                instruction=instruction_path.read_text()
                if instruction_path.exists()
                else elt_bench_instruction(name, destination),
                uses_airbyte="Airbyte" in config,
            )
        )
    return tasks


def ground_truth_dir(bench_dir: Path, destination: str, name: str) -> Path:
    """Return the GT directory for a task, downloading official GT from HuggingFace if missing."""
    gt_dir = bench_dir / "ground_truth" / f"gt_{destination}" / name
    if not gt_dir.is_dir():
        snapshot_download(
            repo_id=GT_DATASET,
            repo_type="dataset",
            allow_patterns=[f"gt_{destination}/{name}/*"],
            local_dir=bench_dir / "ground_truth",
        )
    return gt_dir
