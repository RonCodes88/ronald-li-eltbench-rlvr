"""Official ELT-Bench task `trains` on real Airbyte + Snowflake, driven by a hand-written oracle.

Needs the official setup from the README (Snowflake credentials, local Airbyte, ELT-Bench sources and
generated inputs); skipped otherwise. The oracle does what an agent should: read /workspace/config.yaml,
configure Airbyte with Terraform, trigger and wait for the syncs, then build the dbt model. It also
saves the snapshot of loaded sources that transform-stage training starts from.
"""

import asyncio
import json
import urllib.request
from pathlib import Path

import pytest
import yaml
from test_local_integration import IMAGE, MODEL, ScriptedPolicy, image_available
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout

from eltbench_rl import ELT_BENCH_DIR
from eltbench_rl.destinations import get_destination
from eltbench_rl.env import EltEnvGroupBuilder
from eltbench_rl.tasks import load_tasks

ORACLE = Path(__file__).parent / "oracles" / "trains"
CREDENTIAL = ELT_BENCH_DIR / "setup" / "destination" / "snowflake_credential.json"
CONFIG = ELT_BENCH_DIR / "inputs" / "trains" / "config.yaml"


def official_setup_ready() -> bool:
    if not (CONFIG.exists() and CREDENTIAL.exists() and image_available()):
        return False
    return bool(json.loads(CREDENTIAL.read_text())["password"])


pytestmark = pytest.mark.skipif(
    not official_setup_ready(), reason="needs the official ELT-Bench setup (see README: Official setup)"
)

TEMPLATES = {
    "/workspace/elt/airbyte.tf": (ORACLE / "airbyte.tf").read_text(),
    "/workspace/elt/trains_dbt/dbt_project.yml": (ORACLE / "dbt_project.yml").read_text(),
    "/workspace/elt/trains_dbt/profiles.yml": (ORACLE / "profiles.yml").read_text(),
    "/workspace/elt/trains_dbt/models/trains.sql": (ORACLE / "models" / "trains.sql").read_text(),
}

# Fill the templates from config.yaml inside the sandbox, so nothing about the rollout is hardcoded.
RENDER = f"""python - <<'PY'
import json, pathlib, string, yaml
config = yaml.safe_load(open('/workspace/config.yaml'))
airbyte, snowflake = config['Airbyte']['config'], config['snowflake']['config']
values = {{**airbyte, **snowflake, 'train_url': config['flat_files'][0]['path'],
          'sf_username': snowflake['username'], 'sf_password': snowflake['password'],
          'username': airbyte['username'], 'password': airbyte['password']}}
for path, template in json.loads({json.dumps(json.dumps(TEMPLATES))}).items():
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(string.Template(template).safe_substitute(values))
PY"""

TRIGGER_SYNCS = """python - <<'PY'
import json, requests, yaml
airbyte = yaml.safe_load(open('/workspace/config.yaml'))['Airbyte']['config']
state = json.load(open('/workspace/elt/terraform.tfstate'))
for resource in state['resources']:
    if resource['type'] == 'airbyte_connection':
        connection_id = resource['instances'][0]['attributes']['connection_id']
        response = requests.post(airbyte['server_url'] + 'jobs', json={'jobType': 'sync', 'connectionId': connection_id},
                                 auth=(airbyte['username'], airbyte['password']))
        print(response.status_code, response.text[:200])
PY"""

WAIT_FOR_SYNCS = """python - <<'PY'
import subprocess, yaml
airbyte = yaml.safe_load(open('/workspace/config.yaml'))['Airbyte']['config']
subprocess.run(['python', '/workspace/check_job_status.py', '--server', 'airbyte-abctl-control-plane:80',
                '--username', airbyte['username'], '--password', airbyte['password']], check=True)
PY"""

ORACLE_COMMANDS = [
    RENDER,
    "cd /workspace/elt && terraform init -input=false && terraform apply -auto-approve -input=false",
    TRIGGER_SYNCS,
    WAIT_FOR_SYNCS,
    "cd /workspace/elt/trains_dbt && dbt run",
]


def count_airbyte_connections() -> int:
    airbyte = yaml.safe_load(CONFIG.read_text())["Airbyte"]["config"]
    url = f"http://localhost:8001/api/public/v1/connections?workspaceIds={airbyte['workspace_id']}"
    with urllib.request.urlopen(url) as response:
        return len(json.load(response)["data"])


def make_builder(tmp_path, **kwargs) -> EltEnvGroupBuilder:
    [task] = load_tasks(ELT_BENCH_DIR, "snowflake", ["trains"])
    return EltEnvGroupBuilder(
        task=task,
        model_name=MODEL,
        renderer_name=None,
        group_size=1,
        destination="snowflake",
        credential_path=str(CREDENTIAL),
        docker_image=IMAGE,
        docker_network=None,
        workspace_root=str(tmp_path),
        max_turns=10,
        command_timeout=900,
        **kwargs,
    )


def run_oracle(builder: EltEnvGroupBuilder, commands: list[str]):
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer_utils.get_tokenizer(MODEL))

    async def run():
        [env] = await builder.make_envs()
        namespace = builder._rollouts[0].namespace
        try:
            trajectory = await do_single_rollout(ScriptedPolicy(renderer, commands), env)
        finally:
            await builder.cleanup()
        return trajectory, namespace

    return asyncio.run(run())


def database_exists(name: str) -> bool:
    admin = get_destination("snowflake", str(CREDENTIAL)).connector(None)
    try:
        return not admin.query(f"show databases like '{name}'").empty
    finally:
        admin.close()


def test_oracle_pipeline_scores_one(tmp_path):
    connections_before = count_airbyte_connections()
    trajectory, namespace = run_oracle(make_builder(tmp_path, save_snapshot=True), ORACLE_COMMANDS)
    metrics = trajectory.transitions[-1].metrics
    print(namespace, metrics)

    assert sum(t.reward for t in trajectory.transitions) == pytest.approx(1.0)
    assert metrics["srdel"] == 1.0 and metrics["srdt"] == 1.0
    assert "grading_error" not in metrics

    assert not database_exists(namespace)
    assert count_airbyte_connections() == connections_before
    assert database_exists("trains__snapshot")


def test_transform_stage_from_snapshot_scores_one(tmp_path):
    # Runs after the oracle above saved the snapshot: only render the files and run dbt.
    builder = make_builder(tmp_path, stage="transform")
    trajectory, _ = run_oracle(builder, [RENDER, ORACLE_COMMANDS[-1]])
    metrics = trajectory.transitions[-1].metrics
    print(metrics)

    assert sum(t.reward for t in trajectory.transitions) == pytest.approx(1.0)
    assert builder.docker_network is None  # no Airbyte network needed, only the internet
