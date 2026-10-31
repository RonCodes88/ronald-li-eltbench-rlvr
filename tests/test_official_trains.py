"""Official ELT-Bench task `trains` on real Airbyte + Snowflake, driven by a hand-written oracle.

Needs the official setup from the README (Snowflake credentials, local Airbyte, ELT-Bench sources and
generated inputs); skipped otherwise. The oracle does what an agent should: read /workspace/config.yaml,
configure Airbyte with Terraform, trigger and wait for the syncs, then build the dbt model. It also
saves the snapshot of loaded sources that transform-stage training starts from.
"""

import asyncio
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from test_local_integration import IMAGE, MODEL, ScriptedPolicy, image_available
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout

from eltbench_rl import ELT_BENCH_DIR, airbyte
from eltbench_rl.destinations import get_destination
from eltbench_rl.docker_sandbox import DockerSandbox
from eltbench_rl.env import PROXY_ENV, SANDBOX_NETWORK, EltEnvGroupBuilder
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
subprocess.run(['python', '/workspace/check_job_status.py', '--server', airbyte['server_url'].split('/')[2],
                '--username', airbyte['username'], '--password', airbyte['password']], check=True)
PY"""

ORACLE_COMMANDS = [
    RENDER,
    "cd /workspace/elt && terraform init -input=false && terraform apply -auto-approve -input=false",
    TRIGGER_SYNCS,
    WAIT_FOR_SYNCS,
    "cd /workspace/elt/trains_dbt && dbt run",
]


def airbyte_workspace_exists(workspace_id: str) -> bool:
    try:
        urllib.request.urlopen(f"{airbyte.API_URL}/workspaces/{workspace_id}")
        return True
    except urllib.error.HTTPError as e:
        assert e.code == 404
        return False


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
        try:
            [env] = await builder.make_envs()
            rollout = builder._rollouts[0]
            trajectory = await do_single_rollout(ScriptedPolicy(renderer, commands), env)
        finally:
            await builder.cleanup()
        return trajectory, rollout

    return asyncio.run(run())


def snowflake_query(sql: str):
    admin = get_destination("snowflake", str(CREDENTIAL)).connector(None)
    try:
        return admin.query(sql)
    finally:
        admin.close()


def database_exists(name: str) -> bool:
    return not snowflake_query(f"show databases like '{name}'").empty


def test_sandbox_reaches_only_snowflake_and_the_airbyte_gate(tmp_path):
    account = json.loads(CREDENTIAL.read_text())["account"]
    urls = {
        f"https://{account}.snowflakecomputing.com": True,
        "http://airbyte-gate:80/api/public/v1/health": True,
        # Airbyte itself, which has no auth; the proxy refuses it too.
        "--noproxy '*' http://airbyte-abctl-control-plane:80/api/public/v1/health": False,
        "https://huggingface.co/datasets/tttjjj/elt_bench": False,  # the public ground truth
        "https://github.com": False,
        "https://pypi.org": False,
    }

    async def check():
        sandbox = await DockerSandbox.create(IMAGE, tmp_path, SANDBOX_NETWORK, PROXY_ENV)
        try:
            codes = {}
            for url in [*urls, "--noproxy '*' https://huggingface.co"]:
                result = await sandbox.run_command(f"curl -s -o /dev/null -m 15 -w '%{{http_code}}' {url}", timeout=30)
                codes[url] = result.stdout
            return codes
        finally:
            await sandbox.cleanup()

    codes = asyncio.run(check())
    print(codes)
    for url, reachable in urls.items():
        assert (codes[url] != "000") == reachable, url
    assert codes["--noproxy '*' https://huggingface.co"] == "000"


GATE_REQUESTS = """python - <<'PY'
import json, requests
for name, (login, method, path, body) in json.load(open("/workspace/requests.json")).items():
    response = requests.request(method, login["server_url"] + path, json=body, auth=(login["username"], login["password"]))
    print(json.dumps([name, response.status_code]))
PY"""


GROUND_TRUTH_URL = "https://huggingface.co/datasets/tttjjj/elt_bench/resolve/main/gt_snowflake/trains/trains.csv"


def test_airbyte_gate_keeps_rollouts_in_their_own_workspace(tmp_path):
    mine, other = airbyte.create_workspace("gate_test_mine"), airbyte.create_workspace("gate_test_other")
    file_source = {"sourceType": "file", "format": "csv", "provider": {"storage": "HTTPS"}, "dataset_name": "train"}
    requests = {
        "own workspace": (mine, "GET", f"workspaces/{mine['workspace_id']}", None),
        "other workspace": (mine, "GET", f"workspaces/{other['workspace_id']}", None),
        "list all workspaces": (mine, "GET", "workspaces", None),
        "source in other workspace": (mine, "POST", "sources", {"workspaceId": other["workspace_id"], "name": "x"}),
        "wrong password": ({**mine, "password": "guess"}, "GET", f"workspaces/{mine['workspace_id']}", None),
        "Airbyte's internal API": (mine, "POST", "../../v1/workspaces/list", {}),
        # Airbyte can reach the internet, so a source could load the public ground truth.
        "source loading the ground truth": (
            mine, "POST", "sources",
            {"workspaceId": mine["workspace_id"], "name": "x", "configuration": {**file_source, "url": GROUND_TRUTH_URL}},
        ),  # fmt: skip
    }
    (tmp_path / "requests.json").write_text(json.dumps(requests))

    async def check():
        sandbox = await DockerSandbox.create(IMAGE, tmp_path, SANDBOX_NETWORK, PROXY_ENV)
        try:
            return await sandbox.run_command(GATE_REQUESTS, timeout=60)
        finally:
            await sandbox.cleanup()

    try:
        result = asyncio.run(check())
    finally:
        airbyte.delete_workspace(mine["workspace_id"])
        airbyte.delete_workspace(other["workspace_id"])
    print(result.stdout, result.stderr)
    codes = dict(json.loads(line) for line in result.stdout.splitlines())
    assert codes == {
        "own workspace": 200,
        "other workspace": 403,
        "list all workspaces": 403,
        "source in other workspace": 403,
        "wrong password": 401,
        "Airbyte's internal API": 403,
        "source loading the ground truth": 403,
    }


def test_oracle_pipeline_scores_one(tmp_path):
    trajectory, rollout = run_oracle(make_builder(tmp_path, save_snapshot=True), ORACLE_COMMANDS)
    namespace = rollout.namespace
    metrics = trajectory.transitions[-1].metrics
    print(namespace, metrics)

    assert sum(t.reward for t in trajectory.transitions) == pytest.approx(1.0)
    assert metrics["srdel"] == 1.0 and metrics["srdt"] == 1.0
    assert "grading_error" not in metrics

    assert not database_exists(namespace)
    assert snowflake_query(f"show users like '{namespace}_USER'").empty  # per-rollout user and role are gone
    assert snowflake_query(f"show roles like '{namespace}_ROLE'").empty
    assert not database_exists(f"{namespace}__sources")
    assert not airbyte_workspace_exists(rollout.airbyte_workspace)
    assert metrics["hardcoded_frac"] == 0.0
    assert database_exists("trains__snapshot")


def test_transform_stage_from_snapshot_scores_one(tmp_path):
    # Runs after the oracle above saved the snapshot: only render the files and run dbt.
    builder = make_builder(tmp_path, stage="transform")
    trajectory, _ = run_oracle(builder, [RENDER, ORACLE_COMMANDS[-1]])
    metrics = trajectory.transitions[-1].metrics
    print(metrics)

    assert sum(t.reward for t in trajectory.transitions) == pytest.approx(1.0)
    assert builder.docker_network == SANDBOX_NETWORK
