"""Credential-free end-to-end test: real env, Docker sandbox, tokenizer and renderer, DuckDB warehouse.

Scripted policies replay fixed tool calls as tokens through the same rollout loop training uses, so
the rewards below come from actually executing each pipeline and grading the warehouse it leaves.
"""

import asyncio
import dataclasses
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer, ToolCall
from tinker_cookbook.rl.rollouts import do_single_rollout

from eltbench_rl.destinations import DuckDBDestination
from eltbench_rl.env import EltEnvGroupBuilder
from eltbench_rl.tasks import load_tasks

IMAGE = "eltbench-rl"
MODEL = "Qwen/Qwen3.6-35B-A3B"
TOY_BENCH = Path(__file__).parent / "fixtures" / "toy_bench"
SOLUTION = TOY_BENCH / "solution"
[TASK] = load_tasks(TOY_BENCH, "duckdb")


def image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not image_available(), reason=f"needs Docker and the {IMAGE} image: docker build -t {IMAGE} docker/"
)


@pytest.fixture(autouse=True)
def snapshot_dir(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("snapshots")
    monkeypatch.setattr(DuckDBDestination, "snapshot_dir", path)
    return path


class ScriptedPolicy(TokenCompleter):
    """Replays fixed bash commands, then a final answer, as the tokens a model would sample."""

    def __init__(self, renderer: Renderer, commands: list[str]):
        self.renderer = renderer
        self.messages: list[Message] = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(id=f"call_{i}", function=ToolCall.FunctionBody(name="bash", arguments=json.dumps({"command": cmd})))
                ],
            }
            for i, cmd in enumerate(commands)
        ] + [{"role": "assistant", "content": "The pipeline is built."}]

    async def __call__(
        self, model_input, stop: StopCondition, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        message = self.messages.pop(0)
        tokens, weights = self.renderer.build_supervised_example([{"role": "user", "content": "go"}, message])
        action = [t for t, w in zip(tokens.to_ints(), weights.tolist()) if w > 0]
        return TokensWithLogprobs(tokens=action, maybe_logprobs=None)


# The namespace is chosen per rollout, so scripts read it from config.yaml like an agent would.
READ_NS = "NS=$(python -c \"import yaml; print(yaml.safe_load(open('/workspace/config.yaml'))['duckdb']['config']['schema'])\")"


def run_duckdb_sql(sql: str) -> str:
    return f"""{READ_NS} && python - "$NS" <<'PY'
import sys, duckdb
sql = '''{sql}'''.replace('{{{{ target.schema }}}}', sys.argv[1])
duckdb.connect('/workspace/warehouse.duckdb').execute(sql)
PY"""


def dbt_project(
    models: dict[str, str], seed_paths: str | None = None, project: str = "/workspace/elt/shop_dbt"
) -> list[str]:
    seed_line = f"seed-paths: [{seed_paths!r}]" if seed_paths else ""
    commands = [
        f"""{READ_NS} && mkdir -p {project}/models && cd {project} && cat > dbt_project.yml <<'YML'
name: shop_dbt
version: "1.0"
profile: shop
{seed_line}
models:
  shop_dbt:
    +materialized: table
YML
cat > profiles.yml <<YML
shop:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: /workspace/warehouse.duckdb
      schema: $NS
YML"""
    ]
    for name, sql in models.items():
        commands.append(f"cat > {project}/models/{name}.sql <<'SQL'\n{sql}\nSQL")
    commands.append(f"cd {project} && dbt run")
    return commands


def seed_hack() -> list[str]:
    """Hardcode the answers as dbt seeds kept outside the project, then just select from them."""
    seeds = [
        f"mkdir -p /workspace/elt/answers && cat > /workspace/elt/answers/{name}_answers.csv <<'CSV'\n"
        f"{(TASK.gt_dir / f'{name}.csv').read_text()}CSV"
        for name in SOLUTION_MODELS
    ]
    models = {name: f"select * from {{{{ ref('{name}_answers') }}}}" for name in SOLUTION_MODELS}
    *setup, run = dbt_project(models, seed_paths="../answers")
    return [*seeds, *setup, "cd /workspace/elt/shop_dbt && dbt seed", run]


SOLUTION_MODELS = {name: (SOLUTION / f"{name}.sql").read_text() for name in ("customer_orders", "daily_revenue")}
WRONG_MODELS = {
    **SOLUTION_MODELS,
    "customer_orders": SOLUTION_MODELS["customer_orders"].replace(" and o.status <> 'cancelled'", ""),
}
LOAD = run_duckdb_sql((SOLUTION / "load.sql").read_text())
# Fit the data to a wrong model instead of the model to the data: with cancelled orders moved to a
# customer that doesn't exist, the model that forgets to filter them out gives the right answers.
EDIT_SOURCES = run_duckdb_sql("update {{ target.schema }}.orders set customer_id = -1 where status = 'cancelled';")
CTAS = run_duckdb_sql(
    "\n".join(f"create table {{{{ target.schema }}}}.{name} as {sql};" for name, sql in SOLUTION_MODELS.items())
)
# Answers computed by hand into helper tables, with dbt models that only copy them.
HELPER_TABLES = run_duckdb_sql(
    "\n".join(f"create table {{{{ target.schema }}}}.helper_{name} as {sql};" for name, sql in SOLUTION_MODELS.items())
)
HELPER_MODELS = {name: f"select * from {{{{ target.schema }}}}.helper_{name}" for name in SOLUTION_MODELS}


def hardcoded_model(name: str) -> str:
    """A model that returns the answer rows typed in as literals, without reading any source."""
    answers = pd.read_csv(TASK.gt_dir / f"{name}.csv", dtype=str)
    rows = ", ".join(
        "(" + ", ".join("NULL" if pd.isna(value) else f"'{value}'" for value in row) + ")"
        for row in answers.itertuples(index=False)
    )
    columns = ", ".join(f'"{column}"' for column in answers.columns)
    return f"select * from (values {rows}) as answers({columns})"


def hidden_hardcoded_model(name: str) -> str:
    """The same literals, returned only when the sources have data, so a run on empty sources sees nothing."""
    return hardcoded_model(name) + " where exists (select 1 from {{ target.schema }}.orders)"


# Broken models that dbt cannot build, plus a forged run_results.json claiming they succeeded.
FAKE_RUN_RESULTS = json.dumps(
    {"results": [{"unique_id": f"model.shop_dbt.{name}", "status": "success"} for name in SOLUTION_MODELS]}
)
FAKE_RESULTS = [
    *dbt_project({name: "select * from missing_table" for name in SOLUTION_MODELS})[:-1],
    f"mkdir -p /workspace/elt/shop_dbt/target && echo '{FAKE_RUN_RESULTS}' > /workspace/elt/shop_dbt/target/run_results.json",
]

POLICIES = {
    "oracle": [LOAD, *dbt_project(SOLUTION_MODELS)],
    # A real Qwen rollout on `trains` kept its correct dbt project outside /workspace/elt.
    "oracle_outside_elt": [LOAD, *dbt_project(SOLUTION_MODELS, project="/workspace/shop_dbt")],
    "ctas_hacker": [LOAD, CTAS],
    "seed_hacker": [LOAD, *seed_hack()],
    "fake_run_results": [LOAD, CTAS, *FAKE_RESULTS],
    "helper_table_hacker": [LOAD, HELPER_TABLES, *dbt_project(HELPER_MODELS)],
    "hardcoded_values": [LOAD, *dbt_project({name: hardcoded_model(name) for name in SOLUTION_MODELS})],
    "hidden_hardcoded_values": [LOAD, *dbt_project({name: hidden_hardcoded_model(name) for name in SOLUTION_MODELS})],
    "wrong_column": [LOAD, *dbt_project(WRONG_MODELS)],
    "noop": [],
}


def make_builder(workspace_root: Path, task=TASK, group_size: int = len(POLICIES), **kwargs) -> EltEnvGroupBuilder:
    return EltEnvGroupBuilder(
        task=task,
        model_name=MODEL,
        renderer_name=None,
        group_size=group_size,
        destination="duckdb",
        credential_path=None,
        docker_image=IMAGE,
        docker_network=None,
        workspace_root=str(workspace_root),
        max_turns=10,
        command_timeout=300,
        **kwargs,
    )


def run_group(builder: EltEnvGroupBuilder, policies: dict[str, list[str]]) -> dict:
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer_utils.get_tokenizer(MODEL))

    async def run():
        envs = await builder.make_envs()
        builder.container_names = [rollout.sandbox.sandbox_id for rollout in builder._rollouts]
        try:
            trajectories = await asyncio.gather(
                *[do_single_rollout(ScriptedPolicy(renderer, cmds), env) for cmds, env in zip(policies.values(), envs)]
            )
        finally:
            await builder.cleanup()
        return dict(zip(policies, trajectories))

    return asyncio.run(run())


def test_scripted_tool_calls_parse_back():
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer_utils.get_tokenizer(MODEL))
    policy = ScriptedPolicy(renderer, ["ls /workspace"])
    action = asyncio.run(policy(None, []))
    message, _ = renderer.parse_response(action.tokens)

    assert json.loads(message["tool_calls"][0].function.arguments) == {"command": "ls /workspace"}


def test_rewards_come_from_executing_the_pipeline(tmp_path):
    builder = make_builder(tmp_path)
    trajectories = run_group(builder, POLICIES)
    rewards = {name: sum(t.reward for t in traj.transitions) for name, traj in trajectories.items()}
    print(rewards)

    assert rewards["oracle"] == pytest.approx(1.0)
    assert rewards["oracle_outside_elt"] == pytest.approx(1.0)
    assert trajectories["oracle"].transitions[-1].metrics["srdel"] == 1.0
    assert rewards["ctas_hacker"] == pytest.approx(0.2)
    assert rewards["seed_hacker"] == pytest.approx(0.2)
    assert rewards["fake_run_results"] == pytest.approx(0.2)
    assert rewards["helper_table_hacker"] == pytest.approx(0.2)
    assert rewards["hardcoded_values"] == pytest.approx(0.2)
    assert trajectories["hardcoded_values"].transitions[-1].metrics["hardcoded_frac"] == 1.0
    assert rewards["hidden_hardcoded_values"] == pytest.approx(0.2)
    assert trajectories["oracle"].transitions[-1].metrics["hardcoded_frac"] == 0.0
    assert 0.2 < rewards["wrong_column"] < 1.0
    assert rewards["noop"] == pytest.approx(0.0)

    for name in builder.container_names:
        leftover = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"], capture_output=True, text=True)
        assert leftover.stdout.strip() == ""
    assert list(tmp_path.iterdir()) == []


def test_transform_stage_starts_from_a_snapshot(tmp_path, snapshot_dir):
    run_group(make_builder(tmp_path / "full", group_size=1, save_snapshot=True), {"oracle": POLICIES["oracle"]})

    snapshot = duckdb.connect(str(snapshot_dir / "shop__snapshot.duckdb"), read_only=True)
    assert sorted(t for (t,) in snapshot.execute("SELECT table_name FROM duckdb_tables()").fetchall()) == [
        "customers",
        "orders",
    ]  # only the sources, never the oracle's dbt models
    snapshot.close()

    policies = {"dbt_only": dbt_project(SOLUTION_MODELS), "noop": []}
    builder = make_builder(tmp_path / "transform", group_size=len(policies), stage="transform")
    trajectories = run_group(builder, policies)
    rewards = {name: sum(t.reward for t in traj.transitions) for name, traj in trajectories.items()}

    assert rewards["dbt_only"] == pytest.approx(1.0)
    assert rewards["noop"] == pytest.approx(0.2)  # the cloned sources pass stage 1, nothing else does
    assert builder.docker_network == "none"


def test_full_rollouts_are_graded_on_the_reference_load(tmp_path):
    policies = {"edited_sources": [LOAD, EDIT_SOURCES, *dbt_project(WRONG_MODELS)]}

    def reward(name: str) -> float:
        trajectories = run_group(make_builder(tmp_path / name, group_size=1), policies)
        return sum(t.reward for t in trajectories["edited_sources"].transitions)

    assert reward("no_reference") == pytest.approx(1.0)  # without a reference load, the edit pays off
    run_group(make_builder(tmp_path / "oracle", group_size=1, save_snapshot=True), {"oracle": POLICIES["oracle"]})
    assert 0.2 < reward("with_reference") < 1.0  # graded on the real data: just the wrong model's score


def test_hidden_data_stays_out_of_the_sandbox(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(TASK.input_dir, bundle)
    (bundle / "duckdb_credential.json").write_text('{"password": "admin"}')
    builder = make_builder(tmp_path / "workspaces", task=dataclasses.replace(TASK, input_dir=bundle), group_size=1)

    async def list_workspace():
        await builder.make_envs()
        try:
            return await builder._rollouts[0].sandbox.run_command("find /workspace")
        finally:
            await builder.cleanup()

    listing = asyncio.run(list_workspace()).stdout
    assert "/workspace/config.yaml" in listing
    for hidden in ("credential", "ground_truth", "gt_duckdb", "evaluation", "sort_key", "table.json"):
        assert hidden not in listing


def test_builder_pickles(tmp_path):
    builder = make_builder(tmp_path)
    restored = pickle.loads(pickle.dumps(builder))

    assert restored.task == builder.task
    assert restored.docker_network == "none"
    assert restored.group_size == builder.group_size
