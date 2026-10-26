"""Credential-free end-to-end test: real env, Docker sandbox, tokenizer and renderer, DuckDB warehouse.

Scripted policies replay fixed tool calls as tokens through the same rollout loop training uses, so
the rewards below come from actually executing each pipeline and grading the warehouse it leaves.
"""

import asyncio
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import pytest
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer, ToolCall
from tinker_cookbook.rl.rollouts import do_single_rollout

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


def dbt_project(models: dict[str, str], seed: str | None = None) -> list[str]:
    project = "/workspace/elt/shop_dbt"
    commands = [
        f"""{READ_NS} && mkdir -p {project}/models && cd {project} && cat > dbt_project.yml <<'YML'
name: shop_dbt
version: "1.0"
profile: shop
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
    if seed is not None:
        commands.append(f"mkdir -p {project}/seeds && cat > {project}/seeds/answers.csv <<'CSV'\n{seed}CSV")
    commands.append(f"cd {project} && dbt run")
    return commands


SOLUTION_MODELS = {name: (SOLUTION / f"{name}.sql").read_text() for name in ("customer_orders", "daily_revenue")}
WRONG_MODELS = {
    **SOLUTION_MODELS,
    "customer_orders": SOLUTION_MODELS["customer_orders"].replace(" and o.status <> 'cancelled'", ""),
}
LOAD = run_duckdb_sql((SOLUTION / "load.sql").read_text())
CTAS = run_duckdb_sql(
    "\n".join(f"create table {{{{ target.schema }}}}.{name} as {sql};" for name, sql in SOLUTION_MODELS.items())
)
GT_CSV = (TASK.gt_dir / "customer_orders.csv").read_text()

POLICIES = {
    "oracle": [LOAD, *dbt_project(SOLUTION_MODELS)],
    "ctas_hacker": [LOAD, CTAS],
    "seed_hacker": [LOAD, *dbt_project(SOLUTION_MODELS, seed=GT_CSV)],
    "wrong_column": [LOAD, *dbt_project(WRONG_MODELS)],
    "noop": [],
}


def make_builder(workspace_root: Path) -> EltEnvGroupBuilder:
    return EltEnvGroupBuilder(
        task=TASK,
        model_name=MODEL,
        renderer_name=None,
        group_size=len(POLICIES),
        destination="duckdb",
        credential_path=None,
        docker_image=IMAGE,
        docker_network=None,
        workspace_root=str(workspace_root),
        max_turns=10,
        command_timeout=300,
    )


def test_scripted_tool_calls_parse_back():
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer_utils.get_tokenizer(MODEL))
    policy = ScriptedPolicy(renderer, ["ls /workspace"])
    action = asyncio.run(policy(None, []))
    message, _ = renderer.parse_response(action.tokens)

    assert json.loads(message["tool_calls"][0].function.arguments) == {"command": "ls /workspace"}


def test_rewards_come_from_executing_the_pipeline(tmp_path):
    builder = make_builder(tmp_path)
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer_utils.get_tokenizer(MODEL))

    async def run():
        envs = await builder.make_envs()
        try:
            trajectories = await asyncio.gather(
                *[do_single_rollout(ScriptedPolicy(renderer, cmds), env) for cmds, env in zip(POLICIES.values(), envs)]
            )
        finally:
            await builder.cleanup()
        return dict(zip(POLICIES, trajectories))

    trajectories = asyncio.run(run())
    rewards = {name: sum(t.reward for t in traj.transitions) for name, traj in trajectories.items()}
    print(rewards)

    assert rewards["oracle"] == pytest.approx(1.0)
    assert trajectories["oracle"].transitions[-1].metrics["srdel"] == 1.0
    assert rewards["ctas_hacker"] == pytest.approx(0.2)
    assert rewards["seed_hacker"] == pytest.approx(0.2)
    assert 0.2 < rewards["wrong_column"] < 1.0
    assert rewards["noop"] == pytest.approx(0.0)

    leftover = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=eltbench-", "-q"], capture_output=True, text=True
    )
    assert leftover.stdout.strip() == ""
    assert list(tmp_path.iterdir()) == []


def test_builder_pickles(tmp_path):
    builder = make_builder(tmp_path)
    restored = pickle.loads(pickle.dumps(builder))

    assert restored.task == builder.task
    assert restored.docker_network == "none"
    assert restored.group_size == builder.group_size
