"""ELT-Bench as a tinker-cookbook RL environment.

Follows tinker_cookbook/recipes/harbor_rl/harbor_env.py: one bash tool in a sandbox per rollout and a
reward computed by execution when the episode ends. The warehouse and Airbyte are shared between
rollouts, so each rollout gets its own warehouse namespace (with its own credentials) and its own
Airbyte workspace.

With stage="transform", rollouts skip extract/load: their namespace is cloned from the task's
reference load of its sources (a snapshot saved by a trusted full run with save_snapshot=True, like
the oracle test), and the prompt is stage 2 only.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chz
import yaml
from tinker_cookbook import cli_utils, model_info, tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from eltbench_rl import airbyte
from eltbench_rl.destinations import get_destination
from eltbench_rl.docker_sandbox import DockerSandbox
from eltbench_rl.reward import EltReward
from eltbench_rl.tasks import EltTask, load_tasks, transform_instruction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a data engineer skilled in databases, SQL, and building ELT pipelines. "
    "You have a bash tool that runs commands inside a Docker container; the task files are in /workspace. "
    "Always use absolute paths, and only modify files under /workspace/elt. "
    "When the pipeline is complete, reply without calling a tool: that submits it for grading."
)
# Internal Docker network for Snowflake runs (docker/sandbox-network.sh): its only ways out are the
# Airbyte gate and a proxy that lets HTTPS through to Snowflake and nothing else.
SANDBOX_NETWORK = "eltbench-sandbox"
PROXY = "http://eltbench-proxy:8888"
PROXY_ENV = {
    name: value
    for names, value in [(("HTTPS_PROXY", "HTTP_PROXY"), PROXY), (("NO_PROXY",), "airbyte-gate,localhost")]
    for name in (*names, *(n.lower() for n in names))
}


@dataclass
class _Rollout:
    sandbox: DockerSandbox
    namespace: str
    workspace: Path
    airbyte_workspace: str | None = None


def _update_config(config_path: Path, block: str, values: dict[str, str]) -> None:
    config = yaml.safe_load(config_path.read_text())
    config[block]["config"].update(values)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


class EltEnvGroupBuilder(EnvGroupBuilder):
    """Builds `group_size` isolated rollouts of one ELT-Bench task."""

    def __init__(
        self,
        task: EltTask,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        destination: str,
        credential_path: str | None,
        docker_image: str,
        docker_network: str | None,
        workspace_root: str,
        max_turns: int = 100,
        command_timeout: int = 600,
        max_trajectory_tokens: int = 64 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
        stage: str = "full",
        save_snapshot: bool = False,
    ):
        if stage not in ("full", "transform"):
            raise ValueError(f"stage must be 'full' or 'transform', got {stage!r}")
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.destination = destination
        self.credential_path = credential_path
        self.docker_image = docker_image
        self.stage = stage
        self.snapshot = f"{task.name}__snapshot"
        self.save_snapshot = save_snapshot
        # DuckDB runs need no network at all.
        self.docker_network = docker_network or ("none" if destination == "duckdb" else SANDBOX_NETWORK)
        self.workspace_root = workspace_root
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self._rollouts: list[_Rollout] = []

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        # Grade from the task's reference load when there is one (or this group is making it).
        destination = get_destination(self.destination, self.credential_path)
        has_reference = self.save_snapshot or await asyncio.to_thread(destination.has_snapshot, self.snapshot)
        reference = self.snapshot if has_reference else None
        return await asyncio.gather(*[self._make_env(renderer, reference) for _ in range(self.group_size)])

    async def _make_env(self, renderer: Renderer, reference: str | None) -> Env:
        destination = get_destination(self.destination, self.credential_path)
        namespace = f"{self.task.name}__{uuid.uuid4().hex[:8]}"
        workspace = Path(self.workspace_root) / namespace
        # The agent gets what ELT-Bench gives its agents; admin credentials stay on the host.
        shutil.copytree(self.task.input_dir, workspace, ignore=shutil.ignore_patterns("*_credential.json"))

        env = PROXY_ENV if self.docker_network == SANDBOX_NETWORK else {}
        sandbox = await DockerSandbox.create(self.docker_image, workspace, self.docker_network, env)
        rollout = _Rollout(sandbox, namespace, workspace)
        self._rollouts.append(rollout)
        # The namespace and Airbyte workspace come with their own credentials, which replace the
        # shared ones in config.yaml.
        values = await asyncio.to_thread(destination.create_namespace, namespace, workspace)
        _update_config(workspace / "config.yaml", destination.name, values)
        if self.stage == "transform":
            tables = list(self.task.source_tables)
            await asyncio.to_thread(destination.reset_namespace, namespace, workspace, self.snapshot, tables)
        if self.task.uses_airbyte and self.stage == "full":
            values = await asyncio.to_thread(airbyte.create_workspace, namespace)
            rollout.airbyte_workspace = values["workspace_id"]
            _update_config(workspace / "config.yaml", "Airbyte", values)
        instruction = self.task.instruction if self.stage == "full" else transform_instruction(self.task.instruction)

        bash_tool = HarborBashTool(sandbox, command_timeout=self.command_timeout)
        initial_messages = renderer.create_conversation_prefix_with_tools(
            tools=[bash_tool.bash.to_spec()], system_prompt=SYSTEM_PROMPT
        ) + [{"role": "user", "content": instruction}]
        return build_agent_tool_env(
            renderer=renderer,
            tools=[bash_tool.bash],
            initial_messages=initial_messages,
            reward_fn=EltReward(
                self.task, destination, sandbox, workspace, namespace,
                airbyte_workspace=rollout.airbyte_workspace,
                snapshot=reference,
                save_snapshot=self.save_snapshot,
            ),  # fmt: skip
            # Every command shares one container, so run them in order; retry malformed tool calls.
            rollout_config=RolloutConfig(
                parse_errors=ParseErrorPolicy(max_consecutive=2), tool_execution="sequential"
            ),
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            context_overflow_reward=self.context_overflow_reward,
            # A long planning message shouldn't throw away the pipeline built so far: keep the
            # cut-off turn and continue; the trajectory token budget still bounds the episode.
            terminate_on_length=False,
        )

    async def cleanup(self) -> None:
        destination = get_destination(self.destination, self.credential_path)
        for rollout in self._rollouts:
            try:
                # Remove the Airbyte sources/destinations/connections the agent created.
                await rollout.sandbox.run_command(
                    "if [ -f elt/terraform.tfstate ]; then cd elt && terraform destroy -auto-approve; fi",
                    timeout=600,
                )
                if rollout.airbyte_workspace:
                    await asyncio.to_thread(airbyte.delete_workspace, rollout.airbyte_workspace)
                # Files written in the container can be root-owned on Linux hosts.
                await rollout.sandbox.run_command("rm -rf /workspace/* /workspace/.[!.]*", timeout=120)
                await asyncio.to_thread(destination.drop_namespace, rollout.namespace)
                await rollout.sandbox.cleanup()
            except Exception as e:
                logger.warning("Cleanup of %s failed: %s", rollout.namespace, e)
            shutil.rmtree(rollout.workspace, ignore_errors=True)
        self._rollouts.clear()

    def logging_tags(self) -> list[str]:
        return ["eltbench", self.task.name]


class EltDataset(RLDataset):
    def __init__(self, env_group_builders: list[EltEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.env_group_builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class EltDatasetBuilder(RLDatasetBuilder):
    """One env group per ELT-Bench task (or toy task) under `bench_dir`."""

    bench_dir: str
    destination: str
    batch_size: int
    group_size: int
    model_name: str
    tasks: str = ""  # comma-separated task names; empty means every task in bench_dir
    credential_path: str | None = None
    renderer_name: str | None = None
    docker_image: str = "eltbench-rl"
    docker_network: str | None = None
    workspace_root: str = cli_utils.runs_path("eltbench_workspaces")
    max_turns: int = 100
    command_timeout: int = 600
    max_trajectory_tokens: int = 64 * 1024
    max_generation_tokens: int | None = None
    stage: str = "full"  # "transform": start from a snapshot of loaded sources, train dbt only

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        names = [name.strip() for name in self.tasks.split(",") if name.strip()] or None
        builders = [
            EltEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                destination=self.destination,
                credential_path=self.credential_path,
                docker_image=self.docker_image,
                docker_network=self.docker_network,
                workspace_root=self.workspace_root,
                max_turns=self.max_turns,
                command_timeout=self.command_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                stage=self.stage,
            )
            for task in load_tasks(Path(self.bench_dir), self.destination, names)
        ]
        return EltDataset(builders, self.batch_size), None
