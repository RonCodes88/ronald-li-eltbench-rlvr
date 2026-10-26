"""ELT-Bench as a tinker-cookbook RL environment.

Follows tinker_cookbook/recipes/harbor_rl/harbor_env.py: one bash tool in a sandbox per rollout and a
reward computed by execution when the episode ends. Each rollout also gets its own warehouse
namespace, because the warehouse and Airbyte are shared between rollouts.
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

from eltbench_rl.destinations import Destination, get_destination
from eltbench_rl.docker_sandbox import DockerSandbox
from eltbench_rl.reward import EltReward
from eltbench_rl.tasks import EltTask, load_tasks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a data engineer skilled in databases, SQL, and building ELT pipelines. "
    "You have a bash tool that runs commands inside a Docker container; the task files are in /workspace. "
    "Always use absolute paths, and only modify files under /workspace/elt. "
    "When the pipeline is complete, reply without calling a tool: that submits it for grading."
)
# ELT-Bench's docker compose network, where its sources and the Airbyte control plane live.
ELT_BENCH_NETWORK = "elt-docker_elt_network"


@dataclass
class _Rollout:
    sandbox: DockerSandbox
    namespace: str
    workspace: Path


def _set_namespace(config_path: Path, destination: Destination, namespace: str) -> None:
    config = yaml.safe_load(config_path.read_text())
    config[destination.name]["config"][destination.namespace_field] = namespace
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
    ):
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.destination = destination
        self.credential_path = credential_path
        self.docker_image = docker_image
        # Airbyte tasks need ELT-Bench's network; local tasks get no network at all.
        self.docker_network = docker_network or (ELT_BENCH_NETWORK if task.uses_airbyte else "none")
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
        return await asyncio.gather(*[self._make_env(renderer) for _ in range(self.group_size)])

    async def _make_env(self, renderer: Renderer) -> Env:
        destination = get_destination(self.destination, self.credential_path)
        namespace = f"{self.task.name}__{uuid.uuid4().hex[:8]}"
        workspace = Path(self.workspace_root) / namespace
        # The agent gets what ELT-Bench gives its agents; admin credentials stay on the host.
        shutil.copytree(self.task.input_dir, workspace, ignore=shutil.ignore_patterns("*_credential.json"))
        _set_namespace(workspace / "config.yaml", destination, namespace)

        sandbox = await DockerSandbox.create(self.docker_image, workspace, self.docker_network)
        self._rollouts.append(_Rollout(sandbox, namespace, workspace))
        await asyncio.to_thread(destination.create_namespace, namespace)

        bash_tool = HarborBashTool(sandbox, command_timeout=self.command_timeout)
        initial_messages = renderer.create_conversation_prefix_with_tools(
            tools=[bash_tool.bash.to_spec()], system_prompt=SYSTEM_PROMPT
        ) + [{"role": "user", "content": self.task.instruction}]
        return build_agent_tool_env(
            renderer=renderer,
            tools=[bash_tool.bash],
            initial_messages=initial_messages,
            reward_fn=EltReward(self.task, destination, sandbox, workspace, namespace),
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
            )
            for task in load_tasks(Path(self.bench_dir), self.destination, names)
        ]
        return EltDataset(builders, self.batch_size), None
