"""One rollout of a Tinker model on one ELT-Bench task, graded by execution.

Mirrors tinker_cookbook/recipes/harbor_rl/eval.py for a single task:
    python -m eltbench_rl.rollout task=trains credential_path=third_party/ELT-Bench/setup/destination/snowflake_credential.json
    python -m eltbench_rl.rollout bench_dir=tests/fixtures/toy_bench destination=duckdb task=shop
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker_cookbook import cli_utils, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import format_trajectory
from tinker_cookbook.rl.rollouts import do_single_rollout

from eltbench_rl import ELT_BENCH_DIR
from eltbench_rl.env import EltEnvGroupBuilder
from eltbench_rl.tasks import load_tasks


@chz.chz
class RolloutCLIConfig:
    task: str = "trains"
    bench_dir: str = str(ELT_BENCH_DIR)
    destination: str = "snowflake"
    credential_path: str | None = None
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str | None = None
    checkpoint_url: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 1.0
    max_turns: int = 100
    command_timeout: int = 600
    max_trajectory_tokens: int = 64 * 1024
    docker_image: str = "eltbench-rl"
    docker_network: str | None = None
    output_dir: str = cli_utils.runs_path("eltbench_rollouts")


async def run(config: RolloutCLIConfig) -> None:
    [task] = load_tasks(Path(config.bench_dir), config.destination, [config.task])
    out_dir = Path(config.output_dir) / f"{task.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    service_client = tinker.ServiceClient(base_url=config.base_url)
    if config.checkpoint_url:
        sampling_client = service_client.create_sampling_client(
            model_path=config.checkpoint_url, base_model=config.model_name
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client, max_tokens=config.max_tokens, temperature=config.temperature
    )

    builder = EltEnvGroupBuilder(
        task=task,
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        group_size=1,
        destination=config.destination,
        credential_path=config.credential_path,
        docker_image=config.docker_image,
        docker_network=config.docker_network,
        workspace_root=str(out_dir / "workspace"),
        max_turns=config.max_turns,
        command_timeout=config.command_timeout,
        max_trajectory_tokens=config.max_trajectory_tokens,
        max_generation_tokens=config.max_tokens,
    )
    start = time.monotonic()
    try:
        [env] = await builder.make_envs()
        trajectory = await do_single_rollout(policy, env)
    finally:
        await builder.cleanup()

    result = {
        "task": task.name,
        "reward": sum(t.reward for t in trajectory.transitions),
        "turns": len(trajectory.transitions),
        "seconds": round(time.monotonic() - start, 1),
        "metrics": trajectory.transitions[-1].metrics if trajectory.transitions else {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    tokenizer = tokenizer_utils.get_tokenizer(config.model_name)
    (out_dir / "transcript.txt").write_text(format_trajectory(trajectory, tokenizer))
    print(json.dumps(result, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    asyncio.run(run(chz.entrypoint(RolloutCLIConfig)))
