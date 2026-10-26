"""RL training on ELT-Bench tasks with Tinker. Mirrors tinker_cookbook/recipes/harbor_rl/train.py.

Toy task (needs only TINKER_API_KEY):
    python -m eltbench_rl.train bench_dir=tests/fixtures/toy_bench destination=duckdb tasks=shop max_steps=1
Official task (also needs the Snowflake + Airbyte setup from the README):
    python -m eltbench_rl.train tasks=trains credential_path=third_party/ELT-Bench/setup/destination/snowflake_credential.json
"""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, hyperparam_utils
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from eltbench_rl import ELT_BENCH_DIR
from eltbench_rl.env import EltDatasetBuilder


@chz.chz
class CLIConfig:
    """Command-line configuration for ELT-Bench RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0

    # Environment configuration
    bench_dir: str = str(ELT_BENCH_DIR)
    destination: str = "snowflake"
    credential_path: str | None = None
    tasks: str = "trains"
    docker_image: str = "eltbench-rl"
    docker_network: str | None = None
    max_turns: int = 100
    command_timeout: int = 600
    max_trajectory_tokens: int = 64 * 1024

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 1
    learning_rate: float | None = None  # None: hyperparam_utils.get_lr(model_name)
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging / eval / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 0
    save_every: int = 5

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Async rollout configuration: overlap slow ELT rollouts with training
    max_steps_off_policy: int | None = None

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"eltbench-{model_tag}-{cli_config.tasks.replace(',', '+') or 'all'}-"
        f"{cli_config.group_size}group-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or cli_utils.runs_path("eltbench_rl", run_name)

    dataset_builder = EltDatasetBuilder(
        bench_dir=cli_config.bench_dir,
        destination=cli_config.destination,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        tasks=cli_config.tasks,
        credential_path=cli_config.credential_path,
        renderer_name=cli_config.renderer_name,
        docker_image=cli_config.docker_image,
        docker_network=cli_config.docker_network,
        max_turns=cli_config.max_turns,
        command_timeout=cli_config.command_timeout,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_generation_tokens=cli_config.max_tokens,
    )

    config = Config(
        learning_rate=cli_config.learning_rate or hyperparam_utils.get_lr(cli_config.model_name),
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_eltbench_rl",
        renderer_name=cli_config.renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
