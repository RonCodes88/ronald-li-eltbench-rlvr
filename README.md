# ELT-Bench RLVR environment for Tinker

An RL environment for [ELT-Bench](https://arxiv.org/abs/2504.04808) built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) and following its `harbor_rl` recipe, where the model gets a bash tool in a Docker sandbox and has to build the whole pipeline itself, Airbyte through Terraform for extract/load and then dbt for the transforms, and the reward comes from re-running what it built and grading the warehouse against ELT-Bench's hidden ground truth with ELT-Bench's own comparator. It runs on Snowflake for the official tasks and on DuckDB for a local test without credentials, and the design, reward hacking and efficiency notes are in the design doc.

## Install

```bash
git clone --recurse-submodules https://github.com/RonCodes88/ronald-li-eltbench-rlvr.git
cd ronald-li-eltbench-rlvr
uv sync
docker build -t eltbench-rl docker/
```

## 1. Credential-free local integration test

This only needs Docker:

```bash
uv run pytest
```

`tests/test_local_integration.py` runs the real environment on a toy DuckDB task, with scripted policies that replay tool calls through the training rollout loop, so a correct pipeline scores 1.0, each reward-hacking attempt (hand-written tables, seeds, forged dbt results, helper tables, typed-in answers) scores 0.2, a partly wrong model 0.79 and doing nothing 0.0. The official tests skip themselves until the official setup below exists.

## 2. Credentialed rollout on an official task

This needs the official setup below and Tinker credentials (`TINKER_API_KEY` or `tinker auth login`):

```bash
uv run pytest tests/test_official_trains.py -s   # scripted oracle on trains: reward 1.0
uv run python -m eltbench_rl.rollout task=trains destination=snowflake \
    credential_path=third_party/ELT-Bench/setup/destination/snowflake_credential.json \
    model_name=Qwen/Qwen3.6-35B-A3B:peft:262144 max_trajectory_tokens=200000 max_turns=60
```

The oracle test also checks the sandbox network and the Airbyte gate, and saves `TRAINS__SNAPSHOT`, the reference load of the task's sources that check 3 starts from. The rollout prints its reward breakdown and saves a transcript under `~/tinker-runs/eltbench_rollouts/`, and the base model's reward varies a lot between rollouts (we've seen 0.0, 0.2 and 1.0 on `trains`).

## 3. Tinker RL step with execution-derived rewards

This needs Tinker credentials, and the official command needs the oracle test above to have run first:

```bash
# toy task, no other setup
uv run python -m eltbench_rl.train bench_dir=tests/fixtures/toy_bench destination=duckdb tasks=shop \
    group_size=4 max_turns=20 max_steps=1
# official task, dbt-only stage
uv run python -m eltbench_rl.train tasks=trains destination=snowflake stage=transform \
    credential_path=third_party/ELT-Bench/setup/destination/snowflake_credential.json \
    group_size=4 max_turns=30
```

Metrics go to `metrics.jsonl` under `~/tinker-runs/eltbench_rl/<run>/`, and `num_epochs=N` runs more steps on the same task.

## Official setup (Snowflake + local Airbyte)

This follows [ELT-Bench's README](third_party/ELT-Bench/README.md) with fixes for current Airbyte, and the credential files stay in the submodule so they're never committed.

1. **Snowflake.** Use a [trial account](https://signup.snowflake.com/), since ELT-Bench and this repo sign in with passwords, which Snowflake [no longer allows](https://docs.snowflake.com/en/user-guide/security-mfa-rollout) for service users on paid accounts. Create an admin user with `SYSADMIN` and `SECURITYADMIN`, put it in `third_party/ELT-Bench/setup/destination/snowflake_credential.json` as `{"account", "user", "password", "role": "sysadmin", "warehouse": "AIRBYTE_WAREHOUSE"}`, and run ELT-Bench's `setup/destination/setup.sql` as that user.
2. **Airbyte.** Give Docker about 12 GB of memory and run `abctl local install --port 8001 --values docker/airbyte-values.yaml`, where the values file turns off Airbyte's own auth so that ELT-Bench's Terraform provider can sign in to Airbyte 2.x. If you apply it to an existing install, restart Airbyte afterwards or job launches fail with `Unauthorized`:
   ```bash
   docker exec airbyte-abctl-control-plane kubectl --kubeconfig /etc/kubernetes/admin.conf -n airbyte-abctl rollout restart deployment
   ```
3. **ELT-Bench connector.** Import `third_party/ELT-Bench/setup/elt_snowflake.yaml` in the Airbyte UI (Builder → Import YAML → Publish), then fill `third_party/ELT-Bench/setup/airbyte/airbyte_credential.json` with your email, any password, the workspace ID and the connector's definition ID. If publishing fails with `No declarative manifest image version found`, Docker Hub rate-limited Airbyte, and you can register the image by hand:
   ```bash
   docker exec airbyte-abctl-control-plane kubectl --kubeconfig /etc/kubernetes/admin.conf -n airbyte-abctl exec airbyte-db-0 -- \
     psql -U airbyte -d db-airbyte -c "insert into declarative_manifest_image_version (major_version, image_version, image_sha) values (6, '6.33.4', 'sha256:8fd974de2be41d6acad43a3d43576f9940a04c4f289164a2b1a5846d6f676b65')"
   ```
4. **Sources.** `trains` only needs the REST API:
   ```bash
   cd third_party/ELT-Bench/setup && uvx gdown 1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0 && unzip -o data_api.zip -d ../elt-docker/rest_api
   cd ../elt-docker && docker compose up -d python-rest-api
   docker network connect elt-docker_elt_network airbyte-abctl-control-plane
   ```
5. **Sandbox network.** Run `docker/sandbox-network.sh` (safe to re-run), which leaves sandboxes able to reach only Snowflake and a gate that keeps each rollout in its own Airbyte workspace.
6. **Task inputs.** Run `cd third_party/ELT-Bench/setup && uv run --project ../../.. python write_config.py --destination snowflake`, and the ground truth downloads from HuggingFace on first use.

## Adding a warehouse

Implement the `Destination` protocol in `eltbench_rl/destinations.py`, register it in `get_destination` and generate inputs with ELT-Bench's `write_config.py --destination <name>`, and the environment code stays the same (ELT-Bench already ships connectors for Databricks and Redshift).

## Layout

```
eltbench_rl/
  tasks.py           ELT-Bench tasks and prompts
  destinations.py    Destination protocol; Snowflake and DuckDB
  reward.py          execution-based reward
  airbyte.py         per-rollout Airbyte workspaces and sync check
  docker_sandbox.py  cookbook SandboxInterface over docker
  env.py             env group builder and dataset (mirrors harbor_rl)
  train.py           RL training CLI
  rollout.py         single-rollout CLI
docker/              sandbox image, egress proxy, Airbyte gate, network setup
tests/               unit tests, local integration test, official tests
third_party/ELT-Bench  submodule
```

## Credits

- [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) (CC BY-SA 4.0): tasks, setup, evaluation comparator, connectors and task prompt.
- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (Apache-2.0): RL loop, tool-use environment, bash tool and sandbox interface.
