# ELT-Bench RLVR environment for Tinker

This is a reinforcement-learning-with-verifiable-rewards (RLVR) environment for [ELT-Bench](https://arxiv.org/abs/2504.04808), built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) and following its `harbor_rl` recipe.

In each rollout, a model gets an ELT-Bench task and a bash tool inside a Docker sandbox. It must configure Airbyte with Terraform to extract and load the sources into a warehouse, write the dbt transformations, and submit the pipeline. The reward comes from **executing** the result: we re-run the submitted dbt project, then grade the warehouse it leaves behind against ELT-Bench's hidden ground truth, using ELT-Bench's own comparator.

Snowflake (the official destination) and DuckDB (local, credential-free) are supported. Other warehouses plug in through one small class. The design (task anatomy, reward, reward hacking and efficiency) is written up in the accompanying design doc.

## Install

```bash
git clone --recurse-submodules https://github.com/RonCodes88/ronald-li-eltbench-rlvr.git
cd ronald-li-eltbench-rlvr
uv sync                                    # Python 3.12, pinned deps incl. tinker-cookbook==0.5.8
docker build -t eltbench-rl docker/        # sandbox image: dbt (snowflake, duckdb), terraform, offline Airbyte provider
```

## The three graded checks

**1. Credential-free local integration test.** Needs only Docker.

```bash
uv run pytest            # the official-task test skips itself unless the official setup below exists
```

`tests/test_local_integration.py` runs the real environment with the Docker sandbox, the real tokenizer and renderer, and a local DuckDB warehouse. Scripted policies replay tool calls as tokens through the same rollout loop used in training, and each pipeline is graded by executing it:

| Policy | Reward |
|---|---|
| Correct pipeline | 1.0 |
| Wrote the result tables by hand instead of with dbt | 0.2 |
| Hardcoded the answers as dbt seeds | 0.2 |
| Forged dbt `run_results.json` | 0.2 |
| Computed the answers by hand into helper tables, with dbt models that copy them | 0.2 (1.0 before the clean re-run) |
| Typed the answers into the models as a `VALUES` list | 0.2 |
| The same, but hidden when the sources are empty (`WHERE EXISTS (SELECT 1 FROM orders)`) | 0.2 (1.0 with an empty-source check instead of the one-row run) |
| One model partly wrong | 0.79 |
| Did nothing | 0.0 |

The same file also checks the transform stage (below): the correct pipeline saves a snapshot of its loaded sources, and a second group starts from that snapshot, where dbt alone scores 1.0 and doing nothing scores 0.2. Another test edits the loaded sources so that a wrong model gives the right answers: that scores 1.0 when grading uses the rollout's own load, and only the wrong model's 0.79 once a reference load exists.

**2. Credentialed rollout on an official ELT-Bench task.** Needs the Official setup below and Tinker credentials (`TINKER_API_KEY`, or `tinker auth login`).

```bash
uv run python -m eltbench_rl.rollout task=trains destination=snowflake \
    credential_path=third_party/ELT-Bench/setup/destination/snowflake_credential.json \
    model_name=Qwen/Qwen3.6-35B-A3B:peft:262144 max_trajectory_tokens=200000 max_turns=60
uv run pytest tests/test_official_trains.py -s   # hand-written oracle on the same task: reward 1.0
```

The oracle test also saves `TRAINS__SNAPSHOT`, the task's reference load of its source tables, and then checks that a dbt-only run starting from it scores 1.0 too. The same file checks the sandbox network: Snowflake and the Airbyte gate are reachable, Airbyte itself, HuggingFace, GitHub and PyPI are not, and through the gate a rollout can reach only its own Airbyte workspace.

Official tasks need the 256K-context model variant: with a 64K budget, our first `trains` rollout ran out of tokens after 7 turns. The rollout prints the reward breakdown and saves `result.json` and a transcript under `~/tinker-runs/eltbench_rollouts/`. On the final code, the base Qwen3.6-35B-A3B solved `trains` end to end: it configured Airbyte through the gate with Terraform, waited for both syncs, and wrote a dbt model that computes the answers from the loaded `TRAIN` and `CARS` tables. Reward 1.0 (SRDEL 1, SRDT 1, nothing flagged as hardcoded) in 40 turns and 421 s, of which 30.5 s was grading. An earlier full rollout, before the gate, finished the Airbyte stage but ran out of turns before dbt (reward 0.2), so one rollout per configuration says little about the success rate.

**3. A Tinker RL optimization step with execution-derived rewards.** Needs Tinker credentials. The toy run needs nothing else.

```bash
uv run python -m eltbench_rl.train bench_dir=tests/fixtures/toy_bench destination=duckdb tasks=shop \
    group_size=4 max_turns=20 max_steps=1
# official task, transform stage (needs the TRAINS__SNAPSHOT saved by the oracle test):
uv run python -m eltbench_rl.train tasks=trains destination=snowflake stage=transform \
    credential_path=third_party/ELT-Bench/setup/destination/snowflake_credential.json \
    group_size=4 max_turns=30 num_epochs=3   # one task, so each epoch is one step
```

In our toy run the group's rewards were `[1.0, 1.0, 0.2, 1.0]`. On the official `trains` task (transform stage, final code with every anti-hacking check), a one-step run gave `[1.0, 0.9, 0.9, 0.2]`: 274 s in total, 261 s of it sampling (4 rollouts in parallel, 30 turns each), about 25 s of grading per rollout and a 9 s training step, on 0.96M prompt and 74K sampled tokens with the 64K model. A three-step run then gave means of 0.55, 0.375 and 0.2 (`[0.9, 0.2, 0.2, 0.9]`, `[0.2, 0.9, 0.2, 0.2]`, `[0.2, 0.2, 0.2, 0.2]`). The loop works across steps (steps 1 and 2 sample from the updated weights), but four samples of one task per step at the default LoRA learning rate (5e-4) is far too little to show learning, and the last group had no reward variance, so no gradient. Metrics go to `metrics.jsonl` under the log path (`env/all/reward/total`, `env/all/srdel`, `env/all/srdt`, `env/all/hardcoded_frac`, ...).

## Official setup (Snowflake + local Airbyte)

This follows [ELT-Bench's README](third_party/ELT-Bench/README.md), with fixes for current Airbyte. Credential files live in the ELT-Bench submodule and are never committed.

1. **Snowflake.** Use a [trial account](https://signup.snowflake.com/). Snowflake is phasing out password sign-in for service users on other accounts ([timeline](https://docs.snowflake.com/en/user-guide/security-mfa-rollout)), and both ELT-Bench's setup and our per-rollout users sign in with passwords (`TYPE = LEGACY_SERVICE`); trial accounts are exempt.
   - Create an admin user with a password that has `SYSADMIN` and `SECURITYADMIN`.
   - Put it in `third_party/ELT-Bench/setup/destination/snowflake_credential.json` as `{"account", "user", "password", "role": "sysadmin", "warehouse": "AIRBYTE_WAREHOUSE"}`.
   - Run ELT-Bench's `setup/destination/setup.sql` as that user. It creates `AIRBYTE_ROLE`, `AIRBYTE_USER` and `AIRBYTE_WAREHOUSE`.
2. **Airbyte.** Give Docker about 12 GB of memory, then run `abctl local install --port 8001 --values docker/airbyte-values.yaml`.
   - The values file turns Airbyte's own auth off. Airbyte 2.x rejects the username/password API login that ELT-Bench's Terraform provider and job scripts use, and this instance only listens on localhost.
   - If you apply the values to an existing install, restart the services afterwards, or job launches fail with `Unauthorized`:
     ```bash
     docker exec airbyte-abctl-control-plane kubectl --kubeconfig /etc/kubernetes/admin.conf -n airbyte-abctl rollout restart deployment
     ```
3. **ELT-Bench connector.** Import `third_party/ELT-Bench/setup/elt_snowflake.yaml` in the Airbyte UI (Builder → Import YAML → Publish).
   - If publishing fails with `No declarative manifest image version found ... major version 6`, Docker Hub rate-limited Airbyte's version lookup. Register the image Airbyte would have found:
     ```bash
     docker exec airbyte-abctl-control-plane kubectl --kubeconfig /etc/kubernetes/admin.conf -n airbyte-abctl exec airbyte-db-0 -- \
       psql -U airbyte -d db-airbyte -c "insert into declarative_manifest_image_version (major_version, image_version, image_sha) values (6, '6.33.4', 'sha256:8fd974de2be41d6acad43a3d43576f9940a04c4f289164a2b1a5846d6f676b65')"
     ```
   - Then fill `third_party/ELT-Bench/setup/airbyte/airbyte_credential.json` with your email, any password, the workspace ID and the connector's definition ID.
4. **Sources.** `trains` only needs the REST API:
   ```bash
   cd third_party/ELT-Bench/setup && uvx gdown 1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0 && unzip -o data_api.zip -d ../elt-docker/rest_api
   cd ../elt-docker && docker compose up -d python-rest-api   # or all services for other tasks
   docker network connect elt-docker_elt_network airbyte-abctl-control-plane
   ```
5. **Sandbox network.** `docker/sandbox-network.sh` (safe to run again). It creates an internal Docker network whose only exits are a proxy that lets HTTPS through to `*.snowflakecomputing.com`, and the Airbyte gate (`docker/airbyte-gate/`), which keeps each rollout inside its own Airbyte workspace and lets sources load only the files ELT-Bench's tasks list. It also writes the gate's signing secret to `~/.eltbench/airbyte_gate_secret`.
6. **Task inputs.** `cd third_party/ELT-Bench/setup && uv run --project ../../.. python write_config.py --destination snowflake`. Ground-truth CSVs download automatically from HuggingFace (`tttjjj/elt_bench`) on first use.

## How it works

**What gets graded.** The submission is the final state of the rollout: the source tables in its namespace, the Airbyte syncs, and the dbt project under `/workspace/elt`, which grading runs again. So a project that builds counts even if the agent's own last `dbt run` didn't happen, which is more lenient than ELT-Bench's evaluation of the warehouse as the agent left it, and is needed anyway so that only tables our run builds can score.

**Episode.** Each rollout gets a copy of the task bundle (without admin credentials), its own warehouse namespace `<task>__<uuid8>`, its own Airbyte workspace, and its own container. On Snowflake the namespace is a database owned by a throwaway role and user made for this rollout, and their credentials replace the shared `AIRBYTE_USER` in the rollout's `config.yaml`. The Airbyte block of `config.yaml` points at the gate instead of Airbyte, with the workspace ID as the username and an HMAC of it as the password. The container sits on the internal `eltbench-sandbox` network (the gate, plus Snowflake through the proxy), or on no network at all for DuckDB. The model's only tool is the cookbook's `HarborBashTool`. The episode ends when the model replies without a tool call, which submits the pipeline, or when it hits 100 turns, the token budget, or too many malformed tool calls in a row. Cleanup runs `terraform destroy`, deletes the Airbyte workspace, drops the namespace (and its role and user) and removes the container.

**Transform stage.** With `stage=transform`, a rollout skips extract/load. Its namespace is a zero-copy Snowflake clone of the task's reference snapshot (for DuckDB, a copy of a small file), and its prompt is only stage 2 of the instruction. Snapshots hold only source tables, are saved by a trusted full run with `save_snapshot=true` (the oracle test) once it passes stage 1, and never enter the sandbox. This skips Airbyte's ~3.5 minutes per rollout and puts the training signal on dbt, where the model fails: in a group where every rollout finishes the load and none finishes dbt, all rewards are 0.2 and GRPO learns nothing.

**Reward.** `0.2·el_frac + 0.1·built_frac + 0.7·col_frac`, where `el_frac` is the share of source tables loaded with the right row count and exactly the columns of the task's schema file and, for Airbyte tasks, synced by a job that Airbyte's API reports as succeeded into this rollout's namespace, `built_frac` is the share of models our own dbt re-run built, and `col_frac` is the share of ground-truth columns that match, using ELT-Bench's `check_corretness`. As in ELT-Bench, the last two terms only count once every source table is loaded. ELT-Bench's own metrics (SRDEL, SRDT) are logged too.

**Anti-hacking.**
- Hidden data never enters the sandbox, and the sandbox can't download it: it reaches only Snowflake and the Airbyte gate.
- Each rollout's Snowflake user can only see its own database, and through the gate it can only see and change its own Airbyte workspace. So it can't read or sabotage a sibling's tables or connections, which in GRPO would raise its own advantage.
- Stage 1 asks Airbyte which streams it actually synced into this namespace, so loading around Airbyte doesn't count, and checks each source has exactly its schema's columns, so a source can't be pointed at another file with the same row count (for `trains`, the public ground-truth CSV).
- Before grading, the namespace is rebuilt from a snapshot of only the source tables, and our own dbt run rebuilds every model from them. Hand-made tables, helper tables and staged files don't survive, and forged dbt artifacts and seeds are rejected. The snapshot is the task's reference load when one exists (always in transform mode), so source values edited after loading don't count either.
- If anything matches the ground truth, dbt runs once more on the sources cut down to one row each. A model that returns the same rows from one row of data as from all of it has its answers typed in, and counts as not built. This also catches answers hidden behind a check that the sources aren't empty, which a run on empty sources would miss.
- Airbyte can reach the internet (`trains` loads a file from Google Drive), so the gate only lets a source load the files ELT-Bench's tasks list, or hosts on the local Docker network. A source pointed at the public ground truth gets a 403.

Still open: a full-mode run without a reference load can still edit its loaded sources (the reference closes this). A model could also compute the answers itself during the episode and type them in behind a check that looks at the data (for example `WHERE (SELECT COUNT(*) FROM orders) > 1`). Without a reference solution to run on changed data, an outcome check can't tell that apart from a real transformation, but it still means the model computed the answer from the sources.

## Adding a warehouse

Write a class with the `Destination` protocol from `eltbench_rl/destinations.py` (`name`, `namespace_field`, `connector()`, `create_namespace()`, `drop_namespace()`, `save_snapshot()`, `has_snapshot()` and `reset_namespace()`, for the clean grading runs and the transform stage), register it in `get_destination`, and generate task inputs with ELT-Bench's `write_config.py --destination <name>`. ELT-Bench already has `DatabricksConnector` and `RedshiftConnector` in `evaluation/db_connectors.py` for grading. The environment code doesn't change. Only Snowflake and DuckDB are implemented and tested.

## Layout

```
eltbench_rl/
  tasks.py           EltTask + load_tasks (ELT-Bench layout; GT from HuggingFace)
  destinations.py    Destination protocol; Snowflake and DuckDB
  reward.py          sources_loaded() (stage 1), score() (stage 2), EltReward (dbt re-runs, then score)
  airbyte.py         per-rollout Airbyte workspaces and the stage-1 sync check
  docker_sandbox.py  tinker-cookbook SandboxInterface over the docker CLI
  env.py             EltEnvGroupBuilder, dataset and dataset builder (mirrors harbor_rl)
  train.py           RL training CLI (mirrors harbor_rl/train.py)
  rollout.py         single-rollout CLI (mirrors harbor_rl/eval.py)
docker/              sandbox Dockerfile, Terraform provider mirror config, Airbyte values,
                     egress proxy (proxy/), Airbyte gate (airbyte-gate/) and sandbox-network.sh
tests/               unit tests, credential-free integration test, official oracle test
third_party/ELT-Bench  submodule, pinned at fcf3129
```

## Credits

- [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) (CC BY-SA 4.0): tasks, documentation, setup, the evaluation comparator and connectors (imported from the submodule), and the task prompt (adapted from `agents/codex_agent/run.py`).
- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (Apache-2.0): the RL loop, tool-use environment, bash tool and sandbox interface.
