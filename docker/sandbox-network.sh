#!/bin/sh
# One-time setup for Snowflake runs. Sandboxes join an internal network with no route to the
# internet. The only things on it are a proxy that lets HTTPS through to Snowflake, and a gate in
# front of Airbyte that keeps each rollout in its own Airbyte workspace. So an agent can't download
# the public ground truth, or touch another rollout's connections.
set -e
cd "$(dirname "$0")"
docker build -t eltbench-proxy proxy/
docker build -t eltbench-airbyte-gate airbyte-gate/
docker network create --internal eltbench-sandbox
docker run -d --name eltbench-proxy --restart unless-stopped eltbench-proxy
docker network connect eltbench-sandbox eltbench-proxy

# The host signs each rollout's Airbyte password with this secret (eltbench_rl/airbyte.py).
mkdir -p ~/.eltbench
[ -f ~/.eltbench/airbyte_gate_secret ] || (umask 077 && openssl rand -hex 32 > ~/.eltbench/airbyte_gate_secret)
docker run -d --name airbyte-gate --restart unless-stopped --network kind \
  -e GATE_SECRET="$(cat ~/.eltbench/airbyte_gate_secret)" eltbench-airbyte-gate
docker network connect eltbench-sandbox airbyte-gate
