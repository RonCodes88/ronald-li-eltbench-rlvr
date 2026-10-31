#!/bin/sh
# Setup for Snowflake runs; safe to run again (it replaces the proxy and gate containers).
# Sandboxes join an internal network with no route to the internet. The only things on it are a
# proxy that lets HTTPS through to Snowflake, and a gate in front of Airbyte that keeps each rollout
# in its own Airbyte workspace. So an agent can't download the public ground truth, or touch another
# rollout's connections.
set -e
cd "$(dirname "$0")"
docker build -t eltbench-proxy proxy/
docker build -t eltbench-airbyte-gate airbyte-gate/
docker network inspect eltbench-sandbox >/dev/null 2>&1 || docker network create --internal eltbench-sandbox
docker rm -f eltbench-proxy airbyte-gate >/dev/null 2>&1 || true

docker run -d --name eltbench-proxy --restart unless-stopped eltbench-proxy
docker network connect eltbench-sandbox eltbench-proxy

# The host signs each rollout's Airbyte password with this secret (eltbench_rl/airbyte.py).
mkdir -p ~/.eltbench
[ -f ~/.eltbench/airbyte_gate_secret ] || (umask 077 && openssl rand -hex 32 > ~/.eltbench/airbyte_gate_secret)
# Sources may only fetch the flat files ELT-Bench's tasks list.
ALLOWED_URLS="$(grep -rhoE 'path: [a-z0-9]+://[^ ]+' ../third_party/ELT-Bench/elt-bench/*/*/config.yaml | sed 's/^path: //' | sort -u | tr '\n' ' ')"
docker run -d --name airbyte-gate --restart unless-stopped --network kind \
  -e GATE_SECRET="$(cat ~/.eltbench/airbyte_gate_secret)" -e ALLOWED_URLS="$ALLOWED_URLS" eltbench-airbyte-gate
docker network connect eltbench-sandbox airbyte-gate
