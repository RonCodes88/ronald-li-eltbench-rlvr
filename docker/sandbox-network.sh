#!/bin/sh
# One-time setup for Snowflake runs. Sandboxes join an internal network with no route to the
# internet; the only things on it are the Airbyte control plane and a proxy that lets HTTPS
# through to Snowflake. So an agent can't download the public ground truth, or anything else.
set -e
cd "$(dirname "$0")"
docker build -t eltbench-proxy proxy/
docker network create --internal eltbench-sandbox
docker run -d --name eltbench-proxy --restart unless-stopped eltbench-proxy
docker network connect eltbench-sandbox eltbench-proxy
docker network connect eltbench-sandbox airbyte-abctl-control-plane
