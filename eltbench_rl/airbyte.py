"""One Airbyte workspace per rollout, reached through docker/airbyte-gate.

Local Airbyte has no per-workspace permissions, so sandboxes only reach the gate, which keeps each
rollout inside its own workspace. A rollout signs in with its workspace id as the username and an
HMAC of it as the password; the secret is shared by the host and the gate (docker/sandbox-network.sh).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.request
from pathlib import Path

API_URL = "http://localhost:8001/api/public/v1"  # as the host sees it (abctl local install --port 8001)
GATE_URL = "http://airbyte-gate:80/api/public/v1/"  # as sandboxes see it
SECRET_PATH = Path.home() / ".eltbench" / "airbyte_gate_secret"


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{API_URL}/{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read()
    return json.loads(text) if text else {}


def create_workspace(name: str) -> dict[str, str]:
    """Create a workspace; return the config.yaml values a rollout uses to reach it."""
    secret = SECRET_PATH.read_text().strip().encode()
    workspace_id = _request("POST", "workspaces", {"name": name})["workspaceId"]
    password = hmac.new(secret, workspace_id.encode(), hashlib.sha256).hexdigest()
    return {"workspace_id": workspace_id, "server_url": GATE_URL, "username": workspace_id, "password": password}


def delete_workspace(workspace_id: str) -> None:
    _request("DELETE", f"workspaces/{workspace_id}")


def synced_tables(workspace_id: str, namespace_field: str, namespace: str) -> set[str]:
    """Streams that Airbyte itself reports as successfully synced into `namespace`.

    Asks Airbyte directly instead of reading anything the agent wrote (like terraform.tfstate),
    so tables loaded around Airbyte don't count.
    """
    tables = set()
    for connection in _request("GET", f"connections?workspaceIds={workspace_id}&limit=100")["data"]:
        destination = _request("GET", f"destinations/{connection['destinationId']}")
        if str(destination["configuration"].get(namespace_field, "")).lower() != namespace.lower():
            continue
        if _request("GET", f"jobs?connectionId={connection['connectionId']}&jobType=sync&status=succeeded")["data"]:
            tables |= {stream["name"].lower() for stream in connection["configurations"]["streams"]}
    return tables
