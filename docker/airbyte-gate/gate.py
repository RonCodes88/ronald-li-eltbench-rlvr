"""Airbyte gate: each rollout can use only its own Airbyte workspace.

Local Airbyte (community edition) has no per-workspace permissions, and its API is open to anyone
who can reach it. So sandboxes can't reach Airbyte; they reach this gate. A rollout signs in with its
workspace id as the username and HMAC(GATE_SECRET, workspace id) as the password, which only the
host can compute. A request goes through only if every workspace, source, destination, connection
and job it names belongs to that workspace, and list requests are narrowed to it.

Airbyte itself can reach the internet (tasks load flat files from Google Drive), so a source may
only fetch the files ELT-Bench's tasks list (ALLOWED_URLS, set by docker/sandbox-network.sh) or
hosts on the local Docker network. Otherwise a rollout could load the public ground truth.
"""

import base64
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AIRBYTE = os.environ.get("AIRBYTE_URL", "http://airbyte-abctl-control-plane:80")
SECRET = os.environ["GATE_SECRET"].encode()
API = "/api/public/v1/"
RESOURCES = {"sources", "destinations", "connections", "jobs", "streams", "health"}
LISTS = {"sources", "destinations", "connections", "jobs"}
ID_FIELDS = {"sourceId": "sources", "destinationId": "destinations", "connectionId": "connections"}
ALLOWED_URLS = set(os.environ.get("ALLOWED_URLS", "").split())


def password(workspace_id: str) -> str:
    return hmac.new(SECRET, workspace_id.encode(), hashlib.sha256).hexdigest()


def forward(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request = urllib.request.Request(AIRBYTE + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def workspace_of(kind: str, resource_id: str) -> str | None:
    status, body = forward("GET", f"{API}{kind}/{urllib.parse.quote(resource_id)}")
    if status != 200:
        return None
    resource = json.loads(body)
    if kind == "jobs":
        return workspace_of("connections", resource["connectionId"])
    return resource.get("workspaceId")


def fetches_unlisted_url(value, key: str = "") -> bool:
    """Whether a source configuration points at a file or remote host outside ELT-Bench's tasks."""
    if isinstance(value, dict):
        return any(fetches_unlisted_url(v, k) for k, v in value.items())
    if isinstance(value, list):
        return any(fetches_unlisted_url(v) for v in value)
    if not isinstance(value, str) or value in ALLOWED_URLS:
        return False
    # The File source's `url` may leave out the scheme. Hosts without a dot are local containers.
    host = urllib.parse.urlsplit(value).hostname or ""
    return key.lower() == "url" or ("://" in value and "." in host)


def allowed(workspace: str, method: str, kind: str, resource_id: str, named: dict) -> bool:
    if kind == "workspaces":
        return method == "GET" and resource_id == workspace
    if kind not in RESOURCES:
        return False
    if resource_id and workspace_of(kind, resource_id) != workspace:
        return False
    if named.get("workspaceId", workspace) != workspace:
        return False
    return all(workspace_of(ID_FIELDS[field], named[field]) == workspace for field in ID_FIELDS if field in named)


class Gate(BaseHTTPRequestHandler):
    def handle_any(self):
        url = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(url.query))
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {}
        named = {**query, **(payload if isinstance(payload, dict) else {})}

        workspace = self.signed_in_workspace()
        kind, _, resource_id = url.path.removeprefix(API).strip("/").partition("/")
        if workspace is None:
            return self.reply(401, {"message": "sign in with the Airbyte username and password in config.yaml"})
        if not url.path.startswith(API) or not allowed(workspace, self.command, kind, resource_id, named):
            return self.reply(403, {"message": "outside this rollout's Airbyte workspace"})
        if kind == "sources" and self.command != "GET" and fetches_unlisted_url(named.get("configuration")):
            return self.reply(403, {"message": "sources may only load the files and hosts given in /workspace/config.yaml"})
        query.pop("workspaceIds", None)
        if kind in LISTS and not resource_id and "connectionId" not in query:
            query["workspaceIds"] = workspace
        status, response = forward(self.command, f"{url.path}?{urllib.parse.urlencode(query)}", body)
        self.reply(status, response)

    do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = handle_any

    def signed_in_workspace(self) -> str | None:
        scheme, _, credentials = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "basic":
            return None
        try:
            user, _, given = base64.b64decode(credentials).decode().partition(":")
        except ValueError:
            return None
        return user if hmac.compare_digest(given, password(user)) else None

    def reply(self, status: int, body: bytes | dict):
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("", 80), Gate).serve_forever()
