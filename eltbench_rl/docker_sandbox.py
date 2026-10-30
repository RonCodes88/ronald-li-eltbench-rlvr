"""A tinker-cookbook sandbox backed by a local Docker container.

The cookbook ships Modal and SandboxFusion sandboxes, but ELT-Bench rollouts must reach services on
the local Docker network (the benchmark's sources and Airbyte), so each rollout gets a plain
container, the same way ELT-Bench's own agent runners do it.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from pathlib import Path

from tinker_cookbook.sandbox import SandboxResult

DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024


async def _docker(*args: str, stdin: bytes | None = None) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(stdin)
    assert proc.returncode is not None
    return proc.returncode, stdout, stderr


class DockerSandbox:
    """SandboxInterface over `docker run ... sleep infinity` and `docker exec`."""

    def __init__(self, container: str):
        self.container = container

    @classmethod
    async def create(
        cls, image: str, workspace: Path, network: str | None, env: dict[str, str] | None = None
    ) -> DockerSandbox:
        container = f"eltbench-{uuid.uuid4().hex[:8]}"
        args = ["run", "-d", "--rm", "--name", container]
        args += ["-v", f"{workspace.resolve()}:/workspace", "-w", "/workspace"]
        if network:
            args += ["--network", network]
        for name, value in (env or {}).items():
            args += ["-e", f"{name}={value}"]
        code, _, stderr = await _docker(*args, image, "sleep", "infinity")
        if code != 0:
            raise RuntimeError(f"docker run failed: {stderr.decode(errors='replace')}")
        return cls(container)

    @property
    def sandbox_id(self) -> str:
        return self.container

    async def send_heartbeat(self, timeout: int = 30) -> None:
        pass

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        # `timeout` runs inside the container: killing the docker client would leave the command running.
        code, stdout, stderr = await _docker(
            "exec", "-w", workdir or "/workspace", self.container,
            "timeout", "-k", "5", str(timeout), "bash", "-lc", command,
        )  # fmt: skip
        cap = max_output_bytes or DEFAULT_MAX_OUTPUT_BYTES
        return SandboxResult(
            stdout=stdout[:cap].decode(errors="replace"),
            stderr=stderr[:cap].decode(errors="replace"),
            exit_code=code,
        )

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        return await self.run_command(f"cat {shlex.quote(path)}", timeout=timeout, max_output_bytes=max_bytes)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        quoted = shlex.quote(path)
        script = f'mkdir -p "$(dirname {quoted})" && cat > {quoted}'
        if executable:
            script += f" && chmod +x {quoted}"
        data = content.encode() if isinstance(content, str) else content
        code, stdout, stderr = await _docker(
            "exec", "-i", self.container, "timeout", str(timeout), "sh", "-c", script, stdin=data
        )
        return SandboxResult(
            stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"), exit_code=code
        )

    async def cleanup(self) -> None:
        await _docker("rm", "-f", self.container)
