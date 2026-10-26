import asyncio
import json
import shutil
import subprocess
import time

import duckdb
import pytest
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool
from tinker_cookbook.tool_use import ToolInput

from eltbench_rl import ELT_BENCH_DIR
from eltbench_rl.docker_sandbox import DockerSandbox

IMAGE = "eltbench-rl"


def image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not image_available(), reason=f"needs Docker and the {IMAGE} image: docker build -t {IMAGE} docker/"
)


def in_sandbox(tmp_path, fn, network=None):
    """Run `fn(sandbox)` in a fresh container and always remove it afterwards."""

    async def main():
        sandbox = await DockerSandbox.create(IMAGE, tmp_path, network)
        try:
            return await fn(sandbox)
        finally:
            await sandbox.cleanup()

    return asyncio.run(main())


def test_run_command_output_and_exit_code(tmp_path):
    result = in_sandbox(tmp_path, lambda sb: sb.run_command("echo hi; exit 3"))

    assert result.stdout == "hi\n"
    assert result.exit_code == 3


def test_timeout_kills_the_command_inside_the_container(tmp_path):
    async def check(sandbox):
        start = time.monotonic()
        result = await sandbox.run_command("sleep 30", timeout=2)
        elapsed = time.monotonic() - start
        leftover = await sandbox.run_command("ps -eo args | grep -c '^sleep 30$' || true")
        return result, elapsed, leftover

    result, elapsed, leftover = in_sandbox(tmp_path, check)
    assert result.exit_code == 124
    assert elapsed < 10
    assert leftover.stdout.strip() == "0"


def test_write_and_read_file_roundtrip(tmp_path):
    content = 'line "one"\n$HOME and `ticks`\n'

    async def check(sandbox):
        await sandbox.write_file("/workspace/nested/dir/file.txt", content)
        return await sandbox.read_file("/workspace/nested/dir/file.txt")

    assert in_sandbox(tmp_path, check).stdout == content
    assert (tmp_path / "nested" / "dir" / "file.txt").read_text() == content


def test_image_has_dbt_adapters(tmp_path):
    result = in_sandbox(tmp_path, lambda sb: sb.run_command("dbt --version", timeout=120))

    assert "duckdb" in result.stdout and "snowflake" in result.stdout


def test_duckdb_version_matches_host(tmp_path):
    cmd = "python -c 'import duckdb; print(duckdb.__version__)'"
    result = in_sandbox(tmp_path, lambda sb: sb.run_command(cmd))

    assert result.stdout.strip() == duckdb.__version__


def test_terraform_init_works_offline(tmp_path):
    main_tf = (ELT_BENCH_DIR / "setup" / "main.tf").read_text()

    async def check(sandbox):
        await sandbox.write_file("/workspace/elt/main.tf", main_tf)
        return await sandbox.run_command("cd /workspace/elt && terraform init -input=false", timeout=120)

    result = in_sandbox(tmp_path, check, network="none")
    assert result.exit_code == 0, result.stdout + result.stderr


def test_cookbook_bash_tool_runs_in_sandbox(tmp_path):
    async def check(sandbox):
        return await HarborBashTool(sandbox).bash.run(ToolInput(arguments={"command": "ls /workspace"}))

    result = in_sandbox(tmp_path, check)
    assert json.loads(result.messages[0]["content"])["exit_code"] == 0


def test_cleanup_removes_container(tmp_path):
    async def get_id(sandbox):
        return sandbox.sandbox_id

    container = in_sandbox(tmp_path, get_id)
    listed = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container}", "-q"], capture_output=True, text=True
    )
    assert listed.stdout.strip() == ""
