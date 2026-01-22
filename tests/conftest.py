"""Pytest configuration and fixtures for StackOne AI tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for a server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="session")
def mcp_mock_server() -> Generator[str, None, None]:
    """
    Start the Node MCP mock server for integration tests.

    This fixture starts the Hono-based MCP mock server using bun,
    importing from the stackone-ai-node submodule.

    Requires: bun to be installed (via Nix flake).

    Usage:
        def test_mcp_integration(mcp_mock_server):
            toolset = StackOneToolSet(
                api_key="test-key",
                base_url=mcp_mock_server,
            )
            tools = toolset.fetch_tools()
    """
    project_root = Path(__file__).parent.parent
    serve_script = project_root / "tests" / "mocks" / "serve.ts"
    vendor_dir = project_root / "vendor" / "stackone-ai-node"

    if not serve_script.exists():
        pytest.skip("MCP mock server script not found at tests/mocks/serve.ts")

    if not vendor_dir.exists():
        pytest.skip("stackone-ai-node submodule not found. Run 'git submodule update --init'")

    # find port
    port = _find_free_port()
    base_url = f"http://localhost:{port}"

    # Start the server from project root
    env = os.environ.copy()
    env["PORT"] = str(port)

    process = subprocess.Popen(
        [str(serve_script)],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        if not _wait_for_server("localhost", port, timeout=30.0):
            stdout, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"MCP mock server failed to start:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        yield base_url

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture
def mcp_server_url(mcp_mock_server: str) -> str:
    """Alias for mcp_mock_server for clearer test naming."""
    return mcp_mock_server
