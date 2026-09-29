"""Pytest configuration and fixtures for StackOne AI tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
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


@contextmanager
def _run_mcp_mock_server(extra_env: dict[str, str] | None = None) -> Iterator[str]:
    """Start the Node MCP mock server, yield its base URL, and stop it afterwards."""
    project_root = Path(__file__).parent.parent
    serve_script = project_root / "tests" / "mocks" / "serve.ts"

    # Fail rather than skip: the mock server is committed to this repo, so a missing
    # script or missing node_modules is a broken checkout, not an absent optional
    # dependency. Skipping here previously let these tests vanish silently while CI
    # stayed green.
    if not serve_script.exists():
        pytest.fail(f"MCP mock server script missing at {serve_script}")

    if not (project_root / "node_modules").is_dir():
        pytest.fail("Node dependencies missing for the MCP mock server. Run 'pnpm install'.")

    # find port
    port = _find_free_port()
    base_url = f"http://localhost:{port}"

    # Start the server from project root
    env = os.environ.copy()
    env["PORT"] = str(port)
    env.update(extra_env or {})

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
            try:
                stdout, stderr = process.communicate(timeout=5)
                msg = (
                    f"MCP mock server failed to start:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
                )
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                msg = f"MCP mock server timed out:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            raise RuntimeError(msg)

        yield base_url

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="session")
def mcp_mock_server() -> Generator[str, None, None]:
    """
    Start the Node MCP mock server for integration tests.

    This fixture starts the Hono-based MCP mock server using tsx. It serves the
    global feedback tool, as a project with feedback enabled does.

    Requires: pnpm install (provides tsx and the Hono dependencies).

    Usage:
        def test_mcp_integration(mcp_mock_server):
            toolset = StackOneToolSet(
                api_key="test-key",
                base_url=mcp_mock_server,
            )
            tools = toolset.fetch_tools()
    """
    with _run_mcp_mock_server() as base_url:
        yield base_url


@pytest.fixture(scope="session")
def mcp_mock_server_without_feedback() -> Generator[str, None, None]:
    """The same mock, serving a project with feedback disabled: the tool is absent."""
    with _run_mcp_mock_server({"MOCK_SUBMIT_FEEDBACK": "off"}) as base_url:
        yield base_url
