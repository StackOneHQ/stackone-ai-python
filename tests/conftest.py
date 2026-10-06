"""Pytest configuration and fixtures for StackOne AI tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_account_id_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A STACKONE_ACCOUNT_ID in the developer's shell would make every toolset warn."""
    monkeypatch.delenv("STACKONE_ACCOUNT_ID", raising=False)


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """A test that reaches a real host passes or fails on that host, not on the code.

    Loopback stays open for the MCP mock server. The refusal is raised, and the test also
    fails afterwards: code under test that swallows a failed request would otherwise hide it.
    Yields the hosts refused so far.
    """
    refused: list[str] = []

    def _refuse_unless_loopback(address: object) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOOPBACK_HOSTS:
            refused.append(host)
            raise RuntimeError(
                f"Tests must not reach the network: a connection to {host!r} was refused. Stub it."
            )

    connect, connect_ex, create_connection = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
    )

    def guarded_connect(self: socket.socket, address: Any) -> None:
        _refuse_unless_loopback(address)
        return connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        _refuse_unless_loopback(address)
        return connect_ex(self, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        _refuse_unless_loopback(address)
        return create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    yield refused
    if refused:
        pytest.fail(
            f"Tests must not reach the network: connections to {', '.join(refused)} were refused. Stub them."
        )


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


@pytest.fixture(scope="session")
def mcp_mock_server_with_end_users() -> Generator[str, None, None]:
    """The same mock, listing acc1 as non-shared (end user ``end-user-1``) and acc2 as shared.

    Like the real API, it refuses an MCP request for acc1 that lacks acc1's ``x-end-user-id``.
    """
    with _run_mcp_mock_server({"MOCK_END_USERS": "on"}) as base_url:
        yield base_url
