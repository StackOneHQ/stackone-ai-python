"""The suite cannot reach the network: a test that did would pass or fail on a real host."""

from __future__ import annotations

import httpx
import pytest


def test_a_request_to_a_real_host_is_refused(_no_network: list[str]) -> None:
    # Which name the guard sees depends on the connect path (a resolved IP, say), so only the
    # refusal is pinned. No proxy, so the attempt is at the host itself.
    with (
        httpx.Client(trust_env=False) as client,
        pytest.raises(RuntimeError, match="must not reach the network"),
    ):
        client.get("https://api.stackone.com/accounts")
    assert _no_network
    # Cleared, so the fixture does not also fail this test for the attempt it expected.
    _no_network.clear()


def test_loopback_is_still_reachable(mcp_mock_server: str) -> None:
    assert httpx.get(mcp_mock_server).status_code < 500
