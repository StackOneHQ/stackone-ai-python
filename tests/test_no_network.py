"""The suite cannot reach the network: a test that did would pass or fail on a real host."""

from __future__ import annotations

import httpx
import pytest


def test_a_request_to_a_real_host_is_refused_naming_it(_no_network: list[str]) -> None:
    with pytest.raises(RuntimeError, match="'api.stackone.com'"):
        httpx.get("https://api.stackone.com/accounts")
    assert _no_network == ["api.stackone.com"]
    # Cleared, so the fixture does not also fail this test for the attempt it expected.
    _no_network.clear()


def test_loopback_is_still_reachable(mcp_mock_server: str) -> None:
    assert httpx.get(mcp_mock_server).status_code < 500
