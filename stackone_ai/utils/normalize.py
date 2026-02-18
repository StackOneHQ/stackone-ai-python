"""Action name normalization utilities."""

from __future__ import annotations

import re

_VERSIONED_ACTION_RE = re.compile(r"^[a-z][a-z0-9]*_\d+(?:\.\d+)+_(.+)_global$")


def _normalize_action_name(action_name: str) -> str:
    """Convert semantic search API action name to MCP tool name.

    API:  'calendly_1.0.0_calendly_create_scheduling_link_global'
    MCP:  'calendly_create_scheduling_link'
    """
    match = _VERSIONED_ACTION_RE.match(action_name)
    return match.group(1) if match else action_name
