"""Action name normalization utilities."""

from __future__ import annotations

import re

_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")


def _normalize_action_name(composite_id: str) -> str:
    """Convert semantic search API composite ID to MCP tool name.

    Composite ID format: {connector}_{version}_{actionId}_{projectId}

    Examples:
        'slack_1.0.0_send_message_global'         -> 'slack_send_message'
        'calendly_1.0.0_calendly_list_events_global' -> 'calendly_list_events'
        'jira_1.0.0_search_issues_103/dev-56501'  -> 'jira_search_issues'
        'bamboohr_create_employee'                 -> 'bamboohr_create_employee' (unchanged)
    """
    parts = composite_id.split("_")

    # Find the version segment (e.g. "1.0.0")
    version_idx = None
    for i, part in enumerate(parts):
        if _VERSION_RE.match(part):
            version_idx = i
            break

    if version_idx is None or version_idx < 1:
        return composite_id

    connector = "_".join(parts[:version_idx])

    # Everything after version, excluding the last segment (projectId)
    after_version = parts[version_idx + 1 :]
    if len(after_version) < 2:
        return composite_id

    action_parts = after_version[:-1]  # drop projectId (last segment)
    action_id = "_".join(action_parts)

    # If action_id already starts with connector prefix, don't duplicate
    if action_id.startswith(f"{connector}_"):
        return action_id
    return f"{connector}_{action_id}"
