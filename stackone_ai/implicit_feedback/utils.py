from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SENSITIVE_KEYS = {"password", "token", "secret", "authorization", "api_key"}
MAX_STRING_LENGTH = 512
MAX_ITEMS = 25
MAX_DEPTH = 4


def sanitize_value(value: Any, *, depth: int = 0) -> Any:
    """Strip large or sensitive data from payloads before logging."""

    if depth > MAX_DEPTH:
        return "<truncated>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) <= MAX_STRING_LENGTH:
            return value
        return value[:MAX_STRING_LENGTH] + "…"

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_ITEMS]:
            key_str = str(key)
            if key_str.lower() in SENSITIVE_KEYS:
                sanitized[key_str] = "<redacted>"
                continue
            sanitized[key_str] = sanitize_value(item, depth=depth + 1)
        if len(value) > MAX_ITEMS:
            sanitized["_truncated"] = True
        return sanitized

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        sanitized_sequence = [sanitize_value(item, depth=depth + 1) for item in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            sanitized_sequence.append("<truncated>")
        return sanitized_sequence

    return str(value)


def sanitize_payload(payload: Any) -> Any:
    return sanitize_value(payload)

