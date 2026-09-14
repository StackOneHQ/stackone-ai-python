"""Shared types, configuration and errors for the StackOne AI SDK."""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any, TypeAlias, TypedDict
from urllib.parse import unquote

from pydantic import BaseModel, BeforeValidator, Field

JsonDict: TypeAlias = dict[str, Any]
Headers: TypeAlias = dict[str, str]

# StackOne API base URL
DEFAULT_BASE_URL: str = "https://api.stackone.com"


class StackOneError(Exception):
    """Base exception for StackOne errors"""

    pass


class StackOneAPIError(StackOneError):
    """Raised when the StackOne API returns an error"""

    def __init__(self, message: str, status_code: int, response_body: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ToolsetError(Exception):
    """Base exception for toolset errors"""

    pass


class ToolsetConfigError(ToolsetError):
    """Raised when there is an error in the toolset configuration"""

    pass


class ToolsetLoadError(ToolsetError):
    """Raised when there is an error loading tools"""

    pass


class ExecuteToolsConfig(TypedDict, total=False):
    """Execution configuration for the StackOneToolSet constructor.

    Controls default account scoping and timeout for tool execution.
    """

    account_ids: list[str]
    """Account IDs to scope tool discovery and execution."""

    timeout: float
    """Request timeout in seconds. Default: 60. Can also be set as a top-level
    constructor param which takes precedence."""


class ParameterLocation(str, Enum):
    """Valid locations for parameters in requests"""

    HEADER = "header"
    QUERY = "query"
    PATH = "path"
    BODY = "body"
    FILE = "file"  # For file uploads


def validate_method(v: str) -> str:
    """Validate HTTP method is uppercase and supported"""
    method = v.upper()
    if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
        raise ValueError(f"Unsupported HTTP method: {method}")
    return method


def is_json_content_type(content_type: str) -> bool:
    """Whether a response body should be parsed as JSON based on its Content-Type.

    Only genuine JSON media types are parsed (``application/json`` and structured
    suffixes such as ``application/problem+json``). Anything else - including a
    missing Content-Type - is treated as opaque content (a file download), so the
    raw bytes are returned instead of being force-decoded as UTF-8/JSON. This mirrors
    how the StackOne generated SDKs default unknown bodies to ``application/octet-stream``.
    """
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def filename_from_content_disposition(value: str | None) -> str | None:
    """Extract the filename from a Content-Disposition header value, if present.

    Handles both the plain ``filename="example.pdf"`` form and the RFC 5987 extended
    ``filename*=UTF-8''example%20file.pdf`` form (which takes precedence when present).
    The extended form is percent-decoded using its declared charset (RFC 5987 permits
    both ``UTF-8`` and ``ISO-8859-1``); an unknown or empty charset falls back to UTF-8.
    """
    if not value:
        return None
    extended = re.search(r"filename\*\s*=\s*([^']*)'[^']*'([^;]+)", value, re.IGNORECASE)
    if extended:
        charset = extended.group(1).strip() or "utf-8"
        encoded = extended.group(2).strip().strip('"')
        try:
            return unquote(encoded, encoding=charset, errors="replace") or None
        except LookupError:
            # Unrecognised charset label - decode as UTF-8 rather than failing.
            return unquote(encoded, encoding="utf-8", errors="replace") or None
    quoted = re.search(r'filename\s*=\s*"([^"]*)"', value, re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip() or None
    bare = re.search(r"filename\s*=\s*([^;]+)", value, re.IGNORECASE)
    if bare:
        return bare.group(1).strip().strip('"') or None
    return None


class ExecuteConfig(BaseModel):
    """Configuration for executing a tool against an API endpoint"""

    headers: Headers = Field(default_factory=dict, description="HTTP headers to include in the request")
    method: Annotated[str, BeforeValidator(validate_method)] = Field(description="HTTP method to use")
    url: str = Field(description="API endpoint URL")
    name: str = Field(description="Tool name")
    body_type: str | None = Field(default=None, description="Content type for request body")
    parameter_locations: dict[str, ParameterLocation] = Field(
        default_factory=dict, description="Maps parameter names to their location in the request"
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")


class ToolParameters(BaseModel):
    """Schema definition for tool parameters.

    ``properties`` is a faithful mirror of the ``inputSchema`` the MCP server served,
    with the SDK's internal ``nullable`` marker added per property. Consumers that
    need the raw served schema (for example the ADK plugin) read this directly, so
    nothing here may be invented or dropped.
    """

    type: str = Field(description="JSON Schema type")
    properties: JsonDict = Field(description="JSON Schema properties")
