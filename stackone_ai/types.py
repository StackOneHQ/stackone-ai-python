"""Shared types, configuration and errors for the StackOne AI SDK."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field

JsonDict: TypeAlias = dict[str, Any]
Headers: TypeAlias = dict[str, str]

# StackOne API base URL
DEFAULT_BASE_URL: str = "https://api.stackone.com"


ToolMode = Literal["individual", "search_execute"]
"""How the MCP endpoint lists tools.

``"individual"`` (the server default) lists one tool per action — hundreds per
account. ``"search_execute"`` lists two meta tools per connector instead, a
``*_search_actions`` that ranks actions for a natural-language query and an
``*_execute_action`` that runs one by id. The catalog stays small regardless of
how many accounts are linked, which is what keeps it inside a model's context.
"""

SUBMIT_FEEDBACK_TOOL_NAME: str = "stackone_submit_feedback"
"""The one global tool the MCP endpoint serves in every mode when feedback is enabled.

It is not a connector action, and it is global rather than account-scoped, so the toolset
keeps one copy of it however many accounts list it.
"""

FeedbackRating = Literal["positive", "negative", "neutral"]
FeedbackSource = Literal["model", "user", "system"]
FeedbackCategory = Literal["search", "execute", "defender", "connection", "general"]


class StackOneError(Exception):
    """Base exception for StackOne errors"""

    pass


class StackOneAPIError(StackOneError):
    """Raised when the StackOne API returns an error"""

    def __init__(self, message: str, status_code: int, response_body: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ToolArgumentsError(StackOneError, ValueError):
    """Raised when a tool's arguments are unusable: not JSON, not an object, or not encodable.

    Raised before any request is made. A subclass of both StackOneError, so
    ``except StackOneError`` catches it, and ValueError, which is what these errors were
    before, so existing ``except ValueError`` clauses still do.
    """

    pass


class ToolsetError(StackOneError):
    """Base exception for toolset errors.

    A subclass of StackOneError, so ``except StackOneError`` catches everything this
    SDK raises. The two used to be unrelated siblings, which meant the obvious
    catch-all silently missed ToolsetConfigError and ToolsetLoadError — the errors a
    user is most likely to hit on their very first call. Existing
    ``except ToolsetError`` clauses are unaffected.
    """

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


class ExecuteConfig(BaseModel):
    """How a tool is called: its name, any extra request headers, and the timeout.

    Every tool executes over MCP ``tools/call``, so there is no HTTP method, URL, body
    type or parameter location to configure. Extra fields are refused rather than ignored,
    so code still passing them finds out here instead of assuming they took effect.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool name")
    headers: Headers = Field(
        default_factory=dict,
        description="Extra HTTP headers for the MCP request. Authorization, x-account-id and "
        "User-Agent are always the SDK's own and cannot be replaced here.",
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")


class ToolParameters(BaseModel):
    """Schema definition for tool parameters.

    ``properties`` is a faithful mirror of the ``inputSchema`` the MCP server served,
    with the SDK's internal ``nullable`` marker added per property. Every other root
    keyword the server sent, ``required`` included, is kept verbatim as an extra field.
    Consumers that need the raw served schema (for example the ADK plugin) read this
    directly, so nothing here may be invented or dropped.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="JSON Schema type")
    properties: JsonDict = Field(description="JSON Schema properties")
