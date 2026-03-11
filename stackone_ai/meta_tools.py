"""Meta tools (tool_search + tool_execute) for LLM-driven workflows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from stackone_ai.models import (
    ExecuteConfig,
    JsonDict,
    ParameterLocation,
    StackOneAPIError,
    StackOneError,
    StackOneTool,
    ToolParameters,
    Tools,
)

if TYPE_CHECKING:
    from stackone_ai.toolset import StackOneToolSet


class MetaToolsOptions(BaseModel):
    """Options for get_meta_tools()."""

    account_ids: list[str] | None = None
    search: Any | None = Field(default=None, description="Search mode: 'auto', 'semantic', or 'local'")
    connector: str | None = None
    top_k: int | None = None
    min_similarity: float | None = None


# --- tool_search ---


class SearchInput(BaseModel):
    """Input validation for tool_search."""

    query: str = Field(..., min_length=1)
    connector: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("query must be a non-empty string")
        return trimmed


class SearchMetaTool(StackOneTool):
    """LLM-callable tool that searches for available StackOne tools."""

    _toolset: Any = None
    _options: MetaToolsOptions = None  # type: ignore[assignment]

    def execute(
        self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
    ) -> JsonDict:
        try:
            if isinstance(arguments, str):
                raw_params = json.loads(arguments)
            else:
                raw_params = arguments or {}

            parsed = SearchInput(**raw_params)

            results = self._toolset.search_tools(
                parsed.query,
                connector=parsed.connector or self._options.connector,
                top_k=parsed.top_k or self._options.top_k or 5,
                min_similarity=self._options.min_similarity,
                search=self._options.search,
                account_ids=self._options.account_ids,
            )

            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters.properties,
                    }
                    for t in results
                ],
                "total": len(results),
                "query": parsed.query,
            }
        except json.JSONDecodeError as exc:
            raise StackOneError(f"Invalid JSON in arguments: {exc}") from exc
        except Exception as error:
            if isinstance(error, StackOneError):
                raise
            raise StackOneError(f"Error searching tools: {error}") from error


# --- tool_execute ---


class ExecuteInput(BaseModel):
    """Input validation for tool_execute."""

    tool_name: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("tool_name must be a non-empty string")
        return trimmed


class ExecuteMetaTool(StackOneTool):
    """LLM-callable tool that executes a StackOne tool by name."""

    _toolset: Any = None
    _options: MetaToolsOptions = None  # type: ignore[assignment]

    def execute(
        self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
    ) -> JsonDict:
        try:
            if isinstance(arguments, str):
                raw_params = json.loads(arguments)
            else:
                raw_params = arguments or {}

            parsed = ExecuteInput(**raw_params)

            all_tools = self._toolset.fetch_tools(account_ids=self._options.account_ids)
            target = all_tools.get_tool(parsed.tool_name)

            if target is None:
                return {
                    "error": f'Tool "{parsed.tool_name}" not found. Use tool_search to find available tools.',
                }

            return target.execute(parsed.parameters, options=options)
        except StackOneAPIError as exc:
            # Return API errors to the LLM so it can adjust parameters and retry
            return {
                "error": str(exc),
                "status_code": exc.status_code,
                "tool_name": parsed.tool_name if "parsed" in dir() else "unknown",
            }
        except json.JSONDecodeError as exc:
            raise StackOneError(f"Invalid JSON in arguments: {exc}") from exc
        except Exception as error:
            if isinstance(error, StackOneError):
                raise
            raise StackOneError(f"Error executing tool: {error}") from error


# --- Factory ---


def create_meta_tools(
    toolset: StackOneToolSet,
    options: MetaToolsOptions | None = None,
) -> Tools:
    """Create tool_search + tool_execute for LLM-driven workflows.

    Args:
        toolset: The StackOneToolSet to delegate search and execution to.
        options: Options to scope search and execution.

    Returns:
        Tools collection containing tool_search and tool_execute.
    """
    opts = options or MetaToolsOptions()
    api_key = toolset.api_key

    # tool_search
    search_tool = _create_search_tool(api_key, opts)
    search_tool._toolset = toolset
    search_tool._options = opts

    # tool_execute
    execute_tool = _create_execute_tool(api_key, opts)
    execute_tool._toolset = toolset
    execute_tool._options = opts

    return Tools([search_tool, execute_tool])


def _create_search_tool(api_key: str, opts: MetaToolsOptions) -> SearchMetaTool:
    name = "tool_search"
    description = (
        "Search for available tools by describing what you need. "
        "Returns matching tool names, descriptions, and parameter schemas. "
        "Use the returned parameter schemas to know exactly what to pass when calling tool_execute."
    )
    parameters = ToolParameters(
        type="object",
        properties={
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of what you need "
                    '(e.g. "create an employee", "list time off requests")'
                ),
            },
            "connector": {
                "type": "string",
                "description": 'Optional connector filter (e.g. "bamboohr", "hibob")',
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (1-50, default 5)",
                "minimum": 1,
                "maximum": 50,
            },
        },
    )
    execute_config = ExecuteConfig(
        name=name,
        method="POST",
        url="local://meta/search",
        parameter_locations={
            "query": ParameterLocation.BODY,
            "connector": ParameterLocation.BODY,
            "top_k": ParameterLocation.BODY,
        },
    )

    tool = SearchMetaTool.__new__(SearchMetaTool)
    StackOneTool.__init__(
        tool,
        description=description,
        parameters=parameters,
        _execute_config=execute_config,
        _api_key=api_key,
    )
    return tool


def _create_execute_tool(api_key: str, opts: MetaToolsOptions) -> ExecuteMetaTool:
    name = "tool_execute"
    description = (
        "Execute a tool by name with the given parameters. "
        "Use tool_search first to find available tools. "
        "The parameters field must match the parameter schema returned by tool_search. "
        "Pass parameters as a nested object matching the schema structure."
    )
    parameters = ToolParameters(
        type="object",
        properties={
            "tool_name": {
                "type": "string",
                "description": "Exact tool name from tool_search results",
            },
            "parameters": {
                "type": "object",
                "description": "Parameters for the tool. Pass {} if none needed.",
            },
        },
    )
    execute_config = ExecuteConfig(
        name=name,
        method="POST",
        url="local://meta/execute",
        parameter_locations={
            "tool_name": ParameterLocation.BODY,
            "parameters": ParameterLocation.BODY,
        },
    )

    tool = ExecuteMetaTool.__new__(ExecuteMetaTool)
    StackOneTool.__init__(
        tool,
        description=description,
        parameters=parameters,
        _execute_config=execute_config,
        _api_key=api_key,
    )
    return tool
