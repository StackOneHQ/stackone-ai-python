from __future__ import annotations

import base64
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, ClassVar, TypeAlias, cast
from urllib.parse import quote

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, BeforeValidator, Field, PrivateAttr

# Type aliases for common types
JsonDict: TypeAlias = dict[str, Any]
Headers: TypeAlias = dict[str, str]


logger = logging.getLogger("stackone.tools")


class StackOneError(Exception):
    """Base exception for StackOne errors"""

    pass


class StackOneAPIError(StackOneError):
    """Raised when the StackOne API returns an error"""

    def __init__(self, message: str, status_code: int, response_body: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


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


class ToolParameters(BaseModel):
    """Schema definition for tool parameters"""

    type: str = Field(description="JSON Schema type")
    properties: JsonDict = Field(description="JSON Schema properties")


class ToolDefinition(BaseModel):
    """Complete definition of a tool including its schema and execution config"""

    description: str = Field(description="Tool description")
    parameters: ToolParameters = Field(description="Tool parameter schema")
    execute: ExecuteConfig = Field(description="Tool execution configuration")


class StackOneTool(BaseModel):
    """Base class for all StackOne tools. Provides functionality for executing API calls
    and converting to various formats (OpenAI, LangChain)."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: ToolParameters = Field(description="Tool parameters")
    _execute_config: ExecuteConfig = PrivateAttr()
    _api_key: str = PrivateAttr()
    _account_id: str | None = PrivateAttr(default=None)
    _FEEDBACK_OPTION_KEYS: ClassVar[set[str]] = {
        "feedback_session_id",
        "feedback_user_id",
        "feedback_metadata",
    }

    def __init__(
        self,
        description: str,
        parameters: ToolParameters,
        _execute_config: ExecuteConfig,
        _api_key: str,
        _account_id: str | None = None,
    ) -> None:
        super().__init__(
            name=_execute_config.name,
            description=description,
            parameters=parameters,
        )
        self._execute_config = _execute_config
        self._api_key = _api_key
        self._account_id = _account_id

    @classmethod
    def _split_feedback_options(cls, params: JsonDict, options: JsonDict | None) -> tuple[JsonDict, JsonDict]:
        merged_params = dict(params)
        feedback_options = dict(options or {})
        for key in cls._FEEDBACK_OPTION_KEYS:
            if key in merged_params and key not in feedback_options:
                feedback_options[key] = merged_params.pop(key)
        return merged_params, feedback_options

    def _prepare_headers(self) -> Headers:
        """Prepare headers for the API request

        Returns:
            Headers to use in the request
        """
        auth_string = base64.b64encode(f"{self._api_key}:".encode()).decode()
        headers: Headers = {
            "Authorization": f"Basic {auth_string}",
            "User-Agent": "stackone-python/1.0.0",
        }

        if self._account_id:
            headers["x-account-id"] = self._account_id

        # Add predefined headers
        headers.update(self._execute_config.headers)
        return headers

    def _prepare_request_params(self, kwargs: JsonDict) -> tuple[str, JsonDict, JsonDict]:
        """Prepare URL and parameters for the API request

        Args:
            kwargs: Arguments to process

        Returns:
            Tuple of (url, body_params, query_params)
        """
        url = self._execute_config.url
        body_params: JsonDict = {}
        query_params: JsonDict = {}

        for key, value in kwargs.items():
            param_location = self._execute_config.parameter_locations.get(key)

            if param_location == ParameterLocation.PATH:
                # Safely encode path parameters to prevent SSRF attacks
                encoded_value = quote(str(value), safe="")
                url = url.replace(f"{{{key}}}", encoded_value)
            elif param_location == ParameterLocation.QUERY:
                query_params[key] = value
            elif param_location in (ParameterLocation.BODY, ParameterLocation.FILE):
                body_params[key] = value
            else:
                # Default behavior
                if f"{{{key}}}" in url:
                    # Safely encode path parameters to prevent SSRF attacks
                    encoded_value = quote(str(value), safe="")
                    url = url.replace(f"{{{key}}}", encoded_value)
                elif self._execute_config.method in {"GET", "DELETE"}:
                    query_params[key] = value
                else:
                    body_params[key] = value

        return url, body_params, query_params

    def execute(
        self, arguments: str | JsonDict | None = None, *, options: JsonDict | None = None
    ) -> JsonDict:
        """Execute the tool with the given parameters

        Args:
            arguments: Tool arguments as string or dict
            options: Execution options (e.g. feedback metadata)

        Returns:
            API response as dict

        Raises:
            StackOneAPIError: If the API request fails
            ValueError: If the arguments are invalid
        """
        datetime.now(UTC)
        feedback_options: JsonDict = {}
        result_payload: JsonDict | None = None
        response_status: int | None = None
        error_message: str | None = None
        status = "success"
        url_used = self._execute_config.url

        try:
            if isinstance(arguments, str):
                parsed_arguments = json.loads(arguments)
            else:
                parsed_arguments = arguments or {}

            if not isinstance(parsed_arguments, dict):
                status = "error"
                error_message = "Tool arguments must be a JSON object"
                raise ValueError(error_message)

            kwargs = parsed_arguments
            dict(kwargs)

            headers = self._prepare_headers()
            url_used, body_params, query_params = self._prepare_request_params(kwargs)

            request_kwargs: dict[str, Any] = {
                "method": self._execute_config.method,
                "url": url_used,
                "headers": headers,
            }

            if body_params:
                body_type = self._execute_config.body_type or "json"
                if body_type == "json":
                    request_kwargs["json"] = body_params
                elif body_type == "form":
                    request_kwargs["data"] = body_params

            if query_params:
                request_kwargs["params"] = query_params

            response = httpx.request(**request_kwargs)
            response_status = response.status_code
            response.raise_for_status()

            result = response.json()
            result_payload = cast(JsonDict, result) if isinstance(result, dict) else {"result": result}
            return result_payload

        except json.JSONDecodeError as exc:
            status = "error"
            error_message = f"Invalid JSON in arguments: {exc}"
            raise ValueError(error_message) from exc
        except httpx.HTTPStatusError as exc:
            status = "error"
            response_body = None
            if exc.response.text:
                try:
                    response_body = exc.response.json()
                except json.JSONDecodeError:
                    response_body = exc.response.text
            raise StackOneAPIError(
                str(exc),
                exc.response.status_code,
                response_body,
            ) from exc
        except httpx.RequestError as exc:
            status = "error"
            raise StackOneError(f"Request failed: {exc}") from exc
        finally:
            datetime.now(UTC)
            metadata: JsonDict = {
                "http_method": self._execute_config.method,
                "url": url_used,
                "status_code": response_status,
                "status": status,
            }

            feedback_metadata = feedback_options.get("feedback_metadata")
            if isinstance(feedback_metadata, dict):
                metadata["feedback_metadata"] = feedback_metadata

            if feedback_options:
                metadata["feedback_options"] = {
                    key: value
                    for key, value in feedback_options.items()
                    if key in {"feedback_session_id", "feedback_user_id"} and value is not None
                }

            # Implicit feedback removed - just API calls

    def call(self, *args: Any, options: JsonDict | None = None, **kwargs: Any) -> JsonDict:
        """Call the tool with the given arguments

        This method provides a more intuitive way to execute tools directly.

        Args:
            *args: If a single argument is provided, it's treated as the full arguments dict/string
            **kwargs: Keyword arguments to pass to the tool
            options: Optional execution options

        Returns:
            API response as dict

        Raises:
            StackOneAPIError: If the API request fails
            ValueError: If the arguments are invalid

        Examples:
            >>> tool.call({"name": "John", "email": "john@example.com"})
            >>> tool.call(name="John", email="john@example.com")
        """
        if args and kwargs:
            raise ValueError("Cannot provide both positional and keyword arguments")

        if args:
            if len(args) > 1:
                raise ValueError("Only one positional argument is allowed")
            return self.execute(args[0])

        return self.execute(kwargs if kwargs else None)

    def to_openai_function(self) -> JsonDict:
        """Convert this tool to OpenAI's function format

        Returns:
            Tool definition in OpenAI function format
        """
        # Clean properties and handle special types
        properties = {}
        required = []

        for name, prop in self.parameters.properties.items():
            if isinstance(prop, dict):
                # Only keep standard JSON Schema properties
                cleaned_prop = {}

                # Copy basic properties
                if "type" in prop:
                    cleaned_prop["type"] = prop["type"]
                if "description" in prop:
                    cleaned_prop["description"] = prop["description"]
                if "enum" in prop:
                    cleaned_prop["enum"] = prop["enum"]

                # Handle array types
                if cleaned_prop.get("type") == "array" and "items" in prop:
                    if isinstance(prop["items"], dict):
                        cleaned_prop["items"] = {
                            k: v for k, v in prop["items"].items() if k in ("type", "description", "enum")
                        }

                # Handle object types
                if cleaned_prop.get("type") == "object" and "properties" in prop:
                    cleaned_prop["properties"] = {
                        k: {sk: sv for sk, sv in v.items() if sk in ("type", "description", "enum")}
                        for k, v in prop["properties"].items()
                    }

                # Handle required fields - if not explicitly nullable
                if not prop.get("nullable", False):
                    required.append(name)

                properties[name] = cleaned_prop
            else:
                properties[name] = {"type": "string"}
                required.append(name)

        # Create the OpenAI function schema
        parameters = {
            "type": "object",
            "properties": properties,
        }

        # Only include required if there are required fields
        if required:
            parameters["required"] = required

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def to_langchain(self) -> BaseTool:
        """Convert this tool to LangChain format

        Returns:
            Tool in LangChain format
        """
        # Create properly annotated schema for the tool
        schema_props: dict[str, Any] = {}
        annotations: dict[str, Any] = {}

        for name, details in self.parameters.properties.items():
            python_type: type = str  # Default to str
            if isinstance(details, dict):
                type_str = details.get("type", "string")
                if type_str == "number":
                    python_type = float
                elif type_str == "integer":
                    python_type = int
                elif type_str == "boolean":
                    python_type = bool

                field = Field(description=details.get("description", ""))
            else:
                field = Field(description="")

            schema_props[name] = field
            annotations[name] = python_type

        # Create the schema class with proper annotations
        schema_class = type(
            f"{self.name.title()}Args",
            (BaseModel,),
            {
                "__annotations__": annotations,
                "__module__": __name__,
                **schema_props,
            },
        )

        parent_tool = self

        class StackOneLangChainTool(BaseTool):
            name: str = parent_tool.name
            description: str = parent_tool.description
            args_schema: type[BaseModel] = schema_class  # ty: ignore[invalid-assignment]
            func = staticmethod(parent_tool.execute)  # Required by CrewAI

            def _run(self, **kwargs: Any) -> Any:
                return parent_tool.execute(kwargs)

        return StackOneLangChainTool()

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for this tool

        Args:
            account_id: The account ID to use, or None to clear it
        """
        self._account_id = account_id

    def get_account_id(self) -> str | None:
        """Get the current account ID for this tool

        Returns:
            Current account ID or None if not set
        """
        return self._account_id


class Tools:
    """Container for Tool instances with lookup capabilities"""

    def __init__(self, tools: list[StackOneTool]) -> None:
        """Initialize Tools container

        Args:
            tools: List of Tool instances to manage
        """
        self.tools = tools
        self._tool_map = {tool.name: tool for tool in tools}

    def __getitem__(self, index: int) -> StackOneTool:
        return self.tools[index]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Any:
        """Make Tools iterable"""
        return iter(self.tools)

    def to_list(self) -> list[StackOneTool]:
        """Convert to list of tools

        Returns:
            List of StackOneTool instances
        """
        return list(self.tools)

    def get_tool(self, name: str) -> StackOneTool | None:
        """Get a tool by its name

        Args:
            name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        return self._tool_map.get(name)

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for all tools in this collection

        Args:
            account_id: The account ID to use, or None to clear it
        """
        for tool in self.tools:
            tool.set_account_id(account_id)

    def get_account_id(self) -> str | None:
        """Get the current account ID for this collection

        Returns:
            The first non-None account ID found, or None if none set
        """
        for tool in self.tools:
            account_id = tool.get_account_id()
            if isinstance(account_id, str):
                return account_id
        return None

    def to_openai(self) -> list[JsonDict]:
        """Convert all tools to OpenAI function format

        Returns:
            List of tools in OpenAI function format
        """
        return [tool.to_openai_function() for tool in self.tools]

    def to_langchain(self) -> Sequence[BaseTool]:
        """Convert all tools to LangChain format

        Returns:
            Sequence of tools in LangChain format
        """
        return [tool.to_langchain() for tool in self.tools]

    def meta_tools(self, hybrid_alpha: float | None = None) -> Tools:
        """Return meta tools for tool discovery and execution

        Meta tools enable dynamic tool discovery and execution based on natural language queries
        using hybrid BM25 + TF-IDF search.

        Args:
            hybrid_alpha: Weight for BM25 in hybrid search (0-1). If not provided, uses
                ToolIndex.DEFAULT_HYBRID_ALPHA (0.2), which gives more weight to BM25 scoring
                and has been shown to provide better tool discovery accuracy
                (10.8% improvement in validation testing).

        Returns:
            Tools collection containing meta_search_tools and meta_execute_tool

        Note:
            This feature is in beta and may change in future versions
        """
        from stackone_ai.meta_tools import (
            ToolIndex,
            create_meta_execute_tool,
            create_meta_search_tools,
        )

        # Create search index with hybrid search
        index = ToolIndex(self.tools, hybrid_alpha=hybrid_alpha)

        # Create meta tools
        filter_tool = create_meta_search_tools(index)
        execute_tool = create_meta_execute_tool(self)

        return Tools([filter_tool, execute_tool])
