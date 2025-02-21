import base64
import json
from collections.abc import Callable, Sequence
from typing import Annotated, Any, ClassVar

import requests
from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field, PrivateAttr


class ExecuteConfig(BaseModel):
    headers: dict = Field(default_factory=dict)  # Keep this with default empty dict
    method: str
    url: str
    name: str
    body_type: str | None = None
    parameter_locations: dict[str, str] = Field(
        default_factory=dict
    )  # Maps param name to location (header, query, path)


class ToolParameters(BaseModel):
    type: str
    properties: dict


class ToolDefinition(BaseModel):
    description: str
    parameters: ToolParameters
    execute: ExecuteConfig


class BaseTool(BaseModel):
    """Base Tool model with Pydantic validation"""

    description: str
    parameters: ToolParameters

    # Private attributes in Pydantic v2
    _execute_config: ExecuteConfig = PrivateAttr()
    _api_key: str = PrivateAttr()
    _account_id: str | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        execute_config = data.get("_execute_config")
        api_key = data.get("_api_key")

        if not isinstance(execute_config, ExecuteConfig):
            raise ValueError("_execute_config must be an ExecuteConfig instance")
        if not isinstance(api_key, str):
            raise ValueError("_api_key must be a string")

        self._execute_config = execute_config
        self._api_key = api_key
        self._account_id = data.get("_account_id")

    def execute(self, arguments: str | dict | None = None) -> dict[str, Any]:
        """
        Execute the tool with the given parameters

        Args:
            arguments: Either a JSON string or dict of arguments
        """
        # Handle both string and dict arguments
        if isinstance(arguments, str):
            kwargs = json.loads(arguments)
        else:
            kwargs = arguments or {}

        url = self._execute_config.url
        body_params = {}
        query_params = {}
        header_params = {}

        # Separate parameters based on their OpenAPI location
        if kwargs:
            for key, value in kwargs.items():
                param_location = self._execute_config.parameter_locations.get(key)

                if param_location == "header":
                    header_params[key] = str(value)
                elif param_location == "query":
                    query_params[key] = value
                elif param_location == "path":
                    url = url.replace(f"{{{key}}}", str(value))
                elif param_location == "body":
                    body_params[key] = value
                else:
                    # Default behavior for backward compatibility
                    if f"{{{key}}}" in url:
                        url = url.replace(f"{{{key}}}", str(value))
                    elif self._execute_config.method.upper() in ["GET", "DELETE"]:
                        query_params[key] = value
                    else:
                        body_params[key] = value

        # Create basic auth header with API key as username
        auth_string = base64.b64encode(f"{self._api_key}:".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_string}",
            "User-Agent": "stackone-python/1.0.0",
        }

        if self._account_id:
            headers["x-account-id"] = self._account_id

        # Add custom header parameters
        headers.update(header_params)

        # Add predefined headers last to ensure they take precedence
        headers.update(self._execute_config.headers)

        request_kwargs: dict[str, Any] = {
            "method": self._execute_config.method,
            "url": url,
            "headers": headers,
        }

        # Handle request body if we have body parameters
        if body_params:
            body_type = self._execute_config.body_type or "json"
            if body_type == "json":
                request_kwargs["json"] = body_params
            elif body_type == "form":
                request_kwargs["data"] = body_params

        if query_params:
            request_kwargs["params"] = query_params

        response = requests.request(**request_kwargs)
        response.raise_for_status()

        result: dict[str, Any] = response.json()
        return result

    def to_openai_function(self) -> dict:
        """Convert this tool to OpenAI's function format"""
        return {
            "type": "function",
            "function": {
                "name": self._execute_config.name,
                "description": self.description,
                "parameters": {
                    "type": self.parameters.type,
                    "properties": self.parameters.properties,
                    "required": list(self.parameters.properties.keys()),
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @property
    def name(self) -> str:
        """Get the tool's name"""
        return self._execute_config.name

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for this tool.

        Args:
            account_id: The account ID to use, or None to clear it
        """
        self._account_id = account_id

    def get_account_id(self) -> str | None:
        """Get the current account ID for this tool."""
        return self._account_id


class Tools:
    """Container for Tool instances"""

    def __init__(self, tools: list[BaseTool]):
        self.tools = tools
        # Create a name -> tool mapping for faster lookups
        self._tool_map = {tool.name: tool for tool in tools}

    def __getitem__(self, index: int) -> BaseTool:
        return self.tools[index]

    def __len__(self) -> int:
        return len(self.tools)

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by its name"""
        return self._tool_map.get(name)

    def to_openai(self) -> list[dict]:
        """Convert all tools to OpenAI function format"""
        return [tool.to_openai_function() for tool in self.tools]

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for all tools in this collection.

        Args:
            account_id: The account ID to use, or None to clear it
        """
        for tool in self.tools:
            tool.set_account_id(account_id)

    def to_langchain(self) -> Sequence[LangChainBaseTool]:
        """Convert all tools to LangChain tool format"""
        langchain_tools = []

        for tool in self.tools:
            # Create properly annotated schema for the tool
            schema_props: dict[str, Field] = {}
            annotations: dict[str, Any] = {}

            for name, details in tool.parameters.properties.items():
                # Convert OpenAPI types to Python types
                python_type: type = str  # Default to str
                if isinstance(details, dict):  # Check if details is a dict
                    type_str = details.get("type", "string")
                    if type_str == "number":
                        python_type = float
                    elif type_str == "integer":
                        python_type = int
                    elif type_str == "boolean":
                        python_type = bool

                    # Create Field with description if available
                    field = Field(description=details.get("description", ""))
                else:
                    # Handle case where details is a string
                    field = Field(description="")

                schema_props[name] = field
                annotations[name] = Annotated[python_type, field]

            # Create the schema class with proper annotations
            schema_class = type(
                f"{tool.name.title()}Args",
                (BaseModel,),
                {
                    "__annotations__": annotations,
                    "__module__": __name__,
                    **schema_props,
                },
            )

            # Create the LangChain tool with proper type annotations
            tool_annotations = {
                "name": ClassVar[str],
                "description": ClassVar[str],
                "args_schema": ClassVar[type],
            }

            def create_run_method(t: BaseTool) -> Callable[..., Any]:
                def _run(self: Any, **kwargs: Any) -> Any:
                    # Convert kwargs to dict for execution
                    return t.execute(kwargs)

                return _run

            def create_arun_method(t: BaseTool) -> Callable[..., Any]:
                async def _arun(self: Any, **kwargs: Any) -> Any:
                    # Convert kwargs to dict for execution
                    return t.execute(kwargs)

                return _arun

            langchain_tool = type(
                f"StackOne{tool.name.title()}Tool",
                (LangChainBaseTool,),
                {
                    "__annotations__": tool_annotations,
                    "__module__": __name__,
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": schema_class,
                    "_run": create_run_method(tool),
                    "_arun": create_arun_method(tool),
                },
            )
            langchain_tools.append(langchain_tool())

        return langchain_tools
