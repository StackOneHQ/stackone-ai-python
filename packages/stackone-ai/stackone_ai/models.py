import base64
import json
from typing import Any

import requests
from pydantic import BaseModel, PrivateAttr


class ExecuteConfig(BaseModel):
    headers: dict
    method: str
    url: str
    name: str


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

    def execute(self, arguments: str | dict) -> dict[str, Any]:
        """
        Execute the tool with the given parameters

        Args:
            arguments: Either a JSON string or dict of arguments
        """
        # Handle both string and dict arguments
        if isinstance(arguments, str):
            kwargs = json.loads(arguments)
        else:
            kwargs = arguments

        url = self._execute_config.url

        # Replace URL parameters
        for key, value in kwargs.items():
            url = url.replace(f"{{{key}}}", str(value))

        # Create basic auth header with API key as username
        auth_string = base64.b64encode(f"{self._api_key}:".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_string}",
            "User-Agent": "stackone-python/1.0.0",
        }

        if self._account_id:
            headers["x-account-id"] = self._account_id

        headers.update(self._execute_config.headers)

        response = requests.request(method=self._execute_config.method, url=url, headers=headers)

        response.raise_for_status()

        # Explicitly type the return value
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
