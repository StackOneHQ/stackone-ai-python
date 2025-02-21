from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


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


class Tool(BaseModel):
    """Base Tool model"""

    name: str
    description: str
    parameters: ToolParameters

    def execute(self, arguments: str | dict | None = None) -> dict[str, Any]:
        """Execute the tool with the given parameters"""
        raise NotImplementedError

    def to_openai_function(self) -> dict:
        """Convert this tool to OpenAI's function format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
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


class Tools:
    """Container for Tool instances"""

    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self._tool_map = {tool.name: tool for tool in tools}

    def __getitem__(self, index: int) -> Tool:
        return self.tools[index]

    def __len__(self) -> int:
        return len(self.tools)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by its name"""
        return self._tool_map.get(name)

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for all tools in this collection.

        Args:
            account_id: The account ID to use, or None to clear it
        """
        for tool in self.tools:
            tool.set_account_id(account_id)

    def get_account_id(self) -> str | None:
        """Get the current account ID for this tool."""
        for tool in self.tools:
            account_id = tool.get_account_id()
            if isinstance(account_id, str):  # Type guard to ensure we return str | None
                return account_id
        return None

    def to_openai(self) -> list[dict]:
        """Convert all tools to OpenAI function format"""
        return [tool.to_openai_function() for tool in self.tools]

    def to_langchain(self) -> Sequence[BaseTool]:
        """Convert all tools to LangChain format"""
        return [tool.to_langchain() for tool in self.tools]
