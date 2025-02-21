import base64
import json
from typing import Annotated, Any

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from stackone_ai.models import (
    ExecuteConfig,
    ToolParameters,
)
from stackone_ai.models import (
    Tool as StackOneBaseTool,
)


class StackOneTool(StackOneBaseTool):
    """Concrete implementation of StackOne Tool"""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: ToolParameters = Field(description="Tool parameters")
    _execute_config: ExecuteConfig = PrivateAttr()
    _api_key: str = PrivateAttr()
    _account_id: str | None = PrivateAttr(default=None)

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

    def execute(self, arguments: str | dict | None = None) -> dict[str, Any]:
        """Execute the tool with the given parameters"""
        # Handle both string and dict arguments
        if isinstance(arguments, str):
            kwargs = json.loads(arguments)
        else:
            kwargs = arguments or {}

        # Create basic auth header with API key as username
        auth_string = base64.b64encode(f"{self._api_key}:".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_string}",
            "User-Agent": "stackone-python/1.0.0",
        }

        if self._account_id:
            headers["x-account-id"] = self._account_id

        # Add predefined headers
        headers.update(self._execute_config.headers)

        url = self._execute_config.url
        body_params = {}
        query_params = {}

        # Handle parameters based on their location
        for key, value in kwargs.items():
            param_location = self._execute_config.parameter_locations.get(key)

            if param_location == "path":
                url = url.replace(f"{{{key}}}", str(value))
            elif param_location == "query":
                query_params[key] = value
            elif param_location == "body":
                body_params[key] = value
            else:
                # Default behavior
                if f"{{{key}}}" in url:
                    url = url.replace(f"{{{key}}}", str(value))
                elif self._execute_config.method.upper() in ["GET", "DELETE"]:
                    query_params[key] = value
                else:
                    body_params[key] = value

        request_kwargs: dict[str, Any] = {
            "method": self._execute_config.method,
            "url": url,
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

        response = requests.request(**request_kwargs)
        response.raise_for_status()

        # Ensure we return a dict
        result = response.json()
        if not isinstance(result, dict):
            return {"result": result}
        return result

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

    def to_langchain(self) -> BaseTool:
        """Convert this tool to LangChain format"""
        tool_self = self  # Capture self reference for inner class

        # Create properly annotated schema for the tool
        schema_props: dict[str, Field] = {}
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
            annotations[name] = Annotated[python_type, field]

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

        class StackOneLangChainTool(BaseTool):
            name: str = tool_self.name
            description: str = tool_self.description
            args_schema: type[BaseModel] = schema_class
            return_direct: bool = True
            func = staticmethod(tool_self.execute)

            def _run(self, **kwargs: Any) -> Any:
                return tool_self.execute(kwargs)

            async def _arun(self, **kwargs: Any) -> Any:
                return self._run(**kwargs)

        return StackOneLangChainTool()

    def set_account_id(self, account_id: str | None) -> None:
        """Set the account ID for this tool.

        Args:
            account_id: The account ID to use, or None to clear it
        """
        self._account_id = account_id

    def get_account_id(self) -> str | None:
        """Get the current account ID for this tool."""
        return self._account_id
