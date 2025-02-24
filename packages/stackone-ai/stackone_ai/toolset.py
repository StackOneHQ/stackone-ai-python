import os
from typing import Any

from stackone_ai.constants import OAS_DIR
from stackone_ai.models import (
    StackOneTool,
    Tools,
)
from stackone_ai.specs.parser import OpenAPIParser


class ToolsetError(Exception):
    """Base exception for toolset errors"""

    pass


class ToolsetConfigError(ToolsetError):
    """Raised when there is an error in the toolset configuration"""

    pass


class ToolsetLoadError(ToolsetError):
    """Raised when there is an error loading tools"""

    pass


class StackOneToolSet:
    """Main class for accessing StackOne tools"""

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
    ) -> None:
        """Initialize StackOne tools with authentication

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID. If not provided, will try to get from STACKONE_ACCOUNT_ID env var

        Raises:
            ToolsetConfigError: If no API key is provided or found in environment
        """
        api_key_value = api_key or os.getenv("STACKONE_API_KEY")
        if not api_key_value:
            raise ToolsetConfigError(
                "API key must be provided either through api_key parameter or "
                "STACKONE_API_KEY environment variable"
            )
        self.api_key: str = api_key_value
        self.account_id = account_id or os.getenv("STACKONE_ACCOUNT_ID")

    def _parse_parameters(self, parameters: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        """Parse OpenAPI parameters into tool properties

        Args:
            parameters: List of OpenAPI parameter objects

        Returns:
            Dict of parameter properties with name as key and schema details as value
        """
        properties: dict[str, dict[str, str]] = {}
        for param in parameters:
            if param["in"] == "path":
                # Ensure we only include string values in the nested dict
                param_schema = param["schema"]
                properties[param["name"]] = {
                    "type": str(param_schema["type"]),
                    "description": str(param.get("description", "")),
                }
        return properties

    def get_tools(self, vertical: str, account_id: str | None = None) -> Tools:
        """Get tools for a specific vertical

        Args:
            vertical: The vertical to get tools for (e.g. "hris", "crm")
            account_id: Optional account ID override. If not provided, uses the one from initialization

        Returns:
            Collection of tools for the vertical

        Raises:
            ToolsetLoadError: If there is an error loading the tools
        """
        try:
            spec_path = OAS_DIR / f"{vertical}.json"
            if not spec_path.exists():
                raise ToolsetLoadError(f"No spec file found for vertical: {vertical}")

            parser = OpenAPIParser(spec_path)
            tool_definitions = parser.parse_tools()
            effective_account_id = account_id or self.account_id

            tools: list[StackOneTool] = []
            for _, tool_def in tool_definitions.items():
                tool = StackOneTool(
                    description=tool_def.description,
                    parameters=tool_def.parameters,
                    _execute_config=tool_def.execute,
                    _api_key=self.api_key,
                    _account_id=effective_account_id,
                )
                tools.append(tool)

            return Tools(tools)

        except Exception as e:
            if isinstance(e, ToolsetError):
                raise
            raise ToolsetLoadError(f"Error loading tools for vertical {vertical}: {e}") from e
