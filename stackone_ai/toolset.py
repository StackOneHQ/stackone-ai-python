import fnmatch
import os
import warnings
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
        base_url: str | None = None,
    ) -> None:
        """Initialize StackOne tools with authentication

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID. If not provided, will try to get from STACKONE_ACCOUNT_ID env var
            base_url: Optional base URL override for API requests. If not provided, uses the URL from the OAS

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
        self.base_url = base_url

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

    def _matches_filter(self, tool_name: str, filter_pattern: str | list[str]) -> bool:
        """Check if a tool name matches the filter pattern

        Args:
            tool_name: Name of the tool to check
            filter_pattern: String or list of glob patterns to match against.
                          Patterns starting with ! are treated as negative matches.

        Returns:
            True if the tool name matches any positive pattern and no negative patterns,
            False otherwise
        """
        patterns = [filter_pattern] if isinstance(filter_pattern, str) else filter_pattern

        # Split into positive and negative patterns
        positive_patterns = [p for p in patterns if not p.startswith("!")]
        negative_patterns = [p[1:] for p in patterns if p.startswith("!")]

        # If no positive patterns, treat as match all
        matches_positive = (
            any(fnmatch.fnmatch(tool_name, p) for p in positive_patterns) if positive_patterns else True
        )

        # If any negative pattern matches, exclude the tool
        matches_negative = any(fnmatch.fnmatch(tool_name, p) for p in negative_patterns)

        return matches_positive and not matches_negative

    def get_tool(self, name: str, *, account_id: str | None = None) -> StackOneTool | None:
        """Get a specific tool by name

        Args:
            name: Name of the tool to retrieve
            account_id: Optional account ID override. If not provided, uses the one from initialization

        Returns:
            The tool if found, None otherwise

        Raises:
            ToolsetLoadError: If there is an error loading the tools
        """
        tools = self.get_tools(name, account_id=account_id)
        return tools.get_tool(name)

    def get_tools(
        self, filter_pattern: str | list[str] | None = None, *, account_id: str | None = None
    ) -> Tools:
        """Get tools matching the specified filter pattern

        Args:
            filter_pattern: Optional glob pattern or list of patterns to filter tools
                (e.g. "hris_*", ["crm_*", "ats_*"])
            account_id: Optional account ID override. If not provided, uses the one from initialization

        Returns:
            Collection of tools matching the filter pattern

        Raises:
            ToolsetLoadError: If there is an error loading the tools
        """
        if filter_pattern is None:
            warnings.warn(
                "No filter pattern provided. Loading all tools may exceed context windows in "
                "AI applications.",
                UserWarning,
                stacklevel=2,
            )

        try:
            all_tools: list[StackOneTool] = []
            effective_account_id = account_id or self.account_id

            # Load all available specs
            for spec_file in OAS_DIR.glob("*.json"):
                parser = OpenAPIParser(spec_file, base_url=self.base_url)
                tool_definitions = parser.parse_tools()

                # Create tools and filter if pattern is provided
                for _, tool_def in tool_definitions.items():
                    if filter_pattern is None or self._matches_filter(tool_def.execute.name, filter_pattern):
                        tool = StackOneTool(
                            description=tool_def.description,
                            parameters=tool_def.parameters,
                            _execute_config=tool_def.execute,
                            _api_key=self.api_key,
                            _account_id=effective_account_id,
                        )
                        all_tools.append(tool)

            return Tools(all_tools)

        except Exception as e:
            if isinstance(e, ToolsetError):
                raise
            raise ToolsetLoadError(f"Error loading tools: {e}") from e
