import json
import os
from typing import Any

from stackone_ai.constants import OAS_DIR
from stackone_ai.models import BaseTool, ToolDefinition, Tools


class StackOneToolSet:
    """Main class for accessing StackOne tools"""

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
    ) -> None:
        """Initialize StackOne tools with authentication.

        Args:
            api_key: Optional API key. If not provided, will try to get from STACKONE_API_KEY env var
            account_id: Optional account ID. If not provided, will try to get from STACKONE_ACCOUNT_ID env var
        """
        self.api_key = api_key or os.getenv("STACKONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either through api_key parameter or "
                "STACKONE_API_KEY environment variable"
            )

        self.account_id = account_id or os.getenv("STACKONE_ACCOUNT_ID")

    def get_tools(self, vertical: str, account_id: str | None = None) -> Tools:
        """Get tools for a specific vertical.

        Args:
            vertical: The vertical to get tools for (e.g. "hris", "crm")
            account_id: Optional account ID override. If not provided, uses the one from initialization
        """
        spec_path = OAS_DIR / f"{vertical}.json"
        if not spec_path.exists():
            return Tools([])  # Return empty tools list for unknown vertical

        # Use account_id parameter if provided, otherwise use the one from initialization
        effective_account_id = account_id or self.account_id

        with open(spec_path) as f:
            spec = json.load(f)

        tools: list[BaseTool] = []
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            for method, details in methods.items():
                # Skip if no x-speakeasy-name-override (indicates not a tool endpoint)
                if "x-speakeasy-name-override" not in details:
                    continue

                name = details["x-speakeasy-name-override"]
                description = details.get("description", "")
                parameters = details.get("parameters", [])

                # Convert OpenAPI parameters to JSON Schema
                properties: dict[str, Any] = {}
                for param in parameters:
                    if param["in"] == "path":
                        properties[param["name"]] = {
                            "type": param["schema"]["type"],
                            "description": param.get("description", ""),
                        }

                tool_def = ToolDefinition(
                    description=description,
                    parameters={"type": "object", "properties": properties},
                    execute={
                        "headers": {},
                        "method": method.upper(),
                        "url": f"https://api.stackone.com{path}",
                        "name": name,
                    },
                )

                tool = BaseTool(
                    description=tool_def.description,
                    parameters=tool_def.parameters,
                    _execute_config=tool_def.execute,
                    _api_key=self.api_key,
                    _account_id=effective_account_id,
                )
                tools.append(tool)

        return Tools(tools)
