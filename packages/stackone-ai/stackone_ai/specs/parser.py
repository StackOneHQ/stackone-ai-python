import json
from pathlib import Path

from stackone_ai.models import ExecuteConfig, ToolDefinition, ToolParameters


class OpenAPIParser:
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path
        with open(spec_path) as f:
            self.spec = json.load(f)
        # Get base URL from servers array or default to stackone API
        servers = self.spec.get("servers", [{"url": "https://api.stackone_ai.com"}])
        self.base_url = (
            servers[0]["url"] if isinstance(servers, list) else "https://api.stackone.com"
        )

    def parse_tools(self) -> dict[str, ToolDefinition]:
        """Parse OpenAPI spec into tool definitions"""
        tools = {}

        for path, path_item in self.spec.get("paths", {}).items():
            for method, operation in path_item.items():
                # Get the tool name from x-speakeasy-name-override or generate from path
                name = operation.get("x-speakeasy-name-override") or path.strip("/").replace(
                    "/", "_"
                )

                # Create tool definition
                tools[name] = ToolDefinition(
                    description=operation.get("description", ""),
                    parameters=self._parse_parameters(operation),
                    execute=ExecuteConfig(
                        headers={},
                        method=method.upper(),
                        url=f"{self.base_url}{path}",
                        name=name,
                    ),
                )

        return tools

    def _parse_parameters(self, operation: dict) -> ToolParameters:
        """Parse OpenAPI parameters into ToolParameters"""
        properties = {}

        for param in operation.get("parameters", []):
            if param["in"] == "path":
                properties[param["name"]] = {
                    "type": param["schema"]["type"],
                    "description": param.get("description", ""),
                }

        return ToolParameters(type="object", properties=properties)
