import json
from pathlib import Path
from typing import Any

from stackone_ai.models import ExecuteConfig, ToolDefinition, ToolParameters


class OpenAPIParser:
    def __init__(self, spec_path: Path, base_url: str | None = None):
        self.spec_path = spec_path
        with open(spec_path) as f:
            self.spec = json.load(f)
        # Get base URL from servers array or default to stackone API
        servers = self.spec.get("servers", [{"url": "https://api.stackone.com"}])
        default_url = servers[0]["url"] if isinstance(servers, list) else "https://api.stackone.com"
        # Use provided base_url if available, otherwise use the default from the spec
        self.base_url = base_url or default_url

    def _is_file_type(self, schema: dict[str, Any]) -> bool:
        """Check if a schema represents a file upload."""
        return schema.get("type") == "string" and schema.get("format") == "binary"

    def _convert_to_file_type(self, schema: dict[str, Any]) -> None:
        """Convert a binary string schema to a file type."""
        if self._is_file_type(schema):
            schema["type"] = "file"

    def _handle_file_properties(self, schema: dict[str, Any]) -> None:
        """Process schema properties to handle file uploads."""
        if "properties" not in schema:
            return

        for prop_schema in schema["properties"].values():
            # Handle direct file uploads
            self._convert_to_file_type(prop_schema)

            # Handle array of files
            if prop_schema.get("type") == "array" and "items" in prop_schema:
                self._convert_to_file_type(prop_schema["items"])

    def _resolve_schema_ref(
        self, ref: str, visited: set[str] | None = None
    ) -> dict[str, Any] | list[Any] | str:
        """
        Resolve a JSON schema reference in the OpenAPI spec
        """
        if not ref.startswith("#/"):
            raise ValueError(f"Only local references are supported: {ref}")

        if visited is None:
            visited = set()

        if ref in visited:
            raise ValueError(f"Circular reference detected: {ref}")

        visited.add(ref)

        parts = ref.split("/")[1:]  # Skip the '#'
        current = self.spec
        for part in parts:
            current = current[part]

        # After getting the referenced schema, resolve it fully
        return self._resolve_schema(current, visited)

    def _resolve_schema(
        self, schema: dict[str, Any] | list[Any] | str, visited: set[str] | None = None
    ) -> dict[str, Any] | list[Any] | str:
        """
        Resolve all references in a schema, preserving structure
        """
        if visited is None:
            visited = set()

        # Handle primitive types (str, int, etc)
        if not isinstance(schema, dict | list):
            return schema

        if isinstance(schema, list):
            return [self._resolve_schema(item, visited.copy()) for item in schema]

        # Now we know schema is a dict
        # Handle direct reference
        if "$ref" in schema:
            resolved = self._resolve_schema_ref(schema["$ref"], visited)
            if not isinstance(resolved, dict):
                return resolved
            # Merge any additional properties from the original schema
            return {**resolved, **{k: v for k, v in schema.items() if k != "$ref"}}

        # Handle allOf combinations
        if "allOf" in schema:
            merged_schema = {k: v for k, v in schema.items() if k != "allOf"}

            # Merge all schemas in allOf array
            for sub_schema in schema["allOf"]:
                resolved = self._resolve_schema(sub_schema, visited.copy())
                if not isinstance(resolved, dict):
                    continue

                # Merge properties
                if "properties" in resolved:
                    if "properties" not in merged_schema:
                        merged_schema["properties"] = {}
                    merged_schema["properties"].update(resolved["properties"])

                # Merge type and other fields
                for key, value in resolved.items():
                    if key != "properties" and key not in merged_schema:
                        merged_schema[key] = value

            return merged_schema

        # Recursively resolve all nested dictionaries and arrays
        resolved = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_schema(value, visited.copy())
            elif isinstance(value, list):
                resolved[key] = [self._resolve_schema(item, visited.copy()) for item in value]
            else:
                resolved[key] = value

        return resolved

    def _parse_content_schema(
        self, content_type: str, content: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Parse schema from content object for a specific content type."""
        if content_type not in content:
            return None, None

        type_content = content[content_type]
        if not isinstance(type_content, dict):
            return None, None

        schema = type_content.get("schema", {})
        resolved = self._resolve_schema(schema)

        if not isinstance(resolved, dict):
            return None, None

        return resolved, content_type.split("/")[-1]

    def _parse_request_body(self, operation: dict) -> tuple[dict[str, Any] | None, str | None]:
        """Parse request body schema and content type from operation"""
        request_body = operation.get("requestBody", {})
        if not request_body:
            return None, None

        content = request_body.get("content", {})

        # Try JSON first
        schema, body_type = self._parse_content_schema("application/json", content)
        if schema:
            return schema, body_type

        # Try multipart form-data (file uploads)
        schema, _ = self._parse_content_schema("multipart/form-data", content)
        if schema:
            self._handle_file_properties(schema)
            return schema, "multipart"

        # Try form-urlencoded
        schema, body_type = self._parse_content_schema("application/x-www-form-urlencoded", content)
        if schema:
            return schema, "form"

        return None, None

    def _get_parameter_location(self, prop_schema: dict[str, Any]) -> str:
        """Determine the parameter location based on schema type."""
        if prop_schema.get("type") == "file":
            return "file"
        if prop_schema.get("type") == "array" and prop_schema.get("items", {}).get("type") == "file":
            return "file"
        return "body"

    def parse_tools(self) -> dict[str, ToolDefinition]:
        """Parse OpenAPI spec into tool definitions"""
        tools = {}

        for path, path_item in self.spec.get("paths", {}).items():
            for method, operation in path_item.items():
                name = operation.get("operationId")
                if not name:
                    raise ValueError(f"Operation ID is required for tool parsing: {operation}")

                # Parse request body if present
                request_body_schema, body_type = self._parse_request_body(operation)

                # Track parameter locations and properties
                parameter_locations = {}
                properties = {}

                # Parse parameters
                for param in operation.get("parameters", []):
                    param_name = param["name"]
                    param_location = param["in"]  # header, query, path, cookie
                    parameter_locations[param_name] = param_location

                    # Add to properties for tool parameters
                    schema = param.get("schema", {}).copy()
                    if "description" in param:
                        schema["description"] = param["description"]
                    properties[param_name] = self._resolve_schema(schema)

                # Add request body properties if present
                if request_body_schema and isinstance(request_body_schema, dict):
                    body_props = request_body_schema.get("properties", {})
                    for prop_name, prop_schema in body_props.items():
                        properties[prop_name] = prop_schema
                        parameter_locations[prop_name] = self._get_parameter_location(prop_schema)

                # Create tool definition
                tools[name] = ToolDefinition(
                    description=operation.get("summary", ""),
                    parameters=ToolParameters(type="object", properties=properties),
                    execute=ExecuteConfig(
                        method=method.upper(),
                        url=f"{self.base_url}{path}",
                        name=name,
                        parameter_locations=parameter_locations,
                        body_type=body_type,
                    ),
                )

        return tools
