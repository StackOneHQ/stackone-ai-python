import json
from pathlib import Path
from typing import Any

from stackone_ai.models import ExecuteConfig, ToolDefinition, ToolParameters


class OpenAPIParser:
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path
        with open(spec_path) as f:
            self.spec = json.load(f)
        # Get base URL from servers array or default to stackone API
        servers = self.spec.get("servers", [{"url": "https://api.stackone.com"}])
        self.base_url = servers[0]["url"] if isinstance(servers, list) else "https://api.stackone.com"

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

    def _parse_request_body(self, operation: dict) -> tuple[dict[str, Any] | None, str | None]:
        """Parse request body schema and content type from operation"""
        request_body = operation.get("requestBody", {})
        if not request_body:
            return None, None

        content = request_body.get("content", {})

        # Handle application/json
        if "application/json" in content:
            json_content = content["application/json"]
            if isinstance(json_content, dict):
                schema = json_content.get("schema", {})
                resolved = self._resolve_schema(schema)
                # Ensure we only return dict for request body
                if isinstance(resolved, dict):
                    return resolved, "json"
                return None, None

        # Handle form data
        if "application/x-www-form-urlencoded" in content:
            form_content = content["application/x-www-form-urlencoded"]
            if isinstance(form_content, dict):
                schema = form_content.get("schema", {})
                resolved = self._resolve_schema(schema)
                # Ensure we only return dict for request body
                if isinstance(resolved, dict):
                    return resolved, "form"
                return None, None

        return None, None

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
                    properties.update(body_props)
                    # Mark all body parameters
                    for prop_name in body_props:
                        parameter_locations[prop_name] = "body"

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
