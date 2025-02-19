from stackone_ai.constants import OAS_DIR
from stackone_ai.models import ToolDefinition
from stackone_ai.specs.parser import OpenAPIParser


def load_specs() -> dict[str, dict[str, ToolDefinition]]:
    """
    Load all OpenAPI specs from the .stackone/oas directory

    Returns:
        Dict mapping vertical names to their tool definitions
    """
    tools = {}

    for spec_file in OAS_DIR.glob("*.json"):
        vertical = spec_file.stem
        parser = OpenAPIParser(spec_file)
        tools[vertical] = parser.parse_tools()

    return tools
