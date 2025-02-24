from unittest.mock import MagicMock, patch

import pytest
from stackone_ai.models import ExecuteConfig, ToolDefinition, ToolParameters
from stackone_ai.toolset import StackOneToolSet, ToolsetLoadError


def test_toolset_initialization():
    """Test StackOneToolSet initialization and tool creation"""
    mock_spec_content = {
        "paths": {
            "/employee/{id}": {
                "get": {
                    "operationId": "get_employee",
                    "summary": "Get employee details",
                    "parameters": [
                        {
                            "in": "path",
                            "name": "id",
                            "schema": {"type": "string"},
                            "description": "Employee ID",
                        }
                    ],
                }
            }
        }
    }

    # Create mock tool definition
    mock_tool_def = ToolDefinition(
        description="Get employee details",
        parameters=ToolParameters(
            type="object",
            properties={
                "id": {
                    "type": "string",
                    "description": "Employee ID",
                }
            },
        ),
        execute=ExecuteConfig(
            method="GET",
            url="https://api.stackone.com/employee/{id}",
            name="get_employee",
            headers={},
            parameter_locations={"id": "path"},
        ),
    )

    # Mock the OpenAPIParser and file operations
    with (
        patch("stackone_ai.toolset.OAS_DIR") as mock_dir,
        patch("stackone_ai.toolset.OpenAPIParser") as mock_parser_class,
    ):
        # Setup mocks
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_dir.__truediv__.return_value = mock_path

        # Setup parser mock
        mock_parser = MagicMock()
        mock_parser.spec = mock_spec_content
        mock_parser.parse_tools.return_value = {"get_employee": mock_tool_def}
        mock_parser_class.return_value = mock_parser

        # Create and test toolset
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.get_tools(vertical="hris", account_id="test_account")

        # Verify results
        assert len(tools) == 1
        tool = tools.get_tool("get_employee")
        assert tool is not None
        assert tool.description == "Get employee details"
        assert tool._api_key == "test_key"
        assert tool._account_id == "test_account"

        # Verify the tool parameters
        assert tool.parameters.properties["id"]["type"] == "string"
        assert tool.parameters.properties["id"]["description"] == "Employee ID"


def test_unknown_vertical():
    """Test getting tools for unknown vertical"""
    toolset = StackOneToolSet(api_key="test_key")
    with pytest.raises(ToolsetLoadError, match="No spec file found for vertical: unknown"):
        toolset.get_tools(vertical="unknown")
