from unittest.mock import MagicMock, patch

from stackone_ai.toolset import StackOneToolSet


def test_toolset_initialization():
    """Test StackOneToolSet initialization and tool creation"""
    mock_spec_content = {
        "paths": {
            "/employee/{id}": {
                "get": {
                    "x-speakeasy-name-override": "get_employee",
                    "description": "Get employee details",
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

    # Mock the file operations instead of load_specs
    with (
        patch("stackone_ai.toolset.OAS_DIR") as mock_dir,
        patch("json.load") as mock_json,
    ):
        # Setup mocks
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_dir.__truediv__.return_value = mock_path
        mock_json.return_value = mock_spec_content

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
    tools = toolset.get_tools(vertical="unknown")
    assert len(tools) == 0
