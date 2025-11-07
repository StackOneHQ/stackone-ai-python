from unittest.mock import MagicMock, patch

import pytest

from stackone_ai.models import ExecuteConfig, ToolDefinition, ToolParameters
from stackone_ai.toolset import StackOneToolSet


def test_toolset_initialization():
    """Test StackOneToolSet initialization and tool creation"""
    mock_spec_content = {
        "paths": {
            "/employee/{id}": {
                "get": {
                    "operationId": "hris_get_employee",
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
            name="hris_get_employee",
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
        mock_dir.glob.return_value = [mock_path]

        # Setup parser mock
        mock_parser = MagicMock()
        mock_parser.spec = mock_spec_content
        mock_parser.parse_tools.return_value = {"hris_get_employee": mock_tool_def}
        mock_parser_class.return_value = mock_parser

        # Create and test toolset
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.get_tools(filter_pattern="hris_*", account_id="test_account")

        # Verify results
        assert len(tools) == 1
        tool = tools.get_tool("hris_get_employee")
        assert tool is not None
        assert tool.description == "Get employee details"
        assert tool._api_key == "test_key"
        assert tool._account_id == "test_account"

        # Verify the tool parameters
        assert tool.parameters.properties["id"]["type"] == "string"
        assert tool.parameters.properties["id"]["description"] == "Employee ID"


def test_empty_filter_result():
    """Test getting tools with a filter pattern that matches nothing"""
    toolset = StackOneToolSet(api_key="test_key")
    tools = toolset.get_tools(filter_pattern="unknown_*")
    assert len(tools) == 0


def test_toolset_with_base_url():
    """Test StackOneToolSet with a custom base_url"""
    mock_spec_content = {
        "paths": {
            "/employee/{id}": {
                "get": {
                    "operationId": "hris_get_employee",
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

    # Create mock tool definition with default URL
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
            name="hris_get_employee",
            headers={},
            parameter_locations={"id": "path"},
        ),
    )

    # Create mock tool definition with development URL
    mock_tool_def_dev = ToolDefinition(
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
            url="https://api.example-dev.com/employee/{id}",
            name="hris_get_employee",
            headers={},
            parameter_locations={"id": "path"},
        ),
    )

    # Create mock tool definition with experimental URL
    mock_tool_def_exp = ToolDefinition(
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
            url="https://api.example-exp.com/employee/{id}",
            name="hris_get_employee",
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
        mock_dir.glob.return_value = [mock_path]

        # Setup parser mock for default URL
        mock_parser = MagicMock()
        mock_parser.spec = mock_spec_content
        mock_parser.parse_tools.return_value = {"hris_get_employee": mock_tool_def}

        # Setup parser mock for development URL
        mock_parser_dev = MagicMock()
        mock_parser_dev.spec = mock_spec_content
        mock_parser_dev.parse_tools.return_value = {"hris_get_employee": mock_tool_def_dev}

        # Setup parser mock for experimental URL
        mock_parser_exp = MagicMock()
        mock_parser_exp.spec = mock_spec_content
        mock_parser_exp.parse_tools.return_value = {"hris_get_employee": mock_tool_def_exp}

        # Configure the mock parser class to return different instances based on base_url
        def get_parser(spec_path, base_url=None):
            if base_url == "https://api.example-dev.com":
                return mock_parser_dev
            elif base_url == "https://api.example-exp.com":
                return mock_parser_exp
            return mock_parser

        mock_parser_class.side_effect = get_parser

        # Test with default URL
        toolset = StackOneToolSet(api_key="test_key")
        tools = toolset.get_tools(filter_pattern="hris_*")
        tool = tools.get_tool("hris_get_employee")
        assert tool is not None
        assert tool._execute_config.url == "https://api.stackone.com/employee/{id}"

        # Test with development URL
        toolset_dev = StackOneToolSet(api_key="test_key", base_url="https://api.example-dev.com")
        tools_dev = toolset_dev.get_tools(filter_pattern="hris_*")
        tool_dev = tools_dev.get_tool("hris_get_employee")
        assert tool_dev is not None
        assert tool_dev._execute_config.url == "https://api.example-dev.com/employee/{id}"

        # Test with experimental URL
        toolset_exp = StackOneToolSet(api_key="test_key", base_url="https://api.example-exp.com")
        tools_exp = toolset_exp.get_tools(filter_pattern="hris_*")
        tool_exp = tools_exp.get_tool("hris_get_employee")
        assert tool_exp is not None
        assert tool_exp._execute_config.url == "https://api.example-exp.com/employee/{id}"


def test_set_accounts():
    """Test setting account IDs for filtering"""
    toolset = StackOneToolSet(api_key="test_key")
    result = toolset.set_accounts(["acc1", "acc2"])

    # Should return self for chaining
    assert result is toolset
    assert toolset._account_ids == ["acc1", "acc2"]


def test_filter_by_provider():
    """Test provider filtering"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test matching providers
    assert toolset._filter_by_provider("hris_list_employees", ["hris", "ats"])
    assert toolset._filter_by_provider("ats_create_job", ["hris", "ats"])

    # Test non-matching providers
    assert not toolset._filter_by_provider("crm_list_contacts", ["hris", "ats"])

    # Test case-insensitive matching
    assert toolset._filter_by_provider("HRIS_list_employees", ["hris"])
    assert toolset._filter_by_provider("hris_list_employees", ["HRIS"])


def test_filter_by_action():
    """Test action filtering with glob patterns"""
    toolset = StackOneToolSet(api_key="test_key")

    # Test exact match
    assert toolset._filter_by_action("hris_list_employees", ["hris_list_employees"])

    # Test glob pattern
    assert toolset._filter_by_action("hris_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("ats_list_employees", ["*_list_employees"])
    assert toolset._filter_by_action("hris_list_employees", ["hris_*"])
    assert toolset._filter_by_action("hris_create_employee", ["hris_*"])

    # Test non-matching patterns
    assert not toolset._filter_by_action("crm_list_contacts", ["*_list_employees"])
    assert not toolset._filter_by_action("ats_create_job", ["hris_*"])


@pytest.fixture
def mock_tools_setup():
    """Setup mocked tools for filtering tests"""
    # Create mock tool definitions
    tools_defs = {
        "hris_list_employees": ToolDefinition(
            description="List employees",
            parameters=ToolParameters(type="object", properties={}),
            execute=ExecuteConfig(
                method="GET",
                url="https://api.stackone.com/hris/employees",
                name="hris_list_employees",
                headers={},
            ),
        ),
        "hris_create_employee": ToolDefinition(
            description="Create employee",
            parameters=ToolParameters(type="object", properties={}),
            execute=ExecuteConfig(
                method="POST",
                url="https://api.stackone.com/hris/employees",
                name="hris_create_employee",
                headers={},
            ),
        ),
        "ats_list_employees": ToolDefinition(
            description="List ATS employees",
            parameters=ToolParameters(type="object", properties={}),
            execute=ExecuteConfig(
                method="GET",
                url="https://api.stackone.com/ats/employees",
                name="ats_list_employees",
                headers={},
            ),
        ),
        "crm_list_contacts": ToolDefinition(
            description="List contacts",
            parameters=ToolParameters(type="object", properties={}),
            execute=ExecuteConfig(
                method="GET",
                url="https://api.stackone.com/crm/contacts",
                name="crm_list_contacts",
                headers={},
            ),
        ),
    }

    with (
        patch("stackone_ai.toolset.OAS_DIR") as mock_dir,
        patch("stackone_ai.toolset.OpenAPIParser") as mock_parser_class,
    ):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_dir.glob.return_value = [mock_path]

        mock_parser = MagicMock()
        mock_parser.parse_tools.return_value = tools_defs
        mock_parser_class.return_value = mock_parser

        yield


def test_fetch_tools_no_filters(mock_tools_setup):
    """Test fetch_tools without any filters"""
    toolset = StackOneToolSet(api_key="test_key")
    tools = toolset.fetch_tools()

    # Should include all tools (4 regular + 1 feedback tool)
    assert len(tools) == 5


def test_fetch_tools_provider_filter(mock_tools_setup):
    """Test fetch_tools with provider filtering"""
    toolset = StackOneToolSet(api_key="test_key")

    # Filter by single provider
    tools = toolset.fetch_tools(providers=["hris"])
    assert len(tools) == 2
    assert tools.get_tool("hris_list_employees") is not None
    assert tools.get_tool("hris_create_employee") is not None

    # Filter by multiple providers
    tools = toolset.fetch_tools(providers=["hris", "ats"])
    assert len(tools) == 3
    assert tools.get_tool("hris_list_employees") is not None
    assert tools.get_tool("ats_list_employees") is not None


def test_fetch_tools_action_filter(mock_tools_setup):
    """Test fetch_tools with action filtering"""
    toolset = StackOneToolSet(api_key="test_key")

    # Exact action match
    tools = toolset.fetch_tools(actions=["hris_list_employees"])
    assert len(tools) == 1
    assert tools.get_tool("hris_list_employees") is not None

    # Glob pattern match
    tools = toolset.fetch_tools(actions=["*_list_employees"])
    assert len(tools) == 2
    assert tools.get_tool("hris_list_employees") is not None
    assert tools.get_tool("ats_list_employees") is not None


def test_fetch_tools_combined_filters(mock_tools_setup):
    """Test fetch_tools with combined filters"""
    toolset = StackOneToolSet(api_key="test_key")

    # Combine provider and action filters
    tools = toolset.fetch_tools(providers=["hris"], actions=["*_list_*"])
    assert len(tools) == 1
    assert tools.get_tool("hris_list_employees") is not None
    assert tools.get_tool("hris_create_employee") is None


def test_fetch_tools_with_set_accounts(mock_tools_setup):
    """Test fetch_tools using set_accounts"""
    toolset = StackOneToolSet(api_key="test_key")
    toolset.set_accounts(["acc1"])

    tools = toolset.fetch_tools(providers=["hris"])
    assert len(tools) == 2
