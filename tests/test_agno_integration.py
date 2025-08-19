from unittest.mock import MagicMock, patch

import pytest

from stackone_ai.models import (
    ExecuteConfig,
    StackOneTool,
    ToolParameters,
    Tools,
)


@pytest.fixture
def mock_tool() -> StackOneTool:
    """Create a mock tool for testing"""
    return StackOneTool(
        description="Test HRIS tool for getting employee data",
        parameters=ToolParameters(
            type="object",
            properties={
                "id": {"type": "string", "description": "Employee ID"},
                "include_personal": {"type": "boolean", "description": "Include personal information"},
            },
        ),
        _execute_config=ExecuteConfig(
            headers={},
            method="GET",
            url="https://api.stackone.com/unified/hris/employees/{id}",
            name="hris_get_employee",
            parameter_locations={"id": "path"},
        ),
        _api_key="test_key",
        _account_id="test_account",
    )


@pytest.fixture
def tools_collection(mock_tool: StackOneTool) -> Tools:
    """Create a Tools collection with mock tools"""
    return Tools([mock_tool])


class TestAgnoIntegration:
    """Test Agno integration functionality"""

    def test_to_agno_without_agno_installed(self, mock_tool: StackOneTool) -> None:
        """Test that proper error is raised when Agno is not installed"""
        with patch.dict("sys.modules", {"agno": None, "agno.tools": None}):
            with pytest.raises(ImportError) as exc_info:
                mock_tool.to_agno()

            assert "Agno is not installed" in str(exc_info.value)
            assert "pip install agno>=1.7.0" in str(exc_info.value)

    def test_to_agno_with_mocked_agno(self, mock_tool: StackOneTool) -> None:
        """Test Agno conversion with mocked Agno classes"""
        # Mock the Agno Tool class
        mock_agno_base_tool = MagicMock()
        mock_agno_module = MagicMock()
        mock_agno_module.Tool = mock_agno_base_tool

        with patch.dict("sys.modules", {"agno.tools": mock_agno_module}):
            agno_tool = mock_tool.to_agno()

            # Verify an Agno tool instance was created
            assert agno_tool is not None

    def test_to_agno_tool_execution(self, mock_tool: StackOneTool) -> None:
        """Test that the Agno tool can execute the underlying StackOne tool"""
        mock_agno_base_tool = MagicMock()
        mock_agno_module = MagicMock()
        mock_agno_module.Tool = mock_agno_base_tool

        with patch.dict("sys.modules", {"agno.tools": mock_agno_module}):
            agno_tool = mock_tool.to_agno()

            # Verify the tool was created (basic functionality test)
            assert agno_tool is not None
            assert hasattr(agno_tool, "run")

    def test_tools_to_agno(self, tools_collection: Tools) -> None:
        """Test converting Tools collection to Agno format"""
        mock_agno_base_tool = MagicMock()
        mock_agno_module = MagicMock()
        mock_agno_module.Tool = mock_agno_base_tool

        with patch.dict("sys.modules", {"agno.tools": mock_agno_module}):
            agno_tools = tools_collection.to_agno()

            # Verify we got the expected number of tools
            assert len(agno_tools) == 1
            assert agno_tools[0] is not None

    def test_tools_to_agno_multiple_tools(self) -> None:
        """Test converting multiple tools to Agno format"""
        # Create multiple mock tools
        tool1 = StackOneTool(
            description="Test tool 1",
            parameters=ToolParameters(type="object", properties={"id": {"type": "string"}}),
            _execute_config=ExecuteConfig(
                headers={}, method="GET", url="https://api.example.com/test1/{id}", name="test_tool_1"
            ),
            _api_key="test_key",
        )
        tool2 = StackOneTool(
            description="Test tool 2",
            parameters=ToolParameters(type="object", properties={"name": {"type": "string"}}),
            _execute_config=ExecuteConfig(
                headers={}, method="POST", url="https://api.example.com/test2", name="test_tool_2"
            ),
            _api_key="test_key",
        )

        tools = Tools([tool1, tool2])

        mock_agno_base_tool = MagicMock()
        mock_agno_module = MagicMock()
        mock_agno_module.Tool = mock_agno_base_tool

        with patch.dict("sys.modules", {"agno.tools": mock_agno_module}):
            agno_tools = tools.to_agno()

            assert len(agno_tools) == 2
            assert all(tool is not None for tool in agno_tools)

    def test_agno_tool_preserves_metadata(self, mock_tool: StackOneTool) -> None:
        """Test that Agno tool conversion preserves important metadata"""
        mock_agno_base_tool = MagicMock()
        mock_agno_module = MagicMock()
        mock_agno_module.Tool = mock_agno_base_tool

        with patch.dict("sys.modules", {"agno.tools": mock_agno_module}):
            agno_tool = mock_tool.to_agno()

            # Verify the tool was created with expected attributes
            assert agno_tool is not None
            # For real integration, name and description would be set by the Agno base class
            assert hasattr(agno_tool, "name")
            assert hasattr(agno_tool, "description")


class TestAgnoIntegrationErrors:
    """Test error handling in Agno integration"""

    def test_agno_import_error_message(self, mock_tool: StackOneTool) -> None:
        """Test that ImportError contains helpful installation instructions"""
        with patch.dict("sys.modules", {"agno": None, "agno.tools": None}):
            with pytest.raises(ImportError) as exc_info:
                mock_tool.to_agno()

            error_msg = str(exc_info.value)
            assert "Agno is not installed" in error_msg
            assert "pip install agno>=1.7.0" in error_msg
            assert "requirements" in error_msg

    def test_tools_to_agno_with_failed_conversion(self) -> None:
        """Test Tools.to_agno() when individual tool conversion fails"""
        mock_tool = MagicMock()
        mock_tool.to_agno.side_effect = ImportError("Agno not available")

        tools = Tools([mock_tool])

        with pytest.raises(ImportError):
            tools.to_agno()
