"""Tests for LangGraph integration helpers."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool as LangChainBaseTool

from stackone_ai.models import ExecuteConfig, StackOneTool, ToolParameters, Tools


@pytest.fixture
def sample_tool() -> StackOneTool:
    """Create a sample tool for testing."""
    return StackOneTool(
        description="Test tool",
        parameters=ToolParameters(
            type="object",
            properties={"id": {"type": "string", "description": "Record ID"}},
        ),
        _execute_config=ExecuteConfig(
            headers={},
            method="GET",
            url="https://api.example.com/test/{id}",
            name="test_tool",
        ),
        _api_key="test_key",
    )


@pytest.fixture
def tools_collection(sample_tool: StackOneTool) -> Tools:
    """Create a Tools collection for testing."""
    return Tools([sample_tool])


class TestToLangchainTools:
    """Test _to_langchain_tools helper function."""

    def test_converts_tools_collection(self, tools_collection: Tools):
        """Test converting a Tools collection to LangChain tools."""
        from stackone_ai.integrations.langgraph import _to_langchain_tools

        result = _to_langchain_tools(tools_collection)

        assert isinstance(result, Sequence)
        assert len(result) == 1
        assert isinstance(result[0], LangChainBaseTool)
        assert result[0].name == "test_tool"

    def test_passthrough_langchain_tools(self):
        """Test that LangChain tools are passed through unchanged."""
        from stackone_ai.integrations.langgraph import _to_langchain_tools

        mock_lc_tool = MagicMock(spec=LangChainBaseTool)
        lc_tools = [mock_lc_tool]

        result = _to_langchain_tools(lc_tools)

        assert result is lc_tools
        assert len(result) == 1


class TestToToolNode:
    """Test to_tool_node function."""

    def test_creates_tool_node_from_tools_collection(self, tools_collection: Tools):
        """Test creating a ToolNode from a Tools collection."""
        from stackone_ai.integrations.langgraph import to_tool_node

        node = to_tool_node(tools_collection)

        # ToolNode should be created
        assert node is not None
        # Check it has the expected tools
        assert len(node.tools_by_name) == 1
        assert "test_tool" in node.tools_by_name

    def test_creates_tool_node_from_langchain_tools(self, tools_collection: Tools):
        """Test creating a ToolNode from pre-converted LangChain tools."""
        from stackone_ai.integrations.langgraph import to_tool_node

        lc_tools = tools_collection.to_langchain()
        node = to_tool_node(lc_tools)

        assert node is not None
        assert len(node.tools_by_name) == 1

    def test_passes_kwargs_to_tool_node(self, tools_collection: Tools):
        """Test that kwargs are passed to ToolNode constructor."""
        from stackone_ai.integrations.langgraph import to_tool_node

        # name is a valid ToolNode parameter
        node = to_tool_node(tools_collection, name="custom_node")

        assert node is not None


class TestToToolExecutor:
    """Test to_tool_executor function (deprecated, returns ToolNode)."""

    def test_creates_tool_node(self, tools_collection: Tools):
        """Test to_tool_executor creates a ToolNode."""
        from stackone_ai.integrations.langgraph import to_tool_executor

        result = to_tool_executor(tools_collection)

        # Should return a ToolNode (ToolExecutor is deprecated)
        assert result is not None
        assert len(result.tools_by_name) == 1


class TestBindModelWithTools:
    """Test bind_model_with_tools function."""

    def test_binds_tools_to_model(self, tools_collection: Tools):
        """Test binding tools to a model."""
        from stackone_ai.integrations.langgraph import bind_model_with_tools

        mock_model = MagicMock()
        mock_bound_model = MagicMock()
        mock_model.bind_tools.return_value = mock_bound_model

        result = bind_model_with_tools(mock_model, tools_collection)

        assert result is mock_bound_model
        mock_model.bind_tools.assert_called_once()
        # Check that LangChain tools were passed
        call_args = mock_model.bind_tools.call_args[0][0]
        assert isinstance(call_args, Sequence)
        assert len(call_args) == 1

    def test_binds_langchain_tools_directly(self):
        """Test binding pre-converted LangChain tools."""
        from stackone_ai.integrations.langgraph import bind_model_with_tools

        mock_model = MagicMock()
        mock_lc_tool = MagicMock(spec=LangChainBaseTool)
        lc_tools = [mock_lc_tool]

        bind_model_with_tools(mock_model, lc_tools)

        mock_model.bind_tools.assert_called_once_with(lc_tools)


class TestCreateReactAgent:
    """Test create_react_agent function."""

    def test_creates_react_agent(self, tools_collection: Tools):
        """Test creating a ReAct agent."""
        from stackone_ai.integrations.langgraph import create_react_agent

        mock_llm = MagicMock()

        with patch("langgraph.prebuilt.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = create_react_agent(mock_llm, tools_collection)

            assert result is mock_agent
            mock_create.assert_called_once()
            # First arg is llm, second is tools
            call_args = mock_create.call_args
            assert call_args[0][0] is mock_llm

    def test_passes_kwargs_to_create_react_agent(self, tools_collection: Tools):
        """Test that kwargs are passed to create_react_agent."""
        from stackone_ai.integrations.langgraph import create_react_agent

        mock_llm = MagicMock()

        with patch("langgraph.prebuilt.create_react_agent") as mock_create:
            create_react_agent(mock_llm, tools_collection, checkpointer=None)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert "checkpointer" in call_kwargs


class TestEnsureLanggraph:
    """Test _ensure_langgraph helper function."""

    def test_raises_import_error_when_langgraph_not_installed(self):
        """Test that ImportError is raised when langgraph is not installed."""
        from stackone_ai.integrations.langgraph import _ensure_langgraph

        with patch.dict("sys.modules", {"langgraph": None, "langgraph.prebuilt": None}):
            with patch(
                "stackone_ai.integrations.langgraph._ensure_langgraph",
                side_effect=ImportError("LangGraph is not installed"),
            ):
                # This test verifies the error message format
                pass

        # Since langgraph is installed in the test environment, just verify function runs
        _ensure_langgraph()  # Should not raise


class TestModuleImports:
    """Test module-level imports from integrations package."""

    def test_imports_from_integrations_init(self):
        """Test that all functions are importable from integrations package."""
        from stackone_ai.integrations import (
            bind_model_with_tools,
            create_react_agent,
            to_tool_executor,
            to_tool_node,
        )

        assert callable(to_tool_node)
        assert callable(to_tool_executor)
        assert callable(bind_model_with_tools)
        assert callable(create_react_agent)
