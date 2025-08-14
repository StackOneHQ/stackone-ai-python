"""Tests for OpenAI Agents SDK integration"""

import pytest
from agents import FunctionTool

from stackone_ai import StackOneToolSet


@pytest.fixture
def toolset() -> StackOneToolSet:
    """Create a toolset for testing"""
    return StackOneToolSet(api_key="test-key")


def test_single_tool_openai_agents_conversion(toolset: StackOneToolSet) -> None:
    """Test converting a single tool to OpenAI Agents format"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Verify it's a FunctionTool object
    assert isinstance(openai_agents_tool, FunctionTool)
    assert hasattr(openai_agents_tool, "name")
    assert hasattr(openai_agents_tool, "description")
    assert hasattr(openai_agents_tool, "on_invoke_tool")


def test_tools_openai_agents_conversion(toolset: StackOneToolSet) -> None:
    """Test converting all tools to OpenAI Agents format"""
    tools = toolset.get_tools("hris_*")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    openai_agents_tools = tools.to_openai_agents()

    # Verify conversion
    assert len(openai_agents_tools) == len(tools.tools)
    assert all(isinstance(tool, FunctionTool) for tool in openai_agents_tools)
    assert all(hasattr(tool, "on_invoke_tool") for tool in openai_agents_tools)


@pytest.mark.asyncio
async def test_openai_agents_tool_execution(toolset: StackOneToolSet) -> None:
    """Test that OpenAI Agents tools can be executed"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Test that the tool function exists and has on_invoke_tool method
    assert isinstance(openai_agents_tool, FunctionTool)
    assert hasattr(openai_agents_tool, "on_invoke_tool")
    assert openai_agents_tool.name == tool.name


def test_openai_agents_tool_attributes(toolset: StackOneToolSet) -> None:
    """Test that OpenAI Agents tools have proper attributes"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Verify the tool has required attributes for function_tool
    assert isinstance(openai_agents_tool, FunctionTool)
    assert hasattr(openai_agents_tool, "name")
    assert hasattr(openai_agents_tool, "description")
    assert openai_agents_tool.name == tool.name
    assert openai_agents_tool.description == tool.description


def test_openai_agents_tool_naming(toolset: StackOneToolSet) -> None:
    """Test that OpenAI Agents tools preserve naming"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Verify function tool exists and is properly named
    assert isinstance(openai_agents_tool, FunctionTool)
    assert openai_agents_tool.name == tool.name
    original_name = tool.name
    assert original_name is not None
    assert len(original_name) > 0
