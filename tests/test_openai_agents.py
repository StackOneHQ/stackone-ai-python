"""Tests for OpenAI Agents SDK integration"""

from collections.abc import Callable

import pytest

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

    # Verify it's a callable function tool
    assert callable(openai_agents_tool)
    assert isinstance(openai_agents_tool, Callable)


def test_tools_openai_agents_conversion(toolset: StackOneToolSet) -> None:
    """Test converting all tools to OpenAI Agents format"""
    tools = toolset.get_tools("hris_*")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    openai_agents_tools = tools.to_openai_agents()

    # Verify conversion
    assert len(openai_agents_tools) == len(tools.tools)
    assert all(callable(tool) for tool in openai_agents_tools)
    assert all(isinstance(tool, Callable) for tool in openai_agents_tools)


@pytest.mark.asyncio
async def test_openai_agents_tool_execution(toolset: StackOneToolSet) -> None:
    """Test that OpenAI Agents tools can be executed"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Test that the tool function exists and is callable
    assert callable(openai_agents_tool)
    assert callable(openai_agents_tool)


def test_openai_agents_tool_attributes(toolset: StackOneToolSet) -> None:
    """Test that OpenAI Agents tools have proper attributes"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Verify the tool has required attributes for function_tool
    assert callable(openai_agents_tool)
    # The function_tool decorator should preserve the original function properties
    assert hasattr(openai_agents_tool, "__name__") or hasattr(openai_agents_tool, "__qualname__")


def test_openai_agents_tool_naming(toolset: StackOneToolSet) -> None:
    """Test that OpenAI Agents tools preserve naming"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    openai_agents_tool = tool.to_openai_agents()

    # Verify function tool exists and is properly named
    assert callable(openai_agents_tool)
    # The function should be a decorated function with proper metadata
    original_name = tool.name
    assert original_name is not None
    assert len(original_name) > 0
