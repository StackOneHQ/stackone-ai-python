"""Tests for Pydantic AI integration"""

import pytest
from pydantic_ai.tools import Tool as PydanticAITool

from stackone_ai import StackOneToolSet


@pytest.fixture
def toolset() -> StackOneToolSet:
    """Create a toolset for testing"""
    return StackOneToolSet(api_key="test-key")


def test_single_tool_pydantic_ai_conversion(toolset: StackOneToolSet) -> None:
    """Test converting a single tool to Pydantic AI format"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    pydantic_ai_tool = tool.to_pydantic_ai()

    # Verify it's a Pydantic AI tool
    assert isinstance(pydantic_ai_tool, PydanticAITool)
    assert pydantic_ai_tool.description == tool.description


def test_tools_pydantic_ai_conversion(toolset: StackOneToolSet) -> None:
    """Test converting all tools to Pydantic AI format"""
    tools = toolset.get_tools("hris_*")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    pydantic_ai_tools = tools.to_pydantic_ai()

    # Verify conversion
    assert len(pydantic_ai_tools) == len(tools.tools)
    assert all(isinstance(tool, PydanticAITool) for tool in pydantic_ai_tools)

    # Verify tool properties are preserved
    for i, pydantic_ai_tool in enumerate(pydantic_ai_tools):
        original_tool = tools.tools[i]
        assert pydantic_ai_tool.description == original_tool.description


@pytest.mark.asyncio
async def test_pydantic_ai_tool_execution(toolset: StackOneToolSet) -> None:
    """Test that Pydantic AI tools can be executed"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    pydantic_ai_tool = tool.to_pydantic_ai()

    # Test that the tool function exists and is callable
    assert callable(pydantic_ai_tool.function)
    assert callable(pydantic_ai_tool.function)


def test_pydantic_ai_tool_schema_generation(toolset: StackOneToolSet) -> None:
    """Test that Pydantic AI tools generate proper schemas"""
    tools = toolset.get_tools("hris_get_employee")

    if not tools.tools:
        pytest.skip("No tools found for testing")

    tool = tools.tools[0]
    pydantic_ai_tool = tool.to_pydantic_ai()

    # Verify the tool has required attributes
    assert hasattr(pydantic_ai_tool, "function")
    assert hasattr(pydantic_ai_tool, "description")
    assert pydantic_ai_tool.description is not None
    assert len(pydantic_ai_tool.description.strip()) > 0
