"""
Get available tools from your StackOne organisation based on the account id.

This example demonstrates different ways to filter and organize tools:
1. Getting all available tools
2. Filtering by vertical
3. Using multiple patterns for cross-vertical functionality
4. Filtering by specific operations
5. Combining multiple operation patterns

# TODO: experimental - get_available_tools(account_id="your_account_id")

```bash
uv run examples/available_tools.py
```
"""

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet

load_dotenv()


def get_available_tools() -> None:
    toolset = StackOneToolSet()

    # First, get all tools
    all_tools = toolset.get_tools()
    assert len(all_tools) > 100, "Expected at least 100 tools in total"

    # Then, let's get just HRIS tools using a vertical filter
    hris_tools = toolset.get_tools("hris_*")
    assert len(hris_tools) > 10, "Expected at least 10 HRIS tools"

    # Now, let's get people-related tools across verticals
    people_tools = toolset.get_tools(
        [
            "hris_*employee*",
            "crm_*contact*",
        ]
    )
    assert len(people_tools) > 20, "Expected at least 20 people-related tools"
    for tool in people_tools:
        assert "employee" in tool.name or "contact" in tool.name, (
            f"Tool {tool.name} doesn't contain 'employee' or 'contact'"
        )

    # We can also filter by specific operations across all verticals
    upload_tools = toolset.get_tools("*upload*")
    assert len(upload_tools) > 0, "Expected at least one upload tool"
    for tool in upload_tools:
        assert "upload" in tool.name.lower(), f"Tool {tool.name} doesn't contain 'upload'"

    # Get all tools except HRIS
    non_hris_tools = toolset.get_tools("!hris_*")
    assert len(non_hris_tools) > 0, "Expected at least one non-HRIS tool"
    for tool in non_hris_tools:
        assert not tool.name.startswith("hris_"), f"Tool {tool.name} should not be an HRIS tool"

    # Complex filtering with positive and negative patterns
    list_tools = toolset.get_tools(
        [
            "*list*",  # Include list operations
            "*search*",  # Include search operations
            "!*delete*",  # Exclude delete operations
            "!*remove*",  # Exclude remove operations
        ]
    )
    assert len(list_tools) > 0, "Expected at least one list/search tool"
    for tool in list_tools:
        # Should match positive patterns
        assert any(op in tool.name.lower() for op in ["list", "search"]), (
            f"Tool {tool.name} doesn't contain 'list' or 'search'"
        )
        # Should not match negative patterns
        assert not any(op in tool.name.lower() for op in ["delete", "remove"]), (
            f"Tool {tool.name} contains excluded operation"
        )


if __name__ == "__main__":
    get_available_tools()
