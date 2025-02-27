"""
Example demonstrating how to use a custom base URL with StackOne tools.

This is useful for:
1. Testing against development APIs
2. Working with self-hosted StackOne instances

Usage:

```bash
uv run examples/custom_base_url.py
```
"""

from stackone_ai.toolset import StackOneToolSet


def custom_base_url():
    """
    Default base URL
    """
    default_toolset = StackOneToolSet()
    hris_tools = default_toolset.get_tools(filter_pattern="hris_*")

    assert len(hris_tools) > 0
    assert hris_tools[0]._execute_config.url.startswith("https://api.stackone.com")

    """
    Custom base URL
    """
    dev_toolset = StackOneToolSet(base_url="https://api.example-dev.com")
    dev_hris_tools = dev_toolset.get_tools(filter_pattern="hris_*")

    """
    Note this uses the same tools but substitutes the base URL
    """
    assert len(dev_hris_tools) > 0
    assert dev_hris_tools[0]._execute_config.url.startswith("https://api.example-dev.com")


if __name__ == "__main__":
    custom_base_url()
