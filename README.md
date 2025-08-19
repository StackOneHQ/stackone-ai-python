# StackOne AI SDK

StackOne AI provides a unified interface for accessing various SaaS tools through AI-friendly APIs.

## Installation

```bash
pip install stackone-ai
```

## Quick Start

```python
from stackone_ai import StackOneToolSet

# Initialize with API key
toolset = StackOneToolSet()  # Uses STACKONE_API_KEY env var
# Or explicitly: toolset = StackOneToolSet(api_key="your-api-key")

# Get HRIS-related tools with glob patterns
tools = toolset.get_tools("hris_*", account_id="your-account-id")
# Exclude certain tools with negative patterns
tools = toolset.get_tools(["hris_*", "!hris_delete_*"])

# Use a specific tool with the new call method
employee_tool = tools.get_tool("hris_get_employee")
# Call with keyword arguments
employee = employee_tool.call(id="employee-id")
# Or with traditional execute method
employee = employee_tool.execute({"id": "employee-id"})
```

## Meta Tools (Beta)

Meta tools enable dynamic tool discovery and execution without hardcoding tool names:

```python
# Get meta tools for dynamic discovery
tools = toolset.get_tools("hris_*")
meta_tools = tools.meta_tools()

# Search for relevant tools using natural language
filter_tool = meta_tools.get_tool("meta_filter_relevant_tools")
results = filter_tool.call(query="manage employees", limit=5)

# Execute discovered tools dynamically
execute_tool = meta_tools.get_tool("meta_execute_tool")
result = execute_tool.call(toolName="hris_list_employees", params={"limit": 10})
```

## Features

- Unified interface for multiple SaaS tools
- AI-friendly tool descriptions and parameters
- **Tool Calling**: Direct method calling with `tool.call()` for intuitive usage
- **Glob Pattern Filtering**: Advanced tool filtering with patterns like `"hris_*"` and exclusions `"!hris_delete_*"`
- **Meta Tools** (Beta): Dynamic tool discovery and execution based on natural language queries
- Integration with popular AI frameworks:
  - Agno Agents
  - OpenAI Functions
  - LangChain Tools
  - CrewAI Tools
  - LangGraph Tool Node

## Documentation

For more examples and documentation, visit:

- [Error Handling](docs/error-handling.md)
- [StackOne Account IDs](docs/stackone-account-ids.md)
- [Available Tools](docs/available-tools.md)
- [File Uploads](docs/file-uploads.md)

## AI Framework Integration

- [Agno Integration](docs/agno_integration.md)
- [OpenAI Integration](docs/openai_integration.md)
- [LangChain Integration](docs/langchain_integration.md)
- [CrewAI Integration](docs/crewai_integration.md)
- [LangGraph Tool Node](docs/langgraph_tool_node.md)

## License

Apache 2.0 License