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

# Get HRIS-related tools
tools = toolset.get_tools("hris_*", account_id="your-account-id")

# Use a specific tool
employee_tool = tools.get_tool("hris_get_employee")
employee = employee_tool.execute({"id": "employee-id"})
```

## Features

- Unified interface for multiple SaaS tools
- AI-friendly tool descriptions and parameters
- Integration with popular AI frameworks:
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

- [OpenAI Integration](docs/openai-integration.md)
- [LangChain Integration](docs/langchain-integration.md)
- [CrewAI Integration](docs/crewai-integration.md)
- [LangGraph Tool Node](docs/langgraph-tool-node.md)

## License

Apache 2.0 License