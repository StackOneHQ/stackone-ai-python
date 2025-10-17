# StackOne AI SDK

[![PyPI version](https://badge.fury.io/py/stackone-ai.svg)](https://badge.fury.io/py/stackone-ai)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/StackOneHQ/stackone-ai-python)](https://github.com/StackOneHQ/stackone-ai-python/releases)

StackOne AI provides a unified interface for accessing various SaaS tools through AI-friendly APIs.

## Features

- Unified interface for multiple SaaS tools
- AI-friendly tool descriptions and parameters
- **Tool Calling**: Direct method calling with `tool.call()` for intuitive usage
- **Glob Pattern Filtering**: Advanced tool filtering with patterns like `"hris_*"` and exclusions `"!hris_delete_*"`
- **Meta Tools** (Beta): Dynamic tool discovery and execution based on natural language queries
- Integration with popular AI frameworks:
  - OpenAI Functions
  - LangChain Tools
  - CrewAI Tools
  - LangGraph Tool Node

## Requirements

- Python 3.9+ (core SDK functionality)
- Python 3.10+ (for MCP server and CrewAI examples)

## Installation

### Basic Installation

```bash
pip install stackone-ai

# Or with uv
uv add stackone-ai
```

### Optional Features

```bash
# Install with MCP server support (requires Python 3.10+)
uv add 'stackone-ai[mcp]'
# or 
pip install 'stackone-ai[mcp]'

# Install with CrewAI examples (requires Python 3.10+)
uv add 'stackone-ai[examples]'
# or
pip install 'stackone-ai[examples]'

# Install everything
uv add 'stackone-ai[mcp,examples]'
# or
pip install 'stackone-ai[mcp,examples]'
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

## Implicit Feedback (Beta)

The Python SDK can emit implicit behavioural feedback to LangSmith so you can triage low-quality tool results without manually tagging runs.

### Automatic configuration

Set `LANGSMITH_API_KEY` in your environment and the SDK will initialise the implicit feedback manager on first tool execution. You can optionally fine-tune behaviour with:

- `STACKONE_IMPLICIT_FEEDBACK_ENABLED` (`true`/`false`, defaults to `true` when an API key is present)
- `STACKONE_IMPLICIT_FEEDBACK_PROJECT` to pin a LangSmith project name
- `STACKONE_IMPLICIT_FEEDBACK_TAGS` with a comma-separated list of tags applied to every run

### Manual configuration

If you want custom session or user resolvers, call `configure_implicit_feedback` during start-up:

```python
from stackone_ai import configure_implicit_feedback

configure_implicit_feedback(
    api_key="/path/to/langsmith.key",
    project_name="stackone-agents",
    default_tags=["python-sdk"],
)
```

Providing your own `session_resolver`/`user_resolver` callbacks lets you derive identifiers from the request context before events are sent to LangSmith.

### Attaching session context to tool calls

Both `tool.execute` and `tool.call` accept an `options` keyword that is excluded from the API request but forwarded to the feedback manager:

```python
tool.execute(
    {"id": "employee-id"},
    options={
        "feedback_session_id": "chat-42",
        "feedback_user_id": "user-123",
        "feedback_metadata": {"conversation_id": "abc"},
    },
)
```

When two calls for the same session happen within a few seconds, the SDK emits a `refinement_needed` event, and you can inspect suitability scores directly in LangSmith.

## Integration Examples

<details>
<summary>LangChain Integration</summary>

StackOne tools work seamlessly with LangChain, enabling powerful AI agent workflows:

```python
from langchain_openai import ChatOpenAI
from stackone_ai import StackOneToolSet

# Initialize StackOne tools
toolset = StackOneToolSet()
tools = toolset.get_tools("hris_*", account_id="your-account-id")

# Convert to LangChain format
langchain_tools = tools.to_langchain()

# Use with LangChain models
model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools(langchain_tools)

# Execute AI-driven tool calls
response = model_with_tools.invoke("Get employee information for ID: emp123")

# Handle tool calls
for tool_call in response.tool_calls:
    tool = tools.get_tool(tool_call["name"])
    if tool:
        result = tool.execute(tool_call["args"])
        print(f"Result: {result}")
```

</details>

<details>
<summary>LangGraph Integration</summary>

StackOne tools convert to LangChain tools, which LangGraph consumes via its prebuilt nodes:

Prerequisites:

```bash
pip install langgraph langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from stackone_ai import StackOneToolSet
from stackone_ai.integrations.langgraph import to_tool_node, bind_model_with_tools

# Prepare tools
toolset = StackOneToolSet()
tools = toolset.get_tools("hris_*", account_id="your-account-id")
langchain_tools = tools.to_langchain()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build a small agent loop: LLM -> maybe tools -> back to LLM
graph = StateGraph(State)
graph.add_node("tools", to_tool_node(langchain_tools))

def call_llm(state: dict):
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm = bind_model_with_tools(llm, langchain_tools)
    resp = llm.invoke(state["messages"])  # returns AIMessage with optional tool_calls
    return {"messages": state["messages"] + [resp]}

graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
app = graph.compile()

_ = app.invoke({"messages": [("user", "Get employee with id emp123") ]})
```

</details>

<details>
<summary>CrewAI Integration (Python 3.10+)</summary>

CrewAI uses LangChain tools natively, making integration seamless:

> **Note**: CrewAI requires Python 3.10+. Install with `pip install 'stackone-ai[examples]'`

```python
from crewai import Agent, Crew, Task
from stackone_ai import StackOneToolSet

# Get tools and convert to LangChain format
toolset = StackOneToolSet()
tools = toolset.get_tools("hris_*", account_id="your-account-id")
langchain_tools = tools.to_langchain()

# Create CrewAI agent with StackOne tools
agent = Agent(
    role="HR Manager",
    goal="Analyze employee data and generate insights",
    backstory="Expert in HR analytics and employee management",
    tools=langchain_tools,
    llm="gpt-4o-mini"
)

# Define task and execute
task = Task(
    description="Find all employees in the engineering department",
    agent=agent,
    expected_output="List of engineering employees with their details"
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

</details>

## Feedback Collection

The SDK includes a feedback collection tool (`meta_collect_tool_feedback`) that allows users to submit feedback about their experience with StackOne tools. This tool is automatically included in the toolset and is designed to be invoked by AI agents after user permission.

```python
from stackone_ai import StackOneToolSet

toolset = StackOneToolSet()

# Get the feedback tool (included with "meta_*" pattern or all tools)
tools = toolset.get_tools("meta_*")
feedback_tool = tools.get_tool("meta_collect_tool_feedback")

# Submit feedback (typically invoked by AI after user consent)
result = feedback_tool.call(
    feedback="The HRIS tools are working great! Very fast response times.",
    account_id="acc_123456",
    tool_names=["hris_list_employees", "hris_get_employee"]
)
```

**Important**: The AI agent should always ask for user permission before submitting feedback:
- "Are you ok with sending feedback to StackOne? The LLM will take care of sending it."
- Only call the tool after the user explicitly agrees.

## Meta Tools (Beta)

Meta tools enable dynamic tool discovery and execution without hardcoding tool names:

```python
# Get meta tools for dynamic discovery
tools = toolset.get_tools("hris_*")
meta_tools = tools.meta_tools()

# Search for relevant tools using natural language
filter_tool = meta_tools.get_tool("meta_search_tools")
results = filter_tool.call(query="manage employees", limit=5)

# Execute discovered tools dynamically
execute_tool = meta_tools.get_tool("meta_execute_tool")
result = execute_tool.call(toolName="hris_list_employees", params={"limit": 10})
```

## Examples

For more examples, check out the [examples/](examples/) directory:

- [Error Handling](examples/error_handling.py)
- [StackOne Account IDs](examples/stackone_account_ids.py)
- [Available Tools](examples/available_tools.py)
- [File Uploads](examples/file_uploads.py)
- [OpenAI Integration](examples/openai_integration.py)
- [LangChain Integration](examples/langchain_integration.py)
- [CrewAI Integration](examples/crewai_integration.py)
- [LangGraph Tool Node](examples/langgraph_tool_node.py)
- [Meta Tools](examples/meta_tools_example.py)

## License

Apache 2.0 License
