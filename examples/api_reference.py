"""
## StackOneToolSet

The main class for accessing StackOne tools.

### Constructor

```python
StackOneToolSet(
    api_key: str | None = None,
    account_id: str | None = None
)
```

**Parameters:**
- `api_key`: Optional API key. If not provided, uses `STACKONE_API_KEY` env variable
- `account_id`: Optional account ID. If not provided, uses `STACKONE_ACCOUNT_ID` env variable

### Methods

#### get_tools

```python
def get_tools(
    vertical: str,
    account_id: str | None = None
) -> Tools
```

Get tools for a specific vertical.

**Parameters:**
- `vertical`: The vertical to get tools for (e.g. "hris", "crm")
- `account_id`: Optional account ID override. If not provided, uses the one from initialization

**Returns:**
- `Tools` instance containing available tools

## Tools

Container for Tool instances.

### Methods

#### get_tool

```python
def get_tool(name: str) -> BaseTool | None
```

Get a tool by its name.

**Parameters:**
- `name`: Name of the tool to get

**Returns:**
- `BaseTool` instance if found, None otherwise

#### to_openai

```python
def to_openai() -> list[dict]
```

Convert all tools to OpenAI function format.

**Returns:**
- List of tools in OpenAI function format

## BaseTool

Base class for individual tools.

### Methods

#### execute

```python
def execute(arguments: str | dict) -> dict[str, Any]
```

Execute the tool with the given parameters.

**Parameters:**
- `arguments`: Either a JSON string or dict of arguments

**Returns:**
- Tool execution results

#### to_openai_function

```python
def to_openai_function() -> dict
```

Convert this tool to OpenAI's function format.

**Returns:**
- Tool definition in OpenAI function format
"""

# Example usage of the API
from stackone_ai import StackOneToolSet

# Initialize with environment variables
toolset = StackOneToolSet()

# Get tools for HRIS vertical
tools = toolset.get_tools(vertical="hris")

# Get a specific tool
employee_tool = tools.get_tool("get_employee")
if employee_tool:
    # Execute the tool
    result = employee_tool.execute({"id": "employee123"})

    # Convert to OpenAI format
    openai_function = employee_tool.to_openai_function()
