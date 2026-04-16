"""StackOne tools with Pydantic AI.

```bash
uv run examples/pydantic_ai_integration.py
```
"""

from __future__ import annotations

import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

try:
    from pydantic_ai import Agent, Tool
    from pydantic_ai._function_schema import FunctionSchema
except ImportError:
    print("Install pydantic-ai to run this example: pip install pydantic-ai")
    raise SystemExit(1) from None

from pydantic_core import SchemaValidator, core_schema

from stackone_ai import StackOneToolSet

_VALIDATOR = SchemaValidator(core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema()))


def _to_pydantic_ai_tool(stackone_tool) -> Tool:
    """Convert a StackOneTool to a Pydantic AI Tool with proper JSON schema."""
    params_schema = stackone_tool.to_openai_function()["function"]["parameters"]

    def execute(**kwargs) -> str:
        return json.dumps(stackone_tool.execute(kwargs))

    fs = FunctionSchema(
        function=execute,
        name=stackone_tool.name,
        description=stackone_tool.description,
        validator=_VALIDATOR,
        json_schema=params_schema,
        takes_ctx=False,
        is_async=False,
    )
    return Tool(execute, name=stackone_tool.name, description=stackone_tool.description, function_schema=fs)


def pydantic_ai_integration() -> None:
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    for var in ["STACKONE_API_KEY", "STACKONE_ACCOUNT_ID", "OPENAI_API_KEY"]:
        if not os.getenv(var):
            print(f"Set {var} to run this example.")
            return

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(
        actions=["workday_list_workers", "workday_get_worker", "workday_get_current_user"],
        account_ids=[account_id],
    )
    pydantic_tools = [_to_pydantic_ai_tool(t) for t in tools]
    print(f"Loaded {len(pydantic_tools)} tools for Pydantic AI.")

    agent = Agent("openai:gpt-5.4", system_prompt="You are a helpful HR assistant.", tools=pydantic_tools)
    result = agent.run_sync("List the first 5 employees")
    print(f"Result:\n{result.output}")


if __name__ == "__main__":
    pydantic_ai_integration()
