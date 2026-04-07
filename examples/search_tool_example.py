#!/usr/bin/env python
"""
Example demonstrating dynamic tool discovery using search_tool.

The search tool allows AI agents to discover relevant tools based on natural language
queries without hardcoding tool names.

Prerequisites:
- STACKONE_API_KEY environment variable set
- STACKONE_ACCOUNT_ID environment variable set (comma-separated for multiple)
- At least one linked account in StackOne (this example uses BambooHR)

This example is runnable with the following command:
```bash
uv run examples/search_tool_example.py
```
"""

import json
import os

from stackone_ai import StackOneToolSet

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

# Read account IDs from environment — supports comma-separated values
_account_ids = [aid.strip() for aid in os.getenv("STACKONE_ACCOUNT_ID", "").split(",") if aid.strip()]


def example_search_tool_basic():
    """Basic example of using the search tool for tool discovery"""
    print("Example 1: Dynamic tool discovery\n")

    # Initialize StackOne toolset
    toolset = StackOneToolSet(search={})

    # Get all available tools using MCP-backed fetch_tools()
    all_tools = toolset.fetch_tools(account_ids=_account_ids)
    print(f"Total tools available: {len(all_tools)}")

    if not all_tools:
        print("No tools found. Check your linked accounts.")
        return

    # Get a search tool for dynamic discovery
    search_tool = toolset.get_search_tool()

    # Search for employee management tools — returns a Tools collection
    tools = search_tool("manage employees create update list", top_k=5, account_ids=_account_ids)

    print(f"Found {len(tools)} relevant tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print()


def example_search_modes():
    """Comparing semantic vs local search modes.

    Search config can be set at the constructor level or overridden per call:
    - Constructor: StackOneToolSet(search={"method": "semantic"})
    - Per-call: toolset.search_tools(query, search="local")

    The search method controls which backend search_tools() uses:
    - "semantic": cloud-based semantic vector search (higher accuracy for natural language)
    - "local": local BM25+TF-IDF hybrid search (no network call to semantic API)
    - "auto" (default): tries semantic first, falls back to local on failure
    """
    print("Example 2: Semantic vs local search modes\n")

    query = "manage employee time off"

    # Constructor-level config — semantic search as the default for this toolset
    print('Constructor config: StackOneToolSet(search={"method": "semantic"})')
    toolset_semantic = StackOneToolSet(search={"method": "semantic"})
    try:
        tools_semantic = toolset_semantic.search_tools(query, account_ids=_account_ids, top_k=5)
        print(f"  Found {len(tools_semantic)} tools:")
        for tool in tools_semantic:
            print(f"    - {tool.name}")
    except Exception as e:
        print(f"  Semantic search unavailable: {e}")
    print()

    # Constructor-level config — local search (no network call to semantic API)
    print('Constructor config: StackOneToolSet(search={"method": "local"})')
    toolset_local = StackOneToolSet(search={"method": "local"})
    tools_local = toolset_local.search_tools(query, account_ids=_account_ids, top_k=5)
    print(f"  Found {len(tools_local)} tools:")
    for tool in tools_local:
        print(f"    - {tool.name}")
    print()

    # Per-call override — constructor defaults can be overridden on each call
    print("Per-call override: constructor uses semantic, but this call uses local")
    tools_override = toolset_semantic.search_tools(query, account_ids=_account_ids, top_k=5, search="local")
    print(f"  Found {len(tools_override)} tools:")
    for tool in tools_override:
        print(f"    - {tool.name}")
    print()

    # Auto (default) — tries semantic, falls back to local
    print('Default: StackOneToolSet() uses search="auto" (semantic with local fallback)')
    toolset_auto = StackOneToolSet(search={})
    tools_auto = toolset_auto.search_tools(query, account_ids=_account_ids, top_k=5)
    print(f"  Found {len(tools_auto)} tools:")
    for tool in tools_auto:
        print(f"    - {tool.name}")
    print()


def example_top_k_config():
    """Configuring top_k at the constructor level vs per-call.

    Constructor-level top_k applies to all search_tools() and search_action_names()
    calls. Per-call top_k overrides the constructor default for that single call.
    """
    print("Example 3: top_k at constructor vs per-call\n")

    # Constructor-level top_k — all calls default to returning 3 results
    toolset = StackOneToolSet(search={"top_k": 3})

    query = "manage employee records"
    print(f'Constructor top_k=3: searching for "{query}"')
    tools_default = toolset.search_tools(query, account_ids=_account_ids)
    print(f"  Got {len(tools_default)} tools (constructor default)")
    for tool in tools_default:
        print(f"    - {tool.name}")
    print()

    # Per-call override — this single call returns up to 10 results
    print("Per-call top_k=10: overriding constructor default")
    tools_override = toolset.search_tools(query, account_ids=_account_ids, top_k=10)
    print(f"  Got {len(tools_override)} tools (per-call override)")
    for tool in tools_override:
        print(f"    - {tool.name}")
    print()


def example_search_tool_with_execution():
    """Example of discovering and executing tools dynamically"""
    print("Example 4: Dynamic tool execution\n")

    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI not installed: pip install openai")
        print()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipped: Set OPENAI_API_KEY to run this example.")
        print()
        return

    toolset = StackOneToolSet(search={})

    # Step 1: Search for relevant tools
    search_tool = toolset.get_search_tool()
    tools = search_tool("list all employees", top_k=3, account_ids=_account_ids)

    if not tools:
        print("No matching tools found.")
        print()
        return

    print(f"Found {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}")

    # Step 2: Let the LLM pick the right tool and params
    openai_tools = tools.to_openai()
    client = OpenAI()
    messages: list[dict] = [
        {"role": "user", "content": "List all employees. Use the available tools."},
    ]

    for _step in range(5):
        response = client.chat.completions.create(model="gpt-5.4", messages=messages, tools=openai_tools)
        choice = response.choices[0]

        if not choice.message.tool_calls:
            print(f"\nAnswer: {choice.message.content}")
            break

        messages.append(choice.message.model_dump(exclude_none=True))
        for tc in choice.message.tool_calls:
            print(f"  -> {tc.function.name}({tc.function.arguments[:80]})")
            tool = tools.get_tool(tc.function.name)
            if tool:
                try:
                    result = tool.execute(json.loads(tc.function.arguments))
                except Exception as e:
                    result = {"error": str(e)}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    print()


def example_with_openai():
    """Example of using search tool with OpenAI"""
    print("Example 5: Using search tool with OpenAI\n")

    try:
        from openai import OpenAI

        # Initialize OpenAI client
        client = OpenAI()

        # Initialize StackOne toolset
        toolset = StackOneToolSet(search={})

        # Search for BambooHR employee tools
        tools = toolset.search_tools("manage employees", account_ids=_account_ids, top_k=5)

        # Convert to OpenAI format
        openai_tools = tools.to_openai()

        # Create a chat completion with discovered tools
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an HR assistant with access to employee management tools.",
                },
                {"role": "user", "content": "Can you help me find tools for managing employee records?"},
            ],
            tools=openai_tools,
            tool_choice="auto",
        )

        print("OpenAI Response:", response.choices[0].message.content)

        if response.choices[0].message.tool_calls:
            print("\nTool calls made:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - {tool_call.function.name}")

    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
    except Exception as e:
        print(f"OpenAI example failed: {e}")

    print()


def example_with_langchain():
    """Example of using tools with LangChain"""
    print("Example 6: Using tools with LangChain\n")

    try:
        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(f"LangChain dependencies not installed: {e}")
        print("Install with: pip install langchain-openai")
        print()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipped: Set OPENAI_API_KEY to run this example.")
        print()
        return

    toolset = StackOneToolSet(search={})

    # Search for tools and convert to LangChain format
    tools = toolset.search_tools("list employees", account_ids=_account_ids, top_k=5)
    langchain_tools = list(tools.to_langchain())

    print(f"Available tools: {len(langchain_tools)}")
    for tool in langchain_tools:
        print(f"  - {tool.name}")

    # Bind tools to model and run
    model = ChatOpenAI(model="gpt-5.4").bind_tools(langchain_tools)
    tools_by_name = {t.name: t for t in langchain_tools}
    messages = [HumanMessage(content="What employee tools do I have access to?")]

    for _ in range(5):
        response = model.invoke(messages)
        if not response.tool_calls:
            print(f"\nAnswer: {response.content}")
            break

        messages.append(response)
        for tc in response.tool_calls:
            print(f"  -> {tc['name']}({json.dumps(tc['args'])[:80]})")
            tool = tools_by_name[tc["name"]]
            try:
                result = tool.invoke(tc["args"])
            except Exception as e:
                result = {"error": str(e)}
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("StackOne AI SDK - Search Tool Examples")
    print("=" * 60)
    print()

    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run these examples.")
        return

    if not _account_ids:
        print("Set STACKONE_ACCOUNT_ID to run these examples.")
        print("(Comma-separated for multiple accounts)")
        return

    # Basic examples that work without external APIs
    example_search_tool_basic()
    example_search_modes()
    example_top_k_config()
    example_search_tool_with_execution()

    # Examples that require OpenAI API
    if os.getenv("OPENAI_API_KEY"):
        example_with_openai()
        example_with_langchain()
    else:
        print("Set OPENAI_API_KEY to run OpenAI and LangChain examples\n")

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
