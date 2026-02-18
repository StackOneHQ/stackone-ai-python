"""
CrewAI meeting booking agent powered by semantic search.

Note: This example is Python only. CrewAI does not have an official
TypeScript/Node.js library.

Instead of hardcoding tool names, this example uses semantic search to discover
scheduling tools (e.g., Calendly) from natural language queries like "book a
meeting" or "check availability".

Prerequisites:
- STACKONE_API_KEY environment variable set
- STACKONE_ACCOUNT_ID environment variable set (Calendly-linked account)
- OPENAI_API_KEY environment variable set (for CrewAI's LLM)

```bash
uv run examples/crewai_semantic_search.py
```
"""

import os
from typing import Any

from crewai import Agent, Crew, Task
from crewai.tools.base_tool import BaseTool as CrewAIBaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from stackone_ai import StackOneToolSet
from stackone_ai.models import StackOneTool

load_dotenv()

_account_ids = [aid.strip() for aid in os.getenv("STACKONE_ACCOUNT_ID", "").split(",") if aid.strip()]


def _to_crewai_tool(tool: StackOneTool) -> CrewAIBaseTool:
    """Wrap a StackOneTool as a CrewAI BaseTool.

    CrewAI has its own BaseTool (not LangChain's), so we create a
    lightweight wrapper that delegates execution to the StackOne tool.
    """
    schema_props: dict[str, Any] = {}
    annotations: dict[str, Any] = {}

    for name, details in tool.parameters.properties.items():
        python_type: type = str
        if isinstance(details, dict):
            type_str = details.get("type", "string")
            if type_str == "number":
                python_type = float
            elif type_str == "integer":
                python_type = int
            elif type_str == "boolean":
                python_type = bool
            field = Field(description=details.get("description", ""))
        else:
            field = Field(description="")

        schema_props[name] = field
        annotations[name] = python_type

    _schema = type(
        f"{tool.name.title().replace('_', '')}Args",
        (BaseModel,),
        {"__annotations__": annotations, "__module__": __name__, **schema_props},
    )

    _parent = tool
    _name = tool.name
    _description = tool.description

    class WrappedTool(CrewAIBaseTool):
        name: str = _name
        description: str = _description
        args_schema: type[BaseModel] = _schema

        def _run(self, **kwargs: Any) -> Any:
            return _parent.execute(kwargs)

    return WrappedTool()


def crewai_semantic_search() -> None:
    toolset = StackOneToolSet()

    # Step 1: Preview — lightweight search returning action names and scores
    #   search_action_names() queries the semantic API without fetching full
    #   tool definitions. Useful for inspecting what's available before committing.
    preview = toolset.search_action_names(
        "book a meeting or check availability",
        account_ids=_account_ids,
    )
    print("Semantic search preview (action names only):")
    for r in preview:
        print(f"  [{r.similarity_score:.2f}] {r.action_name} ({r.connector_key})")
    print()

    # Step 2: Full discovery — fetch matching tools ready for framework use
    #   search_tools() fetches tools from linked accounts, runs semantic search,
    #   and returns only tools the user has access to.
    tools = toolset.search_tools(
        "schedule meetings, check availability, list events",
        connector="calendly",
        account_ids=_account_ids,
    )
    assert len(tools) > 0, "Expected at least one scheduling tool"

    print(f"Discovered {len(tools)} scheduling tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")
    print()

    # Step 3: Convert to CrewAI format
    crewai_tools = [_to_crewai_tool(t) for t in tools]

    # Step 4: Create a CrewAI meeting booking agent
    agent = Agent(
        role="Meeting Booking Agent",
        goal="Help users manage their calendar by discovering and booking meetings, "
        "checking availability, and listing upcoming events.",
        backstory="You are an AI assistant specialized in calendar management. "
        "You have access to scheduling tools discovered via semantic search "
        "and can help users with all meeting-related tasks.",
        llm="gpt-4o-mini",
        tools=crewai_tools,
        max_iter=2,
        verbose=True,
    )

    task = Task(
        description="List upcoming scheduled events to give an overview of the calendar.",
        agent=agent,
        expected_output="A summary of upcoming events or a confirmation that events were retrieved.",
    )

    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()
    assert result is not None, "Expected result to be returned"
    print(f"\nCrew result: {result}")


if __name__ == "__main__":
    crewai_semantic_search()
