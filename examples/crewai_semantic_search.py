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

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from stackone_ai import StackOneToolSet

load_dotenv()

_account_ids = [aid.strip() for aid in os.getenv("STACKONE_ACCOUNT_ID", "").split(",") if aid.strip()]


def crewai_semantic_search() -> None:
    toolset = StackOneToolSet()

    # Step 1: Preview — lightweight search returning action names and scores
    #   search_action_names() queries the semantic API without fetching full
    #   tool definitions. Useful for inspecting what's available before committing.
    preview = toolset.search_action_names(
        "book a meeting or check availability",
        account_ids=_account_ids
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
        account_ids=_account_ids
    )
    assert len(tools) > 0, "Expected at least one scheduling tool"

    print(f"Discovered {len(tools)} scheduling tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")
    print()

    # Step 3: Convert to CrewAI format
    crewai_tools = tools.to_crewai()

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
