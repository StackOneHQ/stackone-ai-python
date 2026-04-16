"""
This example demonstrates how to use StackOne tools with CrewAI.

CrewAI uses LangChain tools natively.

```bash
uv run examples/crewai_integration.py
```
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from crewai import Agent, Crew, Task

from stackone_ai import StackOneToolSet


def crewai_integration():
    account_id = os.getenv("STACKONE_ACCOUNT_ID")
    if not os.getenv("STACKONE_API_KEY"):
        print("Set STACKONE_API_KEY to run this example.")
        return
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this example.")
        return

    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(
        actions=["workday_list_workers", "workday_get_worker", "workday_get_current_user"],
        account_ids=[account_id],
    )

    # CrewAI uses LangChain tools natively
    langchain_tools = tools.to_langchain()
    print(f"Loaded {len(langchain_tools)} tools for CrewAI.")

    agent = Agent(
        role="HR Manager",
        goal="List the first 5 employees in the company",
        backstory="With over 10 years of experience in HR and employee management, "
        "you excel at finding patterns in complex datasets.",
        llm="gpt-5.1",
        tools=langchain_tools,
        max_iter=2,
    )

    task = Task(
        description="List the first 5 employees in the company",
        agent=agent,
        expected_output="A JSON object containing the employee's information",
    )

    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()
    print(f"Crew result:\n{result}")


if __name__ == "__main__":
    crewai_integration()
