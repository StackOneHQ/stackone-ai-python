"""
This example demonstrates how to use StackOne tools with CrewAI.

CrewAI uses LangChain tools natively.

```bash
uv run examples/crewai_integration.py
```
"""

from crewai import Agent, Crew, Task

from stackone_ai import StackOneToolSet

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def crewai_integration():
    toolset = StackOneToolSet()
    tools = toolset.fetch_tools(actions=["bamboohr_*"], account_ids=[account_id])

    # CrewAI uses LangChain tools natively
    langchain_tools = tools.to_langchain()
    assert len(langchain_tools) > 0, "Expected at least one LangChain tool"

    for tool in langchain_tools:
        assert hasattr(tool, "name"), "Expected tool to have name"
        assert hasattr(tool, "description"), "Expected tool to have description"
        assert hasattr(tool, "_run"), "Expected tool to have _run method"

    agent = Agent(
        role="HR Manager",
        goal=f"What is the employee with the id {employee_id}?",
        backstory="With over 10 years of experience in HR and employee management, "
        "you excel at finding patterns in complex datasets.",
        llm="gpt-4o-mini",
        tools=langchain_tools,
        max_iter=2,
    )

    task = Task(
        description="What is the employee with the id c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA?",
        agent=agent,
        expected_output="A JSON object containing the employee's information",
    )

    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()
    assert result is not None, "Expected result to be returned"


if __name__ == "__main__":
    crewai_integration()
