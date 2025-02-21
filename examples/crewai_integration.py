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
    tools = toolset.get_tools(
        vertical="hris",
        account_id=account_id,
    )

    # CrewAI uses LangChain tools natively
    langchain_tools = tools.to_langchain()

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
    print(crew.kickoff())


if __name__ == "__main__":
    crewai_integration()
