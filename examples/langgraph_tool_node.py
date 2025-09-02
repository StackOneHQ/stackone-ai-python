"""
Minimal LangGraph example identical to the README snippet.

Run:
    uv run examples/langgraph_tool_node.py

Prerequisites:
- `pip install langgraph langchain-openai`
- `STACKONE_API_KEY` and `OPENAI_API_KEY`
- Optionally set `STACKONE_ACCOUNT_ID` (required by some tools)
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from typing_extensions import TypedDict

from stackone_ai import StackOneToolSet
from stackone_ai.integrations.langgraph import bind_model_with_tools, to_tool_node


def main() -> None:
    load_dotenv()

    # Prepare tools
    account_id = os.getenv("STACKONE_ACCOUNT_ID")  # Set if your tools require it
    toolset = StackOneToolSet()
    tools = toolset.get_tools("hris_*", account_id=account_id)
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

    # Kick off with a simple instruction; replace IDs as needed
    _ = app.invoke({"messages": [("user", "Get employee with id emp123")]})


if __name__ == "__main__":
    main()
