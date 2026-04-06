from datetime import date
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOLS
from prompts import SYSTEM_PROMPT

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY_CHAT"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )


def build_agent():
    llm = build_llm().bind_tools(TOOLS)

    def assistant(state: State):
        prompt = SYSTEM_PROMPT.format(today=date.today().strftime("%B %d, %Y"))
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    def route(state: State):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("assistant")
    graph.add_conditional_edges("assistant", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "assistant")

    return graph.compile()


def main():
    agent = build_agent()
    history = []
    print("\nCareer Assistant Ready\n")

    while True:
        user = input("You: ")
        if user.strip().lower() == "exit":
            break
        history.append(HumanMessage(content=user))
        result = agent.invoke({"messages": history})
        history = result["messages"]
        print("\nAssistant:\n", history[-1].content, "\n")


if __name__ == "__main__":
    main()