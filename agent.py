import os
import time
import re
from datetime import date
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from tools import (
    semantic_search_jobs as _semantic_search,
    query_jobs_database  as _query_db,
    get_job_details      as _get_details,
)
from prompts import SYSTEM_PROMPT


#tools
@tool
def semantic_search_jobs(query: str, n_results: int = 5) -> dict:
    """Search jobs by natural language query."""
    if not query or not query.strip():
        return {"error": "query must not be empty"}
    return _semantic_search(query=query.strip(), n_results=min(max(n_results, 1), 20))


@tool
def query_jobs_database(sql: str, description: str) -> dict:
    """Run read-only SQL query on jobs database."""
    if not sql or not sql.strip():
        return {"error": "sql must not be empty"}
    return _query_db(sql=sql.strip(), description=description or "query")


@tool
def get_job_details(job_ids: list[str]) -> dict:
    """Get full job details by job IDs."""
    if not job_ids:
        return {"error": "job_ids must not be empty"}
    return _get_details(job_ids=job_ids)

TOOLS = [semantic_search_jobs, query_jobs_database, get_job_details]


# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────

def _build_llm():
    api_key = os.getenv("GROQ_API_KEY_CHAT") or os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    if not api_key:
        raise EnvironmentError("Missing GROQ API key")

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0,
        max_tokens=1024,
    )



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]



def _build_agent():
    llm = _build_llm().bind_tools(TOOLS)

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(today=date.today().isoformat())
    )

    def assistant(state: AgentState):
        messages = [system_msg] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Tool node (handles tool execution automatically)
    tool_node = ToolNode(TOOLS)

    # Router: decide whether to call tool or finish
    def should_continue(state: AgentState):
        last_msg = state["messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    # Build graph
    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("assistant")

    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "assistant")

    return builder.compile()



def _extract_retry_seconds(error_message: str) -> float:
    match = re.search(r"try again in ([0-9.]+)s", str(error_message))
    return float(match.group(1)) if match else 5.0


def _invoke_with_retry(agent, messages: list, max_retries: int = 4):
    for attempt in range(max_retries):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            err = str(e)

            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            is_too_large = "413" in err or "Request too large" in err

            if is_too_large:
                print("\n[INFO] Response too large.")
                raise

            if is_rate_limit and attempt < max_retries - 1:
                wait = _extract_retry_seconds(err) + 1
                print(f"[Rate limit] waiting {wait:.1f}s...")
                time.sleep(wait)
                continue

            raise

    raise RuntimeError("Max retries exceeded")



def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("AI Career Assistant ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    try:
        agent = _build_agent()
    except Exception as e:
        print("[ERROR]", e)
        return

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        conversation.append(HumanMessage(content=user_input))

        try:
            result = _invoke_with_retry(agent, conversation)

            conversation = result["messages"]
            reply = conversation[-1].content

            print("\nAssistant:\n", reply, "\n")

        except Exception as e:
            print("[ERROR]", e)
            if conversation and conversation[-1].type == "human":
                conversation.pop()


if __name__ == "__main__":
    main()