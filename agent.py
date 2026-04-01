from datetime import date
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOL_IMPLEMENTATIONS

load_dotenv()

@tool
def semantic_search_jobs(query: str, n_results: int = 5):
    """Search jobs using semantic similarity (vector search)."""
    return TOOL_IMPLEMENTATIONS["semantic_search_jobs"](query, n_results)

@tool
def get_job_details(job_ids: list[str]):
    """Get full job details for given job IDs."""
    return TOOL_IMPLEMENTATIONS["get_job_details"](job_ids)

@tool
def get_job_aggregate(operation: str, column: str, role_filter: str = None):
    """
    Use this to calculate stats.
    - operation: Must be one of ['COUNT', 'AVG', 'MIN', 'MAX']
    - column: Must be one of ['yearsexperience', 'posted_at']
    """
    return TOOL_IMPLEMENTATIONS["get_job_aggregate"](
        operation, column, role_filter
    )

@tool
def get_column_distribution(column: str, limit: int = 15):
    """
    Use this to get lists of top items, breakdowns, or distributions.
    - column: Must be one of ['role', 'seniority', 'location', 'company', 'yearsexperience']
    """
    return TOOL_IMPLEMENTATIONS["get_column_distribution"](column, limit)

@tool
def search_jobs_by_criteria(
    role: str = None,
    location: str = None,
    company: str = None,
    max_experience: int = None,
):
    """Find jobs by filtering specific fields like role, location, company name, or max years of experience."""
    return TOOL_IMPLEMENTATIONS["search_jobs_by_criteria"](
        role, location, company, max_experience
    )


@tool
def top_skills(role: str, limit: int = 10):
    """Get top required skills for a specific role."""
    return TOOL_IMPLEMENTATIONS["top_skills"](role, limit)


@tool
def top_skills_all(limit: int = 15):
    """Get top skills across all jobs."""
    return TOOL_IMPLEMENTATIONS["top_skills_all"](limit)


TOOLS = [
    semantic_search_jobs,
    get_job_details,
    get_job_aggregate,
    get_column_distribution,
    search_jobs_by_criteria,
    top_skills,
    top_skills_all,
]

#llm setup(for now its groq latter i will change it to clude)
def build_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY_CHAT"),
        model="llama-3.1-8b-instant",
        temperature=0,#deterministic (less random)
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]#Stores full chat history


SYSTEM_PROMPT = """You are an expert AI Career Assistant specialising in the tech job market.
You have access to a database and several hardcoded tools to answer user questions.

ALLOWED COLUMNS FOR CALCULATIONS:
- Use 'yearsexperience' for experience, background, tenure, or years worked.
- Use 'posted_at' for dates, time, or when jobs were posted.

RULES FOR BEING FLEXIBLE WITH TOOLS:
1. Semantic Column Mapping: Humans use messy words. If a user asks for "average experience", you must realize that maps to the 'yearsexperience' column.
2. Keyword Flexibility: If a user asks for stats on a "software developer", pass "software developer" into the 'role_filter'. The tool is programmed to split the words and look for jobs containing 'software' OR 'developer'.
3. Do not assume the database is empty! Always attempt to call the relevant tool with broad keywords before giving up and falling back to general knowledge.

TOOLS AT YOUR DISPOSAL
1. semantic_search_jobs
   Use for: finding jobs by natural-language description, skill match, or profile similarity.
   Examples: "find Python backend jobs", "jobs at fintech startups", "roles for ML engineers"

2. get_job_aggregate
   Use for: calculating COUNT, AVG, MIN, or MAX.
   - operation: 'COUNT', 'AVG', 'MIN', or 'MAX'
   - column: 'yearsexperience' or 'posted_at'
   Examples: "What is the average experience for Data Scientists?", "Count how many jobs require less than 5 years exp"

3. get_column_distribution
   Use for: lists of top items or general distributions.
   - column: 'role', 'seniority', 'location', 'company', 'yearsexperience'
   Examples: "Which companies hire the most?", "What are the top job locations?", "Give me a breakdown of jobs by seniority"

4. search_jobs_by_criteria
   Use for: filtering jobs by role, location, company, or max_experience.
   Examples: "Find React jobs in Tel Aviv", "Jobs at Google requiring under 3 years of experience"

5. top_skills and top_skills_all
   Use for: finding the most in-demand skills for a specific role or across all roles.
   Examples: "What skills should I learn for a DevOps role?", "What are the top skills overall in the market?"

6. get_job_details
   Use for: fetching the complete record of a specific job by ID.
   Use this after search results return IDs the user wants to know more about.

HOW TO ANSWER EVERY QUESTION
ALWAYS ground your answer in database data when a tool can provide it.
Use your general tech-industry knowledge ONLY when it is directly relevant to the user's question.

RESPONSE FORMAT
- Cite specific numbers from the DB (e.g. "**73%** of Backend job postings require Node.js")
- Use bullet points for skill lists or job listings
- Include job URLs when listing job postings — format as: [Apply here](url)
- Be warm, direct, and actionable — you are a career mentor, not just a data terminal
- If the DB has no data for a question, say so clearly and fall back to general knowledge, labeled as such: "Based on general industry knowledge (not in your current DB)..."
- Never fabricate job IDs, URLs, or company names

Today's date: {today}
"""


def build_agent():
    llm = build_llm().bind_tools(TOOLS) #bind tools to the llm


    def assistant(state: State):
        formatted_prompt = SYSTEM_PROMPT.format(#Adds system prompt
            today=date.today().strftime("%B %d, %Y")
        )

        messages = [SystemMessage(content=formatted_prompt)] + state["messages"]#Sends messages to LLM

        return {"messages": [llm.invoke(messages)]}#Returns response

    tool_node = ToolNode(TOOLS)#Executes tool calls made by LLM

    def route(state: State):#decision maker
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)

    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("assistant")

    graph.add_conditional_edges(
        "assistant", route, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "assistant")

    return graph.compile()


def main():
    agent = build_agent()
    history = []

    print("\nCareer Assistant Ready\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in ["exit"]:
            break

        history.append(HumanMessage(content=user))

        result = agent.invoke({"messages": history})#Run agent
        history = result["messages"]#Update history

        print("\nAssistant:\n", history[-1].content, "\n")


if __name__ == "__main__":
    main()