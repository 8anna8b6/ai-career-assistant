from __future__ import annotations
import sys
import json
from typing import Optional, List

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool          
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import GROQ_API_KEY_CHAT, GROQ_MODEL, VALID_ROLES, VALID_SENIORITY
from db import get_connection
from chroma_store import init_chroma, search_jobs


# ══════════════════════════════════════════════════════════════════════════
# Shared singletons
# ══════════════════════════════════════════════════════════════════════════

_conn       = None
_collection = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = get_connection()
    return _conn


def _get_collection():
    global _collection
    if _collection is None:
        _collection = init_chroma()
    return _collection


# ══════════════════════════════════════════════════════════════════════════
# Tool 1 – filter_jobs (structured SQL builder)
# ══════════════════════════════════════════════════════════════════════════

class FilterJobsInput(BaseModel):
    role: Optional[str] = Field(
        None,
        description=f"Job role category. Must be one of: {', '.join(sorted(VALID_ROLES))}"
    )
    seniority: Optional[str] = Field(
        None,
        description=f"Seniority level. Must be one of: {', '.join(sorted(VALID_SENIORITY))}"
    )
    location: Optional[str] = Field(
        None,
        description="City or region to filter by, e.g. 'Tel Aviv', 'Remote'"
    )
    min_years: Optional[int] = Field(
        None,
        description="Minimum years of experience required"
    )
    max_years: Optional[int] = Field(
        None,
        description="Maximum years of experience required"
    )
    skills: Optional[List[str]] = Field(
        None,
        description="List of required skills to filter by, e.g. ['Python', 'Docker', 'PostgreSQL']"
    )
    company: Optional[str] = Field(
        None,
        description="Company name to filter by"
    )
    limit: int = Field(
        10,
        description="Maximum number of results to return (default 10, max 25)"
    )


def filter_jobs(
    role: Optional[str] = None,
    seniority: Optional[str] = None,
    location: Optional[str] = None,
    min_years: Optional[int] = None,
    max_years: Optional[int] = None,
    skills: Optional[List[str]] = None,
    company: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Build and execute a parameterised SQL query from typed inputs."""

    if role and role not in VALID_ROLES:
        return f"Error: invalid role '{role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}"
    if seniority and seniority not in VALID_SENIORITY:
        return f"Error: invalid seniority '{seniority}'. Valid values: {', '.join(sorted(VALID_SENIORITY))}"

    limit = min(int(limit), 25)

    conditions: list[str] = []
    params: list = []

    if role:
        conditions.append("role = %s")
        params.append(role)
    if seniority:
        conditions.append("seniority = %s")
        params.append(seniority)
    if location:
        conditions.append("location ILIKE %s")
        params.append(f"%{location}%")
    if min_years is not None:
        conditions.append("yearsexperience >= %s")
        params.append(min_years)
    if max_years is not None:
        conditions.append("yearsexperience <= %s")
        params.append(max_years)
    if skills:
        for skill in skills:
            conditions.append("%s ILIKE ANY(skills_must)")
            params.append(skill)
    if company:
        conditions.append("company ILIKE %s")
        params.append(f"%{company}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"""
        SELECT id, title, company, role, seniority, location,
               skills_must, yearsexperience, url, posted_at
        FROM jobs
        {where}
        ORDER BY posted_at DESC
        LIMIT %s
    """
    params.append(limit)

    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        if not rows:
            return "No jobs found matching those filters."

        results = []
        for row in rows:
            record = dict(zip(cols, row))
            for k, v in record.items():
                if isinstance(v, (list, tuple)):
                    record[k] = list(v)
                elif hasattr(v, "isoformat"):
                    record[k] = v.isoformat()
            results.append(record)

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Database error: {e}"


# ══════════════════════════════════════════════════════════════════════════
# Tool 2 – semantic_search (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════

class SemanticSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language description of the ideal job. "
            "e.g. 'senior Python backend engineer with Kubernetes and AWS experience'"
        )
    )
    n_results: int = Field(
        5,
        description="Number of top results to return (default 5)"
    )


def semantic_search(query: str, n_results: int = 5) -> str:
    """Search ChromaDB for jobs semantically similar to the query."""
    query = query.strip()
    if not query:
        return "Error: please provide a non-empty search query."

    try:
        collection = _get_collection()
        hits = search_jobs(collection, query, n_results=n_results * 3)

        if not hits:
            return "No semantically similar jobs found."

        seen: dict[str, dict] = {}
        for hit in hits:
            jid = hit["metadata"].get("job_id", hit["id"])
            if jid not in seen or hit["score"] > seen[jid]["score"]:
                seen[jid] = hit

        top = sorted(seen.values(), key=lambda h: h["score"], reverse=True)[:n_results]

        results = []
        for rank, hit in enumerate(top, 1):
            meta = hit["metadata"]
            results.append({
                "rank":        rank,
                "score":       round(hit["score"], 4),
                "title":       meta.get("title", ""),
                "company":     meta.get("company", ""),
                "role":        meta.get("role", ""),
                "seniority":   meta.get("seniority", ""),
                "location":    meta.get("location", ""),
                "skills_must": meta.get("skills_must", ""),
                "url":         meta.get("url", ""),
            })

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Semantic search error: {e}"


# ══════════════════════════════════════════════════════════════════════════
# Tool 3 – get_job_details
# ══════════════════════════════════════════════════════════════════════════

class GetJobDetailsInput(BaseModel):
    job_id: str = Field(
        ...,
        description="The job ID to retrieve full details for (from a previous search result)"
    )


def get_job_details(job_id: str) -> str:
    """Fetch the full record for a single job by its ID."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, company, role, seniority, location, url,
                       description, skills_must, skills_nice, yearsexperience,
                       past_experience, posted_at
                FROM jobs
                WHERE id = %s
            """, (job_id,))
            cols = [desc[0] for desc in cur.description]
            row  = cur.fetchone()

        if not row:
            return f"No job found with id '{job_id}'."

        record = dict(zip(cols, row))
        for k, v in record.items():
            if isinstance(v, (list, tuple)):
                record[k] = list(v)
            elif hasattr(v, "isoformat"):
                record[k] = v.isoformat()

        return json.dumps(record, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Database error: {e}"


# ══════════════════════════════════════════════════════════════════════════
# Tool definitions
# ══════════════════════════════════════════════════════════════════════════

filter_tool = StructuredTool.from_function(
    func=filter_jobs,
    name="filter_jobs",
    description=(
        "Filter jobs from PostgreSQL using structured parameters. "
        "Use this when the user specifies concrete criteria: role, seniority, "
        "location, years of experience, specific skills, or company name. "
        "All parameters are optional — only pass what the user mentioned."
    ),
    args_schema=FilterJobsInput,
)

semantic_tool = StructuredTool.from_function(
    func=semantic_search,
    name="semantic_search",
    description=(
        "Search jobs by semantic similarity using ChromaDB. "
        "Use this for free-text descriptions, vague queries, or when the user "
        "describes a role conceptually rather than with exact filters. "
        "e.g. 'startup-style fullstack role with ownership and fast pace'."
    ),
    args_schema=SemanticSearchInput,
)

details_tool = StructuredTool.from_function(
    func=get_job_details,
    name="get_job_details",
    description=(
        "Retrieve the full details of a specific job by its ID. "
        "Use this when the user wants to know more about a job that appeared "
        "in a previous filter_jobs or semantic_search result."
    ),
    args_schema=GetJobDetailsInput,
)

TOOLS = [filter_tool, semantic_tool, details_tool]


# ══════════════════════════════════════════════════════════════════════════
# Build agent
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a smart job-search assistant with access to a database of tech jobs scraped from LinkedIn Israel.

You have three tools:
- filter_jobs: use when the user gives concrete criteria (role, seniority, location, skills, years of experience)
- semantic_search: use when the user describes a job in free text or conceptual terms
- get_job_details: use when the user wants full details about a specific job from a previous result

Guidelines:
- You may combine tools in one turn (e.g. filter first, then semantic to enrich)
- Always summarise results in a readable way — don't dump raw JSON to the user
- If filter_jobs returns 0 results, try relaxing the filters or fall back to semantic_search
- When listing jobs, include: title, company, seniority, location, and key skills"""


def build_agent():
    llm = ChatGroq(
        api_key=GROQ_API_KEY_CHAT,
        model=GROQ_MODEL,
        temperature=0,
    )

    return create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )


# ══════════════════════════════════════════════════════════════════════════
# Helper — extract final text from LangGraph response
# ══════════════════════════════════════════════════════════════════════════

def _get_answer(result: dict) -> str:
    """Extract the last AI message content from a LangGraph result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            return content.strip()
    return "No answer returned."


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    agent = build_agent()

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print("\n" + "═" * 60)
        print("Answer:", _get_answer(result))
        return

    print("Job Search Agent — type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        if not user_input:
            continue

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            print("\nAgent:", _get_answer(result), "\n")
        except Exception as e:
            print(f"[Agent error] {e}\n")


if __name__ == "__main__":
    main()