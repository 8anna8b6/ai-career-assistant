from __future__ import annotations
import psycopg2
import psycopg2.extras
from typing import Optional
from langchain.tools import tool

from db import get_connection
from db.chroma import init_chroma, search_jobs as chroma_search


# ── helpers ───────────────────────────────────────────────────────────────

def _conn():
    return get_connection()

def _collection():
    return init_chroma()

def _run_query(sql: str, params: tuple = (), description: str = ""):
    """Safely runs SQL queries using tuple parameters to prevent SQL injection."""
    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
    finally:
        conn.close()
    return {"description": description, "columns": cols, "rows": [dict(r) for r in rows]}


# ── tools ─────────────────────────────────────────────────────────────────

@tool
def semantic_search_jobs(query: str, n_results: int = 5):
    """Search jobs using semantic similarity (vector search).
    Use for: finding jobs by natural-language description, skill match, or profile.
    Examples: 'find Python backend jobs', 'jobs at fintech startups'
    """
    collection = _collection()
    hits = chroma_search(collection, query, n_results=n_results)
    job_ids = [h["metadata"]["job_id"] for h in hits]

    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, company, role, location, url, description FROM jobs WHERE id = ANY(%s)",
                (job_ids,),
            )
            rows = {r["id"]: dict(r) for r in cur.fetchall()}
    finally:
        conn.close()

    return {"jobs": [rows[jid] for jid in job_ids if jid in rows]}


@tool
def get_job_details(job_ids: list[str]):
    """Get full job details for given job IDs.
    Use after search results return IDs the user wants to know more about.
    """
    return _run_query(
        "SELECT * FROM jobs WHERE id = ANY(%s)",
        (job_ids,),
        "Fetch full job details",
    )


@tool
def get_job_aggregate(operation: str, column: str, role_filter: str = None):
    """Calculate COUNT, AVG, MIN, or MAX statistics on job data.
    - operation: one of ['COUNT', 'AVG', 'MIN', 'MAX']
    - column: one of ['yearsexperience', 'posted_at']
    - role_filter: optional role name to filter by (e.g. 'data scientist')
    Examples: average experience for Data Scientists, count of all jobs
    """
    ALLOWED_OPS = {"COUNT", "AVG", "MIN", "MAX"}
    ALLOWED_COLUMNS = {"yearsexperience", "posted_at", "scraped_at", "id"}

    op_upper = operation.upper()
    col_lower = column.lower()

    if op_upper not in ALLOWED_OPS:
        return {"error": f"Operation '{operation}' not allowed. Use COUNT, AVG, MIN, or MAX."}
    if col_lower not in ALLOWED_COLUMNS:
        return {"error": f"Column '{column}' not allowed for calculations."}

    sql = f"SELECT {op_upper}({col_lower}) AS result FROM jobs WHERE 1=1"
    params = []

    if role_filter:
        words = role_filter.split()
        or_conditions = [f"LOWER(role) LIKE LOWER(%s)" for _ in words]
        params.extend(f"%{w}%" for w in words)
        sql += f" AND ({' OR '.join(or_conditions)})"

    return _run_query(sql, tuple(params), f"{op_upper} of {col_lower}")


@tool
def get_column_distribution(column: str, limit: int = 15):
    """Get top items and their counts for a given column — useful for breakdowns.
    - column: one of ['role', 'seniority', 'location', 'company', 'yearsexperience']
    Examples: top companies, seniority breakdown, most common roles
    """
    ALLOWED_COLUMNS = {"role", "seniority", "location", "company", "yearsexperience"}
    col_lower = column.lower()

    if col_lower not in ALLOWED_COLUMNS:
        return {"error": f"Cannot group by column '{column}'."}

    return _run_query(
        f"""
        SELECT {col_lower} AS item, COUNT(*) AS count
        FROM jobs WHERE {col_lower} IS NOT NULL
        GROUP BY {col_lower} ORDER BY count DESC LIMIT %s
        """,
        (limit,),
        f"Distribution of {col_lower}",
    )


@tool
def search_jobs_by_criteria(
    role: str = None,
    location: str = None,
    company: str = None,
    max_experience: int = None,
):
    """Filter jobs by specific fields.
    - role: e.g. 'backend', 'react developer'
    - location: e.g. 'Tel Aviv', 'remote'
    - company: e.g. 'Google', 'startup'
    - max_experience: maximum years of experience required
    """
    sql = "SELECT id, title, company, role, location, url FROM jobs"
    conditions, params = [], []

    if role:
        conditions.append("LOWER(role) LIKE LOWER(%s)")
        params.append(f"%{role}%")
    if location:
        conditions.append("LOWER(location) LIKE LOWER(%s)")
        params.append(f"%{location}%")
    if company:
        conditions.append("LOWER(company) LIKE LOWER(%s)")
        params.append(f"%{company}%")
    if max_experience is not None:
        conditions.append("yearsexperience <= %s")
        params.append(max_experience)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY scraped_at DESC LIMIT %s"
    params.append(10)

    return _run_query(sql, tuple(params), "Filtered job search")


@tool
def top_skills(role: str, limit: int = 10):
    """Get the most required skills for a specific role.
    Examples: top skills for DevOps, most needed skills for Data Scientists
    """
    return _run_query(
        """
        SELECT unnest(skills_must) AS skill, COUNT(*) AS cnt
        FROM jobs WHERE LOWER(role) = LOWER(%s)
        GROUP BY skill ORDER BY cnt DESC LIMIT %s
        """,
        (role, limit),
        f"Top skills for {role}",
    )


@tool
def top_skills_all(limit: int = 15):
    """Get the most required skills across all jobs in the database.
    Use when the user asks about the overall market or doesn't specify a role.
    """
    return _run_query(
        """
        SELECT unnest(skills_must) AS skill, COUNT(*) AS cnt
        FROM jobs GROUP BY skill ORDER BY cnt DESC LIMIT %s
        """,
        (limit,),
        "Top skills overall",
    )


# ── exported list ─────────────────────────────────────────────────────────
# This is the only thing agent.py needs to import

TOOLS = [
    semantic_search_jobs,
    get_job_details,
    get_job_aggregate,
    get_column_distribution,
    search_jobs_by_criteria,
    top_skills,
    top_skills_all,
]