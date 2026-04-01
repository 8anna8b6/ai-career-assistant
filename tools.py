"""
1. Semantic search  (ChromaDB)  — "find jobs similar to X"
2. SQL analytics    (PostgreSQL) — "count / group / aggregate anything"
3. Job detail fetch (PostgreSQL) — "get full info on specific job(s)"
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional
import psycopg2
import psycopg2.extras
from db import get_connection
from chroma_store import init_chroma, search_jobs as chroma_search


_DB_SCHEMA = """
Table: jobs
Columns:
  id                TEXT        PRIMARY KEY
  title             TEXT        — job title as posted
  role              TEXT        — normalised role bucket:
                                  'Software Development','Frontend','Backend','Fullstack',
                                  'AI / ML','Data Scientist','Data Engineer','Data Analyst',
                                  'BI','DevOps / Cloud','Mobile','QA / Automation','Security',
                                  'Embedded / Firmware','Database','Network','System Engineer',
                                  'Product Manager','Team Lead','R&D','Solutions Architect','Other'
  seniority         TEXT        — 'Intern','Junior','Mid','Senior','Lead','Staff',
                                  'Principal','Manager','Director','VP','Not specified'
  company           TEXT        — company name
  location          TEXT        — city / region
  url               TEXT        — original job posting URL
  description       TEXT        — 4-5 sentence AI-generated role summary
  skills_must       TEXT[]      — required skills (array)
  skills_nice       TEXT[]      — nice-to-have skills (array)
  yearsexperience   INTEGER     — years of experience required (may be NULL)
  past_experience   TEXT[]      — required background domains (array)
  keyword           TEXT        — search keyword used to find this job
  source            TEXT        — always 'linkedin'
  posted_at         DATE        — when the job was posted
  scraped_at        TIMESTAMP   — when we scraped it

Useful patterns:
  — unnest arrays:        SELECT unnest(skills_must) AS skill FROM jobs
  — array contains:       skills_must @> ARRAY['Python']
  — array overlap:        skills_must && ARRAY['Python','Go']
  — case-insensitive:     LOWER(role) = LOWER('backend')
  — partial text search:  title ILIKE '%engineer%'
  — recent jobs:          ORDER BY scraped_at DESC / posted_at DESC
  — skill frequency:      SELECT unnest(skills_must) AS skill, COUNT(*) AS cnt
                          FROM jobs GROUP BY skill ORDER BY cnt DESC
  — combine arrays:       SELECT unnest(skills_must || skills_nice) AS skill ...
"""

TOOL_DEFINITIONS: List[Dict] = [
    # Semantic search
    {
        "name": "semantic_search_jobs",
        "description": (
            "Search job postings by meaning using vector similarity. "
            "Use this when the user describes what they are looking for in natural language "
            "— e.g. 'jobs that involve building APIs', 'machine learning roles at startups', "
            "'jobs for someone who knows React and Node'. "
            "Also use to find jobs similar to a CV description or skill list. "
            "Returns ranked job listings with titles, companies, skills and descriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the job to search for. "
                        "Be specific — include skills, role type, technologies, etc."
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": "How many results to return. Default 5, max 20.",
                    "default": 5,
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional metadata filters applied before ranking. Supported keys: "
                        "'role' (exact string from the role enum), "
                        "'seniority' (exact string from the seniority enum), "
                        "'company' (exact company name). "
                        "Example: {\"role\": \"Backend\", \"seniority\": \"Senior\"}"
                    ),
                },
            },
            "required": ["query"],
        },
    },

    #SQL analytics
    {
        "name": "query_jobs_database",
        "description": (
            "Run a read-only SQL SELECT query against the jobs PostgreSQL database. "
            "Use this for ANY question that requires counting, grouping, averaging, ranking, "
            "filtering by structured fields, or computing statistics.\n\n"
            "Examples of what this covers:\n"
            "- 'How many Backend jobs are there?'\n"
            "- 'What is the average years of experience required for Data Scientists?'\n"
            "- 'Which are the top 10 most required skills for AI / ML roles?'\n"
            "- 'How many Junior vs Senior positions exist?'\n"
            "- 'Which companies post the most jobs?'\n"
            "- 'What roles are most common in Tel Aviv?'\n"
            "- 'Show me the last 10 jobs posted'\n"
            "- 'What percentage of jobs require Python?'\n"
            "- 'What is the seniority distribution for Backend engineers?'\n"
            "- 'Which skills appear most in both required and nice-to-have across all jobs?'\n\n"
            "Write standard PostgreSQL. Only SELECT queries are permitted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        f"A valid PostgreSQL SELECT query against the jobs table.\n\n"
                        f"Schema reference:\n{_DB_SCHEMA}\n\n"
                        "Rules:\n"
                        "- Only SELECT or WITH...SELECT is allowed\n"
                        "- No INSERT / UPDATE / DELETE / DROP\n"
                        "- Always add LIMIT (max 200) to avoid huge results\n"
                        "- Use LOWER() for case-insensitive text comparisons\n"
                        "- Use unnest() to work with array columns like skills_must"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "One sentence describing what this query computes (for logging).",
                },
            },
            "required": ["sql", "description"],
        },
    },

    #Job detail
    {
        "name": "get_job_details",
        "description": (
            "Fetch the complete record for one or more specific jobs by their IDs. "
            "Use this after semantic_search_jobs or query_jobs_database returns job IDs "
            "and the user wants the full description, all skills, or the URL for a specific job."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of job IDs to fetch. IDs come from other tool results.",
                },
            },
            "required": ["job_ids"],
        },
    },
]


def _conn():
    return get_connection()


def _collection():
    return init_chroma()


# implementation of Semantic search
def semantic_search_jobs(
    query: str,
    n_results: int = 5,
    filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    n_results = min(n_results, 20)

    # Build ChromaDB where clause from filters
    where: Optional[Dict] = None
    if filters:
        conditions = [
            {k: {"$eq": v}}
            for k, v in filters.items()
            if v
        ]
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

    collection = _collection()
    hits = chroma_search(collection, query, n_results=n_results * 3, where=where)

    # Deduplicate by job_id (multiple vectors per job)
    seen: set = set()
    unique_hits = []
    for h in hits:
        jid = h["metadata"].get("job_id", h["id"])
        if jid not in seen:
            seen.add(jid)
            unique_hits.append(h)
        if len(unique_hits) >= n_results:
            break

    if not unique_hits:
        return {"jobs": [], "total": 0, "message": "No matching jobs found."}

    job_ids = [h["metadata"].get("job_id") for h in unique_hits]
    score_map = {h["metadata"].get("job_id"): round(h["score"], 3) for h in unique_hits}

    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, role, seniority, company, location, url,
                       description, skills_must, skills_nice, yearsexperience,
                       past_experience, posted_at
                FROM jobs WHERE id = ANY(%s)
                """,
                (job_ids,),
            )
            rows = {r["id"]: dict(r) for r in cur.fetchall()}
    finally:
        conn.close()

    results = []
    for jid in job_ids:
        if jid in rows:
            job = rows[jid]
            job["relevance_score"] = score_map.get(jid, 0)
            results.append(job)

    return {"jobs": results, "total": len(results)}


#implementaation of SQL analytics

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|replace|grant|revoke|copy|pg_)\b",
    re.IGNORECASE,
)


def query_jobs_database(sql: str, description: str = "") -> Dict[str, Any]:
    sql = sql.strip().rstrip(";")

    # only allow select for sefty
    if _FORBIDDEN.search(sql):
        return {"error": "Only SELECT queries are permitted.", "sql": sql}
    if not re.match(r"^\s*(with\s+|select\s)", sql, re.IGNORECASE):
        return {"error": "Query must start with SELECT or WITH.", "sql": sql}

    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            truncated = len(rows) > 200
            rows = rows[:200]
            columns = [desc[0] for desc in cur.description] if cur.description else []
    except Exception as e:
        conn.rollback()
        return {"error": str(e), "sql": sql}
    finally:
        conn.close()

    return {
        "columns": columns,
        "rows": [dict(r) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
        "description": description,
    }


#implementaation of Job detail fetch

def get_job_details(job_ids: List[str]) -> Dict[str, Any]:
    if not job_ids:
        return {"jobs": [], "total": 0}

    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, role, seniority, company, location, url,
                       description, skills_must, skills_nice, yearsexperience,
                       past_experience, posted_at, scraped_at
                FROM jobs WHERE id = ANY(%s)
                """,
                (job_ids,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {"jobs": rows, "total": len(rows)}



TOOL_IMPLEMENTATIONS = {
    "semantic_search_jobs": semantic_search_jobs,
    "query_jobs_database":  query_jobs_database,
    "get_job_details":      get_job_details,
}


def run_tool(name: str, inputs: Dict[str, Any]) -> Any:
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if not fn:
        return {"error": f"Unknown tool: '{name}'"}
    try:
        return fn(**inputs)
    except Exception as e:
        return {"error": str(e), "tool": name}