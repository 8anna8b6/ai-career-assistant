from __future__ import annotations
import psycopg2
import psycopg2.extras
from typing import Dict, Any, List, Optional
from db import get_connection
from chroma_store import init_chroma, search_jobs as chroma_search


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

    return {
        "description": description,
        "columns": cols,
        "rows": [dict(r) for r in rows],
    }



def semantic_search_jobs(query: str, n_results: int = 5) -> Dict[str, Any]:
    collection = _collection()
    hits = chroma_search(collection, query, n_results=n_results)

    job_ids = [h["metadata"]["job_id"] for h in hits]

    conn = _conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, company, role, location, url, description
                FROM jobs WHERE id = ANY(%s)
                """,
                (job_ids,),
            )
            rows = {r["id"]: dict(r) for r in cur.fetchall()}
    finally:
        conn.close()

    results = [rows[jid] for jid in job_ids if jid in rows]

    return {"jobs": results}


def get_job_details(job_ids: List[str]) -> Dict[str, Any]:
    return _run_query(
        "SELECT * FROM jobs WHERE id = ANY(%s)",
        (job_ids,),
        "Fetch full job details",
    )


def search_jobs_by_criteria(
    role: Optional[str] = None,
    location: Optional[str] = None,
    company: Optional[str] = None,
    max_experience: Optional[int] = None,
    limit: int = 10,
):
    sql = "SELECT id, title, company, role, location, url FROM jobs"
    
    conditions = []
    params = []

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
    params.append(limit)

    return _run_query(sql, tuple(params), "Filtered job search")



def get_job_aggregate(
    operation: str, column: str, role_filter: Optional[str] = None
):
    """Calculates COUNT, AVG, MIN, or MAX on specific numeric/date columns."""
    ALLOWED_OPS = {"COUNT", "AVG", "MIN", "MAX"}
    ALLOWED_COLUMNS = {"yearsexperience", "posted_at", "scraped_at", "id"}

    op_upper = operation.upper()
    col_lower = column.lower()

    if op_upper not in ALLOWED_OPS:
        return {
            "error": f"Operation '{operation}' not allowed. Use COUNT, AVG, MIN, or MAX."
        }
    if col_lower not in ALLOWED_COLUMNS:
        return {"error": f"Column '{column}' not allowed for calculations."}

    sql = f"SELECT {op_upper}({col_lower}) AS result FROM jobs WHERE 1=1"
    params = []

    if role_filter:
        words = role_filter.split()
        or_conditions = []
        
        for word in words:
            or_conditions.append("LOWER(role) LIKE LOWER(%s)")
            params.append(f"%{word}%")
            
        
        if or_conditions:
            sql += f" AND ({' OR '.join(or_conditions)})"

    return _run_query(sql, tuple(params), f"Generic {op_upper} of {col_lower}")


def get_column_distribution(column: str, limit: int = 15):
    """Groups by a column and counts frequencies."""
    ALLOWED_COLUMNS = {
        "role",
        "seniority",
        "location",
        "company",
        "yearsexperience",
    }
    col_lower = column.lower()

    if col_lower not in ALLOWED_COLUMNS:
        return {"error": f"Cannot group by column '{column}'."}

    sql = f"""
        SELECT {col_lower} AS item, COUNT(*) AS count 
        FROM jobs 
        WHERE {col_lower} IS NOT NULL 
        GROUP BY {col_lower} 
        ORDER BY count DESC 
        LIMIT %s
    """

    return _run_query(sql, (limit,), f"Distribution of {col_lower}")


def top_skills(role: str, limit: int = 10):
    return _run_query(
        """
        SELECT unnest(skills_must) AS skill, COUNT(*) AS cnt
        FROM jobs
        WHERE LOWER(role) = LOWER(%s)
        GROUP BY skill
        ORDER BY cnt DESC
        LIMIT %s
        """,
        (role, limit),
        f"Top skills for {role}",
    )


def top_skills_all(limit: int = 15):
    return _run_query(
        """
        SELECT unnest(skills_must) AS skill, COUNT(*) AS cnt
        FROM jobs
        GROUP BY skill
        ORDER BY cnt DESC
        LIMIT %s
        """,
        (limit,),
        "Top skills overall",
    )


#mapping
TOOL_IMPLEMENTATIONS = {
    "semantic_search_jobs": semantic_search_jobs,
    "get_job_details": get_job_details,
    "search_jobs_by_criteria": search_jobs_by_criteria,
    "get_job_aggregate": get_job_aggregate,
    "get_column_distribution": get_column_distribution,
    "top_skills": top_skills,
    "top_skills_all": top_skills_all,
}


def run_tool(name: str, inputs: Dict[str, Any]):
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return fn(**inputs)