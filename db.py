from __future__ import annotations
from typing import List
import psycopg2
from psycopg2.extras import execute_values
from config import DB_CONFIG

#conction
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


#build the schema
def init_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS test2;
            CREATE TABLE IF NOT EXISTS test2 (
                id                  TEXT PRIMARY KEY UNIQUE,
                title               TEXT,
                role                TEXT,
                seniority           TEXT,
                company             TEXT,
                location            TEXT,
                url                 TEXT,
                description         TEXT,
                requirements        TEXT,
                experience          INTEGER,
                skills_must         TEXT[],
                skills_nice         TEXT[],
                past_experience     TEXT[],
                tools_technologies  TEXT[],
                keyword             TEXT,
                source              TEXT DEFAULT 'linkedin',
                posted_at           DATE,
                scraped_at          TIMESTAMP DEFAULT NOW()
            );
        """)
    conn.commit()


#inserts job to table 

def insert_jobs(conn, jobs: List[dict]) -> int:
    if not jobs:
        return 0

    rows = [
        (
            j["id"], j["title"], j.get("role"), j.get("seniority"),
            j["company"], j["location"], j["url"],
            j.get("description"), j.get("requirements"), j.get("experience"),
            j.get("skills_must", []), j.get("skills_nice", []),
            j.get("past_experience", []), j.get("tools_technologies", []),
            j["keyword"], j.get("source", "linkedin"), j.get("posted_at"),
        )
        for j in jobs
    ]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO test2 (
                id, title, role, seniority, company, location, url,
                description, requirements, experience,
                skills_must, skills_nice, past_experience, tools_technologies,
                keyword, source, posted_at
            )
            VALUES %s
            ON CONFLICT (id) DO NOTHING;
        """, rows)
    conn.commit()
    return len(rows)


def count_jobs(conn) -> int:#for debuf delete later
   
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM test2;")
        return cur.fetchone()[0]


def fetch_all_ids(conn) -> set:#to prevent duplications
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM test2;")
        return {row[0] for row in cur.fetchall()}


def fetch_jobs_without_embeddings(conn, chroma_ids: set) -> List[dict]:#used for back fill the vector db
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, role, seniority, company, location, url,
                   description, requirements, experience,
                   skills_must, skills_nice, past_experience, tools_technologies,
                   keyword, source, posted_at
            FROM test2;
        """)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    jobs = []
    for row in rows:
        job = dict(zip(cols, row))
        if job["id"] not in chroma_ids:
            jobs.append(job)
    return jobs