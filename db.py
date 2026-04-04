from __future__ import annotations
from typing import List
import psycopg2
from psycopg2.extras import execute_values
from config import DB_CONFIG

#conction
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

#DROP TABLE IF EXISTS jobs;
#build the schema
def init_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
           
            CREATE TABLE IF NOT EXISTS jobs (
                id                  TEXT PRIMARY KEY UNIQUE,
                title               TEXT,
                role                TEXT,
                seniority           TEXT,
                company             TEXT,
                location            TEXT,
                url                 TEXT,
                description         TEXT,
                skills_must         TEXT[],
                skills_nice         TEXT[],
                yearsexperience     INTEGER,
                past_experience     TEXT[],
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
            j.get("description"),
            j.get("skills_must", []), j.get("skills_nice", []),
            j.get("yearsexperience"),
            j.get("past_experience", []),
            j["keyword"], j.get("source", "linkedin"), j.get("posted_at"),
        )
        for j in jobs
    ]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO jobs (
                id, title, role, seniority, company, location, url,
                description,
                skills_must, skills_nice, yearsexperience, past_experience,
                keyword, source, posted_at
            )
            VALUES %s
            ON CONFLICT (id) DO NOTHING;
        """, rows)
    conn.commit()
    return len(rows)

def count_jobs(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs")
        return cur.fetchone()[0]


def count_jobs_today(conn) -> int:
    from datetime import date
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM jobs WHERE scraped_at::date = %s",
            (date.today(),)
        )
        return cur.fetchone()[0]


def fetch_all_ids(conn) -> set:#to prevent duplications
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM jobs;")
        return {row[0] for row in cur.fetchall()}


def fetch_jobs_without_embeddings(conn, chroma_ids: set) -> List[dict]:#used for back fill the vector db
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, role, seniority, company, location, url,
                   description,
                   skills_must, skills_nice, yearsexperience, past_experience,
                   keyword, source, posted_at
            FROM jobs;
        """)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    jobs = []
    for row in rows:
        job = dict(zip(cols, row))
        if job["id"] not in chroma_ids:
            jobs.append(job)
    return jobs

    