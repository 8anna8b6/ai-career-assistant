"""
Shared pipeline steps used by both the Airflow DAG and the local runner.
"""
from __future__ import annotations

import logging
import time
from typing import List
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
import config
from db.chroma import get_existing_ids, init_chroma, upsert_jobs
from db.postgres import (
    count_jobs_today,
    fetch_all_ids,
    fetch_jobs_missing_from_chroma,
    get_connection,
    init_db,
    insert_jobs,
)
from pipeline.extractor import extract_with_groq
from pipeline.utils import fmt
from scraper.scraper import build_driver, scrape_keyword

log = logging.getLogger(__name__)


# ── Step 1: Scrape ────────────────────────────────────────────────────────────

def run_scrape(daily_target: int = config.DAILY_TARGET) -> List[dict]:
    """
    Scrape LinkedIn for new jobs up to `daily_target` per day.
    Returns a list of raw stub dicts (with raw_description + posted_at).
    """
    conn          = get_connection()
    init_db(conn)
    scraped_today = count_jobs_today(conn)
    remaining     = daily_target - scraped_today

    log.info("Today's progress: %d/%d jobs scraped.", scraped_today, daily_target)

    if remaining <= 0:
        log.info("Daily target already reached — nothing to scrape.")
        conn.close()
        return []

    seen_ids = fetch_all_ids(conn)
    conn.close()

    driver = build_driver()
    stubs  = []

    for keyword in config.KEYWORDS:
        if len(stubs) >= remaining:
            break

        try:
            for stub in scrape_keyword(driver, keyword, seen_ids, remaining - len(stubs)):
                if stub.get("posted_at"):
                    stub["posted_at"] = str(stub["posted_at"])
                stubs.append(stub)
                log.info("Scraped [%d/%d]: %s | %s", len(stubs), remaining,
                         stub["title"], stub["company"])

        except (InvalidSessionIdException, WebDriverException) as e:
            log.warning("Browser crashed (%s) — rebuilding driver...", e)
            try:
                driver.quit()
            except Exception:
                pass
            time.sleep(3)
            driver = build_driver()

        except Exception as e:
            log.error("Error scraping keyword '%s': %s", keyword, e)

    try:
        driver.quit()
    except Exception:
        pass

    log.info("Scrape complete — %d stubs collected.", len(stubs))
    return stubs


# ── Step 2: Extract ───────────────────────────────────────────────────────────

def run_extract(stubs: List[dict]) -> List[dict]:
    """
    Run Groq extraction on each stub.
    Returns a list of fully structured job dicts ready for storage.
    """
    if not stubs:
        log.info("No stubs to extract.")
        return []

    jobs = []
    for i, stub in enumerate(stubs, start=1):
        log.info("[%d/%d] Extracting: %s | %s", i, len(stubs), stub["title"], stub["company"])
        extracted = extract_with_groq(stub["title"], stub["raw_description"])

        job = {
            "id":              stub["id"],
            "title":           stub["title"],
            "company":         stub["company"],
            "location":        stub["location"],
            "url":             stub["url"],
            "role":            extracted["role"],
            "seniority":       extracted["seniority"],
            "description":     extracted["description"],
            "skills_must":     extracted["skills_must"],
            "skills_nice":     extracted["skills_nice"],
            "yearsexperience": extracted["yearsexperience"],
            "past_experience": extracted["past_experience"],
            "posted_at":       stub.get("posted_at"),
            "keyword":         stub["keyword"],
            "source":          "linkedin",
        }
        jobs.append(job)
        log.info("  ✓ %s | %s | exp: %s", job["role"], job["seniority"],
                 f"{job['yearsexperience']}yr" if job["yearsexperience"] else "?")

    log.info("Extraction complete — %d jobs processed.", len(jobs))
    return jobs


# ── Step 3a: Load PostgreSQL ──────────────────────────────────────────────────

def run_load_postgres(jobs: List[dict]) -> int:
    """Insert jobs into PostgreSQL. Returns number of rows inserted."""
    if not jobs:
        log.info("No jobs to insert into PostgreSQL.")
        return 0

    conn     = get_connection()
    init_db(conn)
    inserted = insert_jobs(conn, jobs)
    conn.close()
    return inserted


# ── Step 3b: Load ChromaDB ────────────────────────────────────────────────────

def run_load_chroma() -> int:
    """
    Backfill ChromaDB with any jobs that exist in PostgreSQL but are missing
    from the vector store. Returns number of vectors upserted.
    """
    collection    = init_chroma()
    conn          = get_connection()
    raw_ids       = get_existing_ids(collection)
    chroma_job_ids = {vid.rsplit("_", 1)[0] for vid in raw_ids}
    missing       = fetch_jobs_missing_from_chroma(conn, chroma_job_ids)
    conn.close()

    if not missing:
        log.info("ChromaDB is up to date — nothing to backfill.")
        return 0

    upserted = upsert_jobs(collection, missing)
    return upserted