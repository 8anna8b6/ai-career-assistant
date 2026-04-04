from __future__ import annotations
import sys
import time
from datetime import date
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
import config
from utils import fmt
from extractor import extract_with_groq
from scraper import build_driver, scrape_keyword
from db import get_connection, init_db, insert_jobs, count_jobs, fetch_all_ids, fetch_jobs_without_embeddings, count_jobs_today
from chroma_store import init_chroma, upsert_jobs, collection_count, get_existing_ids

DAILY_TARGET = 30


def main() -> None:
    run_start  = time.time()
    conn       = get_connection()
    init_db(conn)
    collection = init_chroma()

    # Backfill ChromaDB from PostgreSQL
    raw_chroma_ids = get_existing_ids(collection)
    chroma_job_ids = {vid.rsplit("_", 1)[0] for vid in raw_chroma_ids}
    jobs_to_embed  = fetch_jobs_without_embeddings(conn, chroma_job_ids)
    print(f"Chroma has {len(chroma_job_ids)} unique jobs | Backfilling {len(jobs_to_embed)} missing...")
    for job in jobs_to_embed:
        upsert_jobs(collection, [job])

    # How many have we already scraped today?
    scraped_today = count_jobs_today(conn)
    remaining_today = DAILY_TARGET - scraped_today
    print(f"\nToday's progress: {scraped_today}/{DAILY_TARGET} jobs scraped today")

    if remaining_today <= 0:
        print(f"Already scraped {DAILY_TARGET} jobs today. Done!")
        conn.close()
        return

    seen_ids = fetch_all_ids(conn)
    driver   = build_driver()
    jobs_inserted_this_run = 0

    for keyword in config.KEYWORDS:
        if jobs_inserted_this_run >= remaining_today:
            print(f"Reached daily target of {DAILY_TARGET} jobs. Done!")
            break

        still_needed = remaining_today - jobs_inserted_this_run
        print(f"\nKeyword: '{keyword}' | Need {still_needed} more jobs today"
              f" | elapsed: {fmt(time.time() - run_start)}")

        try:
            for stub in scrape_keyword(driver, keyword, seen_ids, still_needed):

                print(f"  [Groq] Extracting: {stub['title']} | {stub['company']}")
                extracted = extract_with_groq(stub["title"], stub["raw_description"])

                posted_str = str(stub["posted_at"]) if stub.get("posted_at") else "unknown"
                exp_str    = f"{extracted['yearsexperience']}yr" if extracted.get("yearsexperience") else "exp:?"
                print(
                    f"    ✓ {stub['title']} | {extracted['role']} | {extracted['seniority']}"
                    f" | {exp_str} | posted: {posted_str}"
                )

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
                    "posted_at":       stub["posted_at"],
                    "keyword":         stub["keyword"],
                    "source":          "linkedin",
                }

                pg_inserted     = insert_jobs(conn, [job])
                chroma_upserted = upsert_jobs(collection, [job])
                jobs_inserted_this_run += pg_inserted
                print(f"    PostgreSQL: {pg_inserted} inserted | "
                      f"ChromaDB: {chroma_upserted} upserted | "
                      f"Today: {scraped_today + jobs_inserted_this_run}/{DAILY_TARGET}")

                if jobs_inserted_this_run >= remaining_today:
                    print(f"Reached daily target of {DAILY_TARGET} jobs. Done!")
                    break

        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"\nBrowser crashed ({e}). Rebuilding driver...")
            try:
                driver.quit()
            except Exception:
                pass
            time.sleep(3)
            driver = build_driver()
            continue

        except Exception as e:
            print(f"Error scraping '{keyword}': {e}")
            continue

    # Cleanup
    try:
        driver.quit()
    except Exception:
        pass
    conn.close()
    print(f"\nDone in {fmt(time.time() - run_start)}. "
          f"Inserted {jobs_inserted_this_run} jobs this run.")


if __name__ == "__main__":
    main()