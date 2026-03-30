from __future__ import annotations
import sys
import time
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
import config
from utils import fmt
from extractor import extract_with_groq
from scraper import build_driver, is_driver_alive, scrape_keyword
from db import get_connection, init_db, insert_jobs, count_jobs, fetch_all_ids, fetch_jobs_without_embeddings
from chroma_store import init_chroma, upsert_jobs, collection_count, get_existing_ids

def main() -> None:
    run_start = time.time()  # just to see how time it takes
    conn = get_connection()  # conect to posgrase
    init_db(conn)  # initialize the postgrasql db
    collection = init_chroma()

    # Backfill ChromaDB from PostgreSQL
    chroma_ids = get_existing_ids(collection)
    jobs_to_embed = fetch_jobs_without_embeddings(conn, chroma_ids)
    print(f"Backfilling {len(jobs_to_embed)} jobs into Chroma...")
    for job in jobs_to_embed:
        upsert_jobs(collection, [job])

    seen_ids = fetch_all_ids(conn)  # all jobs you we processed(prevent duplications)
    driver = build_driver()  # start Selenium driver

    # scraping loop
    for keyword in config.KEYWORDS:
        total_pg = count_jobs(conn)  # just to show progress delete later
        print(f"\nDB: {total_pg} jobs | Chroma: {collection_count(collection)} vectors"
              f" | elapsed: {fmt(time.time() - run_start)}")

        if total_pg >= config.TARGET_JOBS:  # just for test- limited amount of job post
            print(f"Reached {config.TARGET_JOBS} jobs. Done!")
            break

        remaining = config.TARGET_JOBS - total_pg

        try:
            raw_jobs = scrape_keyword(driver, keyword, seen_ids, remaining)  # scraping
        except (InvalidSessionIdException, WebDriverException) as e:  # if the driver cruch close it ,rebuild
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

        if not raw_jobs:
            continue

        for stub in raw_jobs:
            print(f"  [Groq] Extracting: {stub['title']} | {stub['company']}")
            extracted = extract_with_groq(stub["title"], stub["raw_description"])  # extract using groq
            time.sleep(2)

            posted_str = str(stub["posted_at"]) if stub.get("posted_at") else "unknown"
            exp_str = f"{extracted['yearsexperience']}yr" if extracted.get("yearsexperience") else "exp:?"
            print(
                f"    ✓ {stub['title']} | {extracted['role']} | {extracted['seniority']}"
                f" | {exp_str} | posted: {posted_str}"
            )

            # Build the complete job data
            job = {
                "id": stub["id"],
                "title": stub["title"],
                "company": stub["company"],
                "location": stub["location"],
                "url": stub["url"],
                "role": extracted["role"],
                "seniority": extracted["seniority"],
                "description": extracted["description"],
                "skills_must": extracted["skills_must"],
                "skills_nice": extracted["skills_nice"],
                "yearsexperience": extracted["yearsexperience"],
                "past_experience": extracted["past_experience"],
                "posted_at": stub["posted_at"],
                "keyword": stub["keyword"],
                "source": "linkedin",
            }

            # insert the data into postgraseSQL db
            pg_inserted = insert_jobs(conn, [job])
            print(f"PostgreSQL: {pg_inserted} inserted")

            # insert the data into chromadb
            chroma_upserted = upsert_jobs(collection, [job])
            print(f"ChromaDB:   {chroma_upserted} upserted "
                  f"(total: {collection_count(collection)})")

    # close the driver and the db
    try:
        driver.quit()
    except Exception:
        pass
    conn.close()
    print(f"\nDone in {fmt(time.time() - run_start)}.")


if __name__ == "__main__":
    main()