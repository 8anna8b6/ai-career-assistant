"""
Local debug entry point — runs the full pipeline end-to-end without Airflow.
neet to set DB_HOST=localhost in  .env when running outside Docker.
"""
from __future__ import annotations

import logging
import time

from pipeline.core import run_extract, run_load_chroma, run_load_postgres, run_scrape
from pipeline.utils import fmt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    start = time.time()
    log.info("=== Pipeline starting (local runner) ===")

    # Step 1 — Scrape
    stubs = run_scrape()

    # Step 2 — Extract
    jobs = run_extract(stubs)

    # Step 3a — Load PostgreSQL
    inserted = run_load_postgres(jobs)
    log.info("PostgreSQL: %d jobs inserted.", inserted)

    # Step 3b — Load ChromaDB (also backfills any missing vectors)
    upserted = run_load_chroma()
    log.info("ChromaDB: %d vectors upserted.", upserted)

    log.info("=== Pipeline done in %s ===", fmt(time.time() - start))


if __name__ == "__main__":
    main()