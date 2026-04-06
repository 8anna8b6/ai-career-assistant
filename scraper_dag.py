from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── DAG config ────────────────────────────────────────────────────────────────

default_args = {
    "owner":"airflow",
    "retries": 1,
    "retry_delay":timedelta(minutes=5),
    "start_date":datetime(2026, 1, 1),
}

dag = DAG(
    "job_scraper_daily",
    default_args=default_args,
    description="Scrape, extract, and load jobs daily",
    schedule_interval="10 11 * * *",  #time
    catchup=False,
    max_active_runs=1,
)

# Temp directory for passing data between tasks via JSON files
TEMP_DIR = "/opt/career_assistant/tmp"


# ── File helpers ──────────────────────────────────────────────────────────────

def _run_file(context: dict, name: str) -> str:
    """Return a unique temp file path scoped to this DAG run."""
    run_id = context["run_id"].replace(":", "_").replace("+", "_")
    os.makedirs(TEMP_DIR, exist_ok=True)
    return os.path.join(TEMP_DIR, f"{run_id}_{name}.json")


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Tasks ─────────────────────────────────────────────────────────────────────

def task_scrape(**context):
    from pipeline.core import run_scrape

    stubs      = run_scrape()
    stubs_file = _run_file(context, "stubs")
    _write_json(stubs_file, stubs)
    context["ti"].xcom_push(key="stubs_file", value=stubs_file)
    log.info("Scraped %d stubs → %s", len(stubs), stubs_file)


def task_extract(**context):
    from pipeline.core import run_extract

    stubs_file = context["ti"].xcom_pull(key="stubs_file", task_ids="scrape")
    stubs      = _read_json(stubs_file)
    jobs       = run_extract(stubs)
    jobs_file  = _run_file(context, "jobs")
    _write_json(jobs_file, jobs)
    context["ti"].xcom_push(key="jobs_file", value=jobs_file)
    log.info("Extracted %d jobs → %s", len(jobs), jobs_file)


def task_load_postgres(**context):
    from pipeline.core import run_load_postgres

    jobs_file = context["ti"].xcom_pull(key="jobs_file", task_ids="extract")
    jobs      = _read_json(jobs_file)
    inserted  = run_load_postgres(jobs)
    log.info("PostgreSQL: %d jobs inserted.", inserted)


def task_load_chroma(**context):
    from pipeline.core import run_load_chroma

    upserted = run_load_chroma()
    log.info("ChromaDB: %d vectors upserted.", upserted)


def task_cleanup(**context):
    """Remove temp files created during this run."""
    for name in ("stubs", "jobs"):
        path = _run_file(context, name)
        try:
            os.remove(path)
            log.info("Deleted temp file: %s", path)
        except FileNotFoundError:
            pass


#DAG wiring

t1 = PythonOperator(task_id="scrape",        python_callable=task_scrape,        provide_context=True, dag=dag)
t2 = PythonOperator(task_id="extract",       python_callable=task_extract,       provide_context=True, dag=dag)
t3 = PythonOperator(task_id="load_postgres", python_callable=task_load_postgres, provide_context=True, dag=dag)
t4 = PythonOperator(task_id="load_chroma",  python_callable=task_load_chroma,   provide_context=True, dag=dag)
t5 = PythonOperator(task_id="cleanup",       python_callable=task_cleanup,       provide_context=True, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5