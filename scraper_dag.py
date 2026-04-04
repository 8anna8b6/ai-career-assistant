from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.insert(0, '/app')
from main import main

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2026, 1, 1),  
}

dag = DAG(
    'job_scraper_daily',
    default_args=default_args,
    description='Run job scraper daily at 2 AM',
    schedule_interval='0 2 * * *',  # Daily at 2 AM UTC
    catchup=False,
    max_active_runs=1,
)

scraper_task = PythonOperator(
    task_id='run_scraper',
    python_callable=main,
    dag=dag,
)

scraper_task