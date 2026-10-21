from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pipeline.tasks import dq_snapshot, dq_report

with DAG('dataset_quality', start_date=datetime(2024, 1, 1), schedule='*/5 * * * *', catchup=False, max_active_runs=1, default_args={'retries': 1, 'retry_delay': timedelta(seconds=15)}, tags=['fraud', 'quality']) as dag:
    snapshot = PythonOperator(task_id='snapshot_dataset', python_callable=dq_snapshot, op_kwargs={'run_id': '{{ run_id }}'})
    report = PythonOperator(task_id='write_html_and_enforce_rules', python_callable=dq_report, op_kwargs={'run_id': '{{ run_id }}'})
    snapshot >> report
