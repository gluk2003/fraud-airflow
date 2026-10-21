from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pipeline.tasks import supply_month

with DAG('supply_labeled_month', start_date=datetime(2024, 1, 1), schedule='*/5 * * * *', catchup=False, max_active_runs=1, default_args={'retries': 2, 'retry_delay': timedelta(seconds=15)}, tags=['fraud', 'data']) as dag:
    supply = PythonOperator(task_id='append_next_month_and_prune', python_callable=supply_month, op_kwargs={'run_id': '{{ run_id }}'})
