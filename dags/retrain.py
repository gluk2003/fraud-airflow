"""Graph only; task implementations live in pipeline.tasks."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pipeline.tasks import prepare_dataset, train_model, evaluate_model, publish_model

with DAG('retrain_fraud_model', start_date=datetime(2024, 1, 1), schedule='*/5 * * * *', catchup=False, max_active_runs=1, default_args={'retries': 2, 'retry_delay': timedelta(seconds=15)}, tags=['fraud', 'training']) as dag:
    prepare = PythonOperator(task_id='prepare_temporal_window', python_callable=prepare_dataset, op_kwargs={'run_id': '{{ run_id }}'})
    train = PythonOperator(task_id='train_candidate', python_callable=train_model, op_kwargs={'run_id': '{{ run_id }}'})
    evaluate = PythonOperator(task_id='compare_on_same_holdout', python_callable=evaluate_model, op_kwargs={'run_id': '{{ run_id }}'})
    publish = PythonOperator(task_id='publish_if_not_worse', python_callable=publish_model, op_kwargs={'run_id': '{{ run_id }}'})
    prepare >> train >> evaluate >> publish
