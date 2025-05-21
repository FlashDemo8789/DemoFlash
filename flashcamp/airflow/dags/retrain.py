from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    "camp_retrain_quarterly",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 1 */3 *",   # 1st day each quarter at 02:00
    catchup=False,
    tags=["ml"],
) as dag:
    validate = BashOperator(
        task_id="validate",
        bash_command="python notebooks/01_validate_csv.py "
                     "--in data/new_batch.csv "
                     "--schema backend/schema/camp_schema.yaml "
                     "--out data/gold/latest.parquet"
    )

    train = BashOperator(
        task_id="train",
        bash_command="python pipelines/train_baseline.py "
                     "--data data/gold/latest.parquet "
                     "--models models/"
    )

    shap = BashOperator(
        task_id="shap",
        bash_command="python pipelines/gen_shap.py "
                     "--data data/gold/latest.parquet "
                     "--models models/ "
                     "--out reports/assets/"
    )

    validate >> train >> shap 