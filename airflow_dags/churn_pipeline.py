from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess


# Function to run python scripts
def run_script(script_path):
    subprocess.run(["python", script_path], check=True)


default_args = {
    "owner": "mlops_project",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


with DAG(
    dag_id="customer_churn_ml_pipeline",
    default_args=default_args,
    description="End-to-End Customer Churn MLOps Pipeline",
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "churn"],
) as dag:

    # Task 1 - Data Ingestion
    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_script,
        op_args=["src/data_ingestion.py"],
    )

    # Task 2 - Data Validation (basic check)
    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=run_script,
        op_args=["src/data_ingestion.py"],
    )

    # Task 3 - Feature Engineering / Preprocessing
    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=run_script,
        op_args=["src/preprocessing.py"],
    )

    # Task 4 - Model Training
    model_training = PythonOperator(
        task_id="model_training",
        python_callable=run_script,
        op_args=["src/train.py"],
    )

    # Task 5 - Model Evaluation
    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_script,
        op_args=["src/evaluate.py"],
    )

    # Task 6 - Model Registration
    model_registration = PythonOperator(
        task_id="model_registration",
        python_callable=run_script,
        op_args=["src/evaluate.py"],
    )

    # Define task dependencies
    data_ingestion >> data_validation >> preprocessing >> model_training >> model_evaluation >> model_registration