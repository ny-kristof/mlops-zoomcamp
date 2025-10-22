#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

# Airflow imports (TaskFlow API)
try:
    # Airflow 3.x uses airflow.sdk
    from airflow.sdk import dag as dag_decorator, task, DAG
    from airflow.sdk.definitions.context import get_current_context
    from airflow.models import Param
    AIRFLOW_AVAILABLE = True
except Exception:
    # Allow running this file as a plain script without Airflow installed
    dag_decorator = None
    task = None
    get_current_context = None
    Param = None
    AIRFLOW_AVAILABLE = False

# Resolve directories relative to this file to be safe in Airflow workers
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

# MLflow configuration (can be overridden by env vars)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-experiment")



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    # Configure MLflow at runtime (parse-safe for Airflow)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open(MODELS_DIR / "preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(MODELS_DIR / "preprocessor.b"), artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


# ---------- Airflow DAG (TaskFlow) ----------
def _next_year_month(year: int, month: int):
    """Return the next (year, month)."""
    if month < 12:
        return year, month + 1
    return year + 1, 1


if AIRFLOW_AVAILABLE:
    @dag_decorator(
        dag_id="nyc_taxi_duration_prediction",
        description="Train XGBoost model to predict NYC taxi trip duration with MLflow logging",
        start_date=datetime(2023, 1, 1),
        schedule=None,  # Trigger manually; you can set to "@monthly" later if desired
        catchup=False,
        default_args={"owner": "airflow"},
        params={
            "year": Param(2023, type="integer", minimum=2018, maximum=2030),
            "month": Param(1, type="integer", minimum=1, maximum=12),
        },
        tags=["mlops-zoomcamp", "xgboost", "mlflow", "nyc-taxi"],
    )
    def duration_prediction_dag():
        @task
        def fetch_train() -> str:
            """Download and prepare train dataframe, save locally, return file path."""
            ctx = get_current_context()
            year = int(ctx["params"]["year"])  # type: ignore[index]
            month = int(ctx["params"]["month"])  # type: ignore[index]
            df_train = read_dataframe(year=year, month=month)
            out_path = DATA_DIR / f"train_{year}-{month:02d}.parquet"
            df_train.to_parquet(out_path, index=False)
            return str(out_path)

        @task
        def fetch_val() -> str:
            """Download and prepare validation dataframe for next month, save locally, return file path."""
            ctx = get_current_context()
            year = int(ctx["params"]["year"])  # type: ignore[index]
            month = int(ctx["params"]["month"])  # type: ignore[index]
            ny, nm = _next_year_month(year, month)
            df_val = read_dataframe(year=ny, month=nm)
            out_path = DATA_DIR / f"val_{ny}-{nm:02d}.parquet"
            df_val.to_parquet(out_path, index=False)
            return str(out_path)

        @task
        def train(train_path: str, val_path: str) -> str:
            """Train the model using saved datasets; returns MLflow run_id."""
            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)

            X_train, dv = create_X(df_train)
            X_val, _ = create_X(df_val, dv)

            target = 'duration'
            y_train = df_train[target].values
            y_val = df_val[target].values

            run_id = train_model(X_train, y_train, X_val, y_val, dv)

            # Save run_id to file for downstream use
            with open(BASE_DIR / "run_id.txt", "w") as f:
                f.write(run_id)

            return run_id

        # Define task graph with sequential execution
        train_path = fetch_train()
        val_path = fetch_val()
        run_id = train(train_path, val_path)
        
        # Define task dependencies - sequential chain
        train_path >> val_path >> run_id

    # Instantiate the DAG - this makes it discoverable by Airflow
    nyc_taxi_duration_prediction = duration_prediction_dag()


if __name__ == "__main__":
    # Optional: allow local execution as a script (outside Airflow)
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open(BASE_DIR / "run_id.txt", "w") as f:
        f.write(run_id)