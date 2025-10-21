# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import pickle
import os
import sys

# %%
# Prefer pyarrow for all parquet IO in pandas and print diagnostics
pd.options.io.parquet.engine = 'pyarrow'  # affects read_parquet/to_parquet defaults
print(f"Using pandas parquet engine: {pd.get_option('io.parquet.engine')}")
try:
    import pyarrow as _pa  # noqa: F401
    import pyarrow.parquet as _pq  # noqa: F401
    print("pyarrow is importable ✅")
except Exception as e:
    print(f"Warning: pyarrow import failed: {e}")
    # Don't exit immediately; downstream calls with engine='pyarrow' will raise a clear error



# %%
print(f"pandas version: {pd.__version__}")

# %%
# Configuration: set SAMPLE_FRAC early (can be overridden via env var)
SAMPLE_FRAC: float = float(os.getenv("SAMPLE_FRAC", "0.3"))
print(f"SAMPLE_FRAC set to {SAMPLE_FRAC}")

# %%
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

print(mlflow.get_experiment_by_name("nyc-taxi-experiment"))

# %%
def read_dataframe(filename: str, sample_frac: float | None = None) -> pd.DataFrame:
    # Read only necessary columns to reduce memory
    cols = [
        'lpep_dropoff_datetime', 'lpep_pickup_datetime',
        'PULocationID', 'DOLocationID', 'trip_distance'
    ]
    print(f"Reading parquet from {filename} with pyarrow and columns={cols} ...", flush=True)
    df = pd.read_parquet(filename, engine='pyarrow', columns=cols)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"Sampled dataframe to frac={sample_frac}: shape={df.shape}")
    else:
        print(f"Loaded dataframe shape: {df.shape}")
    return df

# %%
df_train = read_dataframe(
    'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet',
    sample_frac=SAMPLE_FRAC  # default set at top, override via env var
)
df_val = read_dataframe(
    'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet',
    sample_frac=SAMPLE_FRAC
)


# %%
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

# %%
categorical = ['PU_DO'] 
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

# %%
target = 'duration'
y_train = df_train[target].values 
y_val = df_val[target].values


# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# Avoid logging full datasets to reduce memory/overhead; keep models and metrics
mlflow.sklearn.autolog(log_datasets=False)

try:
    models = [
        GradientBoostingRegressor(random_state=42),
        RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=42),
        ExtraTreesRegressor(n_estimators=50, n_jobs=1, random_state=42),
    ]

    for mlmodel in models:
        print(f"Training {mlmodel.__class__.__name__} ...", flush=True)
        with mlflow.start_run():
            mlflow.log_param("train-data-path", "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet")
            mlflow.log_param("valid-data-path", "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet")
            mlflow.log_param("sample_frac", SAMPLE_FRAC)

            mlmodel.fit(X_train, y_train)

            y_pred = mlmodel.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            print(f"{mlmodel.__class__.__name__} RMSE: {rmse:.4f}")
            mlflow.log_metric("rmse", rmse)

except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)
finally:
    print("Experiment completed.")
