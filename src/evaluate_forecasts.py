
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

print("Loading model predictions...")

df = pd.read_parquet(
    DATA_DIR / "model_predictions.parquet"
)

df["date"] = pd.to_datetime(df["date"])

# Calculate forecast errors
df["error"] = (
    df["sales"] - df["lightgbm_prediction"]
)

df["absolute_error"] = df["error"].abs()

df["squared_error"] = df["error"] ** 2

print("\n--- Forecast Evaluation ---")

mae = mean_absolute_error(
    df["sales"],
    df["lightgbm_prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        df["sales"],
        df["lightgbm_prediction"]
    )
)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print("\n--- Forecast Error Statistics ---")

print(
    df["error"].describe().round(4)
)

# Estimate uncertainty from forecast errors.
# Keep the errors grouped by product/store.
error_summary = (
    df.groupby("id")["error"]
    .agg(
        mean_error="mean",
        std_error="std",
        error_count="count"
    )
    .reset_index()
)

print("\n--- Error Summary ---")

print(
    error_summary[
        ["mean_error", "std_error", "error_count"]
    ].describe().round(4)
)

output_path = DATA_DIR / "forecast_errors.parquet"

df.to_parquet(
    output_path,
    index=False
)

summary_path = DATA_DIR / "error_summary.parquet"

error_summary.to_parquet(
    summary_path,
    index=False
)

print("\nForecast errors saved to:", output_path)
print("Error summary saved to:", summary_path)

print("\nForecast evaluation completed!")
