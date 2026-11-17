
import pandas as pd
import numpy as np

print("Loading validation predictions...")

df = pd.read_parquet(
    "data/processed/validation_predictions.parquet"
)

# Error = actual sales - predicted sales
df["error"] = df["sales"] - df["prediction"]

print("\n--- Overall Error Summary ---")
print(df["error"].describe().round(3))

print("\n--- Forecast Accuracy ---")
mae = df["error"].abs().mean()
rmse = np.sqrt((df["error"] ** 2).mean())
bias = df["error"].mean()

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Bias: {bias:.4f}")

print("\n--- Error Quantiles ---")
print(
    df["error"]
    .quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    .round(3)
)

print("\n--- Error Summary by Product-Store ID ---")

by_id = (
    df.groupby("id")["error"]
    .agg(["mean", "std", "count"])
)

print(by_id.describe().round(3))

print("\nSaving validation errors...")

df.to_parquet(
    "data/processed/validation_errors.parquet",
    index=False
)

by_id.to_csv(
    "data/processed/validation_error_summary.csv"
)

print("Saved validation error files.")
