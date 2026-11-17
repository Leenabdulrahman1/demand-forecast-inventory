
import pandas as pd

print("Loading validation errors...")

df = pd.read_parquet(
    "data/processed/validation_errors.parquet"
)

print("\nCalculating uncertainty by product-store ID...")

uncertainty = (
    df.groupby("id")["error"]
    .quantile([0.10, 0.30, 0.50, 0.70, 0.90])
    .unstack()
)

uncertainty.columns = [
    "error_q10",
    "error_q30",
    "error_q50",
    "error_q70",
    "error_q90",
]

uncertainty["error_std"] = (
    df.groupby("id")["error"].std()
)

uncertainty["mae"] = (
    df.groupby("id")["error"]
    .apply(lambda x: x.abs().mean())
)

print("\n--- Product-Level Uncertainty Summary ---")
print(uncertainty.describe().round(3))

print("\n--- Five Products With Highest Error Variability ---")
print(
    uncertainty.sort_values(
        "error_std",
        ascending=False
    ).head(5).round(3)
)

print("\n--- Five Products With Lowest Error Variability ---")
print(
    uncertainty.sort_values(
        "error_std"
    ).head(5).round(3)
)

uncertainty.to_csv(
    "data/processed/product_uncertainty.csv"
)

print("\nSaved: data/processed/product_uncertainty.csv")
