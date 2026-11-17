
import numpy as np
import pandas as pd

print("Loading optimizer data...")

df = pd.read_parquet(
    "data/processed/optimizer_test_data.parquet"
)

print("Loading validation errors...")

validation = pd.read_parquet(
    "data/processed/validation_errors.parquet"
)

# Calculate the required quantiles from validation errors
quantiles = (
    validation.groupby("id")["error"]
    .quantile([0.20, 0.30, 0.40, 0.50])
    .unstack()
)

quantiles.columns = [
    "error_q20",
    "error_q30",
    "error_q40",
    "error_q50",
]

quantiles = quantiles.reset_index()

df = df.merge(
    quantiles,
    on="id",
    how="left",
    validate="many_to_one",
)

required_columns = [
    "error_q20",
    "error_q30",
    "error_q40",
    "error_q50",
]

if df[required_columns].isna().any().any():
    raise ValueError("Missing uncertainty estimates.")

results = []

for margin in [0.20, 0.30, 0.40, 0.50]:

    quantile_column = f"error_q{int(margin * 100)}"

    adjusted_demand = (
        df["lightgbm_prediction"] + df[quantile_column]
    ).clip(lower=0)

    order = np.ceil(adjusted_demand)
    demand = df["sales"]

    underage_units = (demand - order).clip(lower=0)
    overage_units = (order - demand).clip(lower=0)

    underage_cost = (
        underage_units * df["sell_price"] * margin
    ).sum()

    overage_cost = (
        overage_units * df["sell_price"] * (1 - margin)
    ).sum()

    total_demand = demand.sum()
    fulfilled_units = np.minimum(demand, order).sum()

    fill_rate = (
        fulfilled_units / total_demand
        if total_demand > 0 else np.nan
    )

    results.append({
        "Gross Margin": margin,
        "Target Quantile": margin,
        "Units Ordered": order.sum(),
        "Unfulfilled Units": underage_units.sum(),
        "Leftover Units": overage_units.sum(),
        "Underage Cost": underage_cost,
        "Overage Cost": overage_cost,
        "Total Cost": underage_cost + overage_cost,
        "Fill Rate": fill_rate,
    })

results_df = pd.DataFrame(results)

print("\n--- Inventory Sensitivity Analysis ---")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Gross Margin": "{:.0%}".format,
            "Target Quantile": "{:.0%}".format,
            "Underage Cost": "{:,.2f}".format,
            "Overage Cost": "{:,.2f}".format,
            "Total Cost": "{:,.2f}".format,
            "Fill Rate": "{:.2%}".format,
        },
    )
)

results_df.to_csv(
    "data/processed/inventory_sensitivity.csv",
    index=False,
)

print("\nSaved: data/processed/inventory_sensitivity.csv")
print("Sensitivity analysis completed!")
