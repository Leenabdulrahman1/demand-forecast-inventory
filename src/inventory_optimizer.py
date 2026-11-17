
import numpy as np
import pandas as pd

print("Loading optimizer test data...")

df = pd.read_parquet(
    "data/processed/optimizer_test_data.parquet"
)

print("Loading product uncertainty estimates...")

uncertainty = pd.read_csv(
    "data/processed/product_uncertainty.csv"
)

# Attach validation-period uncertainty to each product's test predictions
df = df.merge(
    uncertainty[["id", "error_q30"]],
    on="id",
    how="left",
    validate="many_to_one",
)

if df["error_q30"].isna().any():
    raise ValueError("Some products have no uncertainty estimate.")

# Calculate three ordering strategies
df["seasonal_naive_order"] = np.ceil(
    df["baseline_prediction"].clip(lower=0)
)

df["lightgbm_order"] = np.ceil(
    df["lightgbm_prediction"].clip(lower=0)
)

adjusted_demand = (
    df["lightgbm_prediction"] + df["error_q30"]
).clip(lower=0)

df["uncertainty_adjusted_order"] = np.ceil(
    adjusted_demand
)

# Cost assumptions
# Underage: lost estimated gross profit
# Overage: purchase cost of unsold inventory
df["underage_cost_per_unit"] = (
    df["estimated_unit_profit"]
)

df["overage_cost_per_unit"] = (
    df["estimated_unit_cost"]
)

strategies = {
    "Seasonal Naive": "seasonal_naive_order",
    "LightGBM Forecast": "lightgbm_order",
    "Uncertainty Adjusted": "uncertainty_adjusted_order",
}

results = []

for strategy, order_column in strategies.items():

    order = df[order_column]
    demand = df["sales"]

    # Units that could not be fulfilled
    underage_units = (demand - order).clip(lower=0)

    # Units left unsold
    overage_units = (order - demand).clip(lower=0)

    underage_cost = (
        underage_units * df["underage_cost_per_unit"]
    )

    overage_cost = (
        overage_units * df["overage_cost_per_unit"]
    )

    total_cost = underage_cost + overage_cost

    total_demand = demand.sum()
    fulfilled_units = np.minimum(demand, order).sum()

    results.append({
        "Strategy": strategy,
        "Total Units Ordered": order.sum(),
        "Total Demand": total_demand,
        "Unfulfilled Units": underage_units.sum(),
        "Leftover Units": overage_units.sum(),
        "Underage Cost": underage_cost.sum(),
        "Overage Cost": overage_cost.sum(),
        "Total Inventory Cost": total_cost.sum(),
        "Demand Fill Rate": (
            fulfilled_units / total_demand
            if total_demand > 0 else np.nan
        ),
    })

results_df = pd.DataFrame(results)

# Compare strategies
print("\n--- Inventory Optimization Results ---")

display_columns = [
    "Strategy",
    "Total Units Ordered",
    "Unfulfilled Units",
    "Leftover Units",
    "Underage Cost",
    "Overage Cost",
    "Total Inventory Cost",
    "Demand Fill Rate",
]

print(
    results_df[display_columns].to_string(
        index=False,
        formatters={
            "Underage Cost": "{:,.2f}".format,
            "Overage Cost": "{:,.2f}".format,
            "Total Inventory Cost": "{:,.2f}".format,
            "Demand Fill Rate": "{:.2%}".format,
        },
    )
)

# Save detailed decisions and summary
df.to_parquet(
    "data/processed/inventory_decisions.parquet",
    index=False,
)

results_df.to_csv(
    "data/processed/inventory_strategy_comparison.csv",
    index=False,
)

print("\nSaved:")
print("data/processed/inventory_decisions.parquet")
print("data/processed/inventory_strategy_comparison.csv")

print("\nInventory optimization evaluation completed!")
