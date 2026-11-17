
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

print("Loading sales and price data...")

sales = pd.read_parquet(
    PROCESSED_DIR / "sales_long.parquet"
)

prices = pd.read_csv(
    RAW_DIR / "sell_prices.csv"
)

calendar = pd.read_csv(
    RAW_DIR / "calendar.csv"
)

# Keep only the columns needed
sales = sales[
    ["id", "item_id", "store_id", "date", "sales"]
].copy()

calendar = calendar[
    ["date", "wm_yr_wk"]
].copy()

sales["date"] = pd.to_datetime(sales["date"])
calendar["date"] = pd.to_datetime(calendar["date"])

print("Merging calendar weeks...")

sales = sales.merge(
    calendar,
    on="date",
    how="left",
    validate="many_to_one"
)

print("Merging selling prices...")

sales = sales.merge(
    prices,
    on=["item_id", "store_id", "wm_yr_wk"],
    how="left",
    validate="many_to_one"
)

print("\n--- Merge Results ---")
print("Rows:", len(sales))
print("Missing prices:", sales["sell_price"].isna().sum())

# Keep rows where a selling price is available
sales = sales.dropna(subset=["sell_price"]).copy()

# Initial financial assumptions
GROSS_MARGIN = 0.30

sales["estimated_unit_cost"] = (
    sales["sell_price"] * (1 - GROSS_MARGIN)
)

sales["estimated_unit_profit"] = (
    sales["sell_price"] - sales["estimated_unit_cost"]
)

sales["sales_revenue"] = (
    sales["sales"] * sales["sell_price"]
)

sales["estimated_gross_profit"] = (
    sales["sales"] * sales["estimated_unit_profit"]
)

print("\n--- Financial Data Summary ---")
print(
    sales[
        [
            "sell_price",
            "estimated_unit_cost",
            "estimated_unit_profit"
        ]
    ].describe().round(2)
)

print("\n--- Example Records ---")
print(
    sales[
        [
            "item_id",
            "store_id",
            "date",
            "sales",
            "sell_price",
            "estimated_unit_cost",
            "estimated_unit_profit"
        ]
    ].head().to_string(index=False)
)

output_path = PROCESSED_DIR / "inventory_data.parquet"

sales.to_parquet(output_path, index=False)

print("\nInventory data saved to:", output_path)
print("Inventory data preparation completed!")
