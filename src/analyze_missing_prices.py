
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

print("Loading inventory data...")

df = pd.read_parquet(
    DATA_DIR / "inventory_data.parquet"
)

print("Loading original sales data...")

sales = pd.read_parquet(
    DATA_DIR / "sales_long.parquet"
)

# Reconstruct which records had missing prices
# by comparing the original and merged datasets.
print("\n--- Price Coverage ---")

total_rows = len(sales)
priced_rows = len(df)
missing_rows = total_rows - priced_rows

print(f"Total sales records: {total_rows:,}")
print(f"Records with prices: {priced_rows:,}")
print(f"Records without prices: {missing_rows:,}")
print(f"Missing percentage: {missing_rows / total_rows * 100:.2f}%")

# Investigate missing prices by product and store.
calendar = pd.read_csv(
    PROJECT_DIR / "data" / "raw" / "calendar.csv"
)

prices = pd.read_csv(
    PROJECT_DIR / "data" / "raw" / "sell_prices.csv"
)

calendar = calendar[["date", "wm_yr_wk"]]
calendar["date"] = pd.to_datetime(calendar["date"])

sales = sales.merge(
    calendar,
    on="date",
    how="left"
)

coverage = sales.merge(
    prices,
    on=["item_id", "store_id", "wm_yr_wk"],
    how="left"
)

missing = coverage[coverage["sell_price"].isna()].copy()

print("\n--- Missing Prices by Category ---")
print(
    missing.groupby("cat_id").size()
    if "cat_id" in missing.columns
    else "Category not available"
)

print("\n--- Missing Prices by Store ---")
print(missing.groupby("store_id").size())

print("\n--- Missing Prices by Sales Quantity ---")
print(
    missing["sales"].describe()
)

print("\n--- Missing Price Records with Positive Sales ---")
print(
    (missing["sales"] > 0).sum()
)

print("\nMissing price analysis completed!")
