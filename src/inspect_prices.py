
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

print("Loading M5 selling prices...")

prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

print("\n--- Dataset Shape ---")
print(prices.shape)

print("\n--- Columns ---")
print(prices.columns.tolist())

print("\n--- First Five Rows ---")
print(prices.head().to_string(index=False))

print("\n--- Missing Values ---")
print(prices.isna().sum())

print("\n--- Price Statistics ---")
print(prices["sell_price"].describe())

print("\n--- Unique Products ---")
print(prices["item_id"].nunique())

print("\n--- Unique Stores ---")
print(prices["store_id"].nunique())

print("\n--- Unique Weeks ---")
print(prices["wm_yr_wk"].nunique())

print("\n--- Sample Product Prices ---")

sample_item = prices["item_id"].iloc[0]
sample_store = prices["store_id"].iloc[0]

sample_prices = prices[
    (prices["item_id"] == sample_item)
    & (prices["store_id"] == sample_store)
].sort_values("wm_yr_wk")

print(sample_prices.head(15).to_string(index=False))

print("\nPrice inspection completed!")
