
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Load the sales dataset
sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")

print("Dataset shape:", sales.shape)

print("\nFirst five rows:")
print(sales.head())

print("\nColumn names:")
print(sales.columns.tolist()[:15])

print("\nProduct and store information:")
print(sales[["item_id", "store_id", "dept_id", "cat_id"]].head())

# Explore the dataset

daily_columns = [
    col for col in sales.columns if col.startswith("d_")
]

print("\nNumber of stores:", sales["store_id"].nunique())
print("Number of products:", sales["item_id"].nunique())
print("Number of categories:", sales["cat_id"].nunique())

# Total sales across all recorded days
total_sales = sales[daily_columns].sum().sum()
print("\nTotal units sold:", total_sales)

# Average daily sales per product-store combination
average_sales = sales[daily_columns].mean(axis=1).mean()
print("Average daily sales per product-store:", round(average_sales, 2))

# Products with the highest total sales
sales["total_sales"] = sales[daily_columns].sum(axis=1)

print("\nTop 5 product-store combinations:")
print(
    sales[["item_id", "store_id", "total_sales"]]
    .sort_values("total_sales", ascending=False)
    .head()
)

# Data quality investigation

daily_columns = [
    col for col in sales.columns if col.startswith("d_")
]

# Check missing values
missing_values = sales[daily_columns].isna().sum().sum()

print("\n--- Data Quality ---")
print("Missing daily sales values:", missing_values)

# Check zero-sales days
zero_sales_days = (sales[daily_columns] == 0).sum().sum()
total_observations = sales[daily_columns].size

print("Zero-sales observations:", zero_sales_days)
print(
    "Percentage of zero sales:",
    round(zero_sales_days / total_observations * 100, 2),
    "%"
)

# Check negative sales
negative_sales = (sales[daily_columns] < 0).sum().sum()
print("Negative sales observations:", negative_sales)
# Calendar investigation

calendar = pd.read_csv(DATA_DIR / "calendar.csv")

print("\n--- Calendar Information ---")
print("Calendar shape:", calendar.shape)

print("\nFirst five calendar rows:")
print(calendar.head())

print("\nCalendar columns:")
print(calendar.columns.tolist())

print("\nDate range:")
print("First date:", calendar["date"].min())
print("Last date:", calendar["date"].max())