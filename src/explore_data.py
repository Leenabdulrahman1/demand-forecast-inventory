
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "docs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading processed sales data...")

    sales = pd.read_parquet(
        PROCESSED_DIR / "sales_long.parquet"
    )

    print("Dataset shape:", sales.shape)

    # 1. Sales by category
    print("\n--- Sales by Category ---")

    category_sales = (
        sales.groupby("cat_id")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print(category_sales)

    category_sales.plot(
        kind="bar",
        title="Total Sales by Category",
        xlabel="Category",
        ylabel="Units Sold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_category.png")
    plt.close()

    # 2. Average sales by weekday
    print("\n--- Average Sales by Weekday ---")

    weekday_sales = (
        sales.groupby("weekday")["sales"]
        .mean()
    )

    print(weekday_sales)

    weekday_sales.plot(
        kind="bar",
        title="Average Sales by Weekday",
        xlabel="Weekday",
        ylabel="Average Units Sold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_weekday.png")
    plt.close()

    # 3. Total daily sales over time
    print("\n--- Daily Sales Trend ---")

    daily_sales = (
        sales.groupby("date")["sales"]
        .sum()
        .sort_index()
    )

    print(daily_sales.head())

    daily_sales.plot(
        figsize=(14, 5),
        title="Total Daily Sales Over Time",
        xlabel="Date",
        ylabel="Units Sold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_sales_trend.png")
    plt.close()

    # 4. Monthly sales patterns
    print("\n--- Average Sales by Month ---")

    monthly_sales = (
        sales.groupby("month")["sales"]
        .mean()
    )

    print(monthly_sales)

    monthly_sales.plot(
        kind="bar",
        title="Average Sales by Month",
        xlabel="Month",
        ylabel="Average Units Sold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_month.png")
    plt.close()

    print("\nEDA completed!")
    print("Charts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
