
from pathlib import Path

import pandas as pd


# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading sales data...")

    sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")

    print("Original shape:", sales.shape)

    # Identify daily sales columns
    daily_columns = [
        col for col in sales.columns if col.startswith("d_")
    ]

    # Convert wide format to long format
    print("Converting sales data to long format...")

    sales_long = sales.melt(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ],
        value_vars=daily_columns,
        var_name="d",
        value_name="sales",
    )

    print("Long format shape:", sales_long.shape)

    # Load calendar
    print("Loading calendar...")

    calendar = pd.read_csv(DATA_DIR / "calendar.csv")

    # Join sales with calendar
    print("Joining sales with calendar...")

    sales_long = sales_long.merge(
        calendar[["d", "date", "weekday", "month", "year"]],
        on="d",
        how="left",
    )

    # Convert date to datetime
    sales_long["date"] = pd.to_datetime(sales_long["date"])

    # Save as Parquet
    output_path = PROCESSED_DIR / "sales_long.parquet"

    print("Saving processed data...")

    sales_long.to_parquet(
        output_path,
        index=False,
    )

    # Validate processed data
    print("\n--- Validation Checks ---")

    print(
        "Missing dates:",
        sales_long["date"].isna().sum()
    )

    print(
        "Missing sales:",
        sales_long["sales"].isna().sum()
    )

    print(
        "Unique days:",
        sales_long["d"].nunique()
    )

    print(
        "Unique product-store combinations:",
        sales_long["id"].nunique()
    )

    print(
        "Date range:",
        sales_long["date"].min(),
        "to",
        sales_long["date"].max()
    )

    print("\nDone!")
    print("Final shape:", sales_long.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
