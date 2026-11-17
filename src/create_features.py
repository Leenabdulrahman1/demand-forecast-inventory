
from pathlib import Path

import pandas as pd


# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def main():
    print("Loading processed sales data...")

    sales = pd.read_parquet(
        PROCESSED_DIR / "sales_long.parquet"
    )

    # Select a sample of 100 product-store combinations
    print("Selecting sample...")

    selected_ids = (
        sales["id"]
        .drop_duplicates()
        .head(100)
    )

    sample = sales[
        sales["id"].isin(selected_ids)
    ].copy()

    # Sort by product-store combination and date
    sample = sample.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    print("Sample shape:", sample.shape)

    # Create calendar features
    print("Creating calendar features...")

    sample["day_of_week"] = sample["date"].dt.day_of_week
    sample["month"] = sample["date"].dt.month
    sample["day_of_month"] = sample["date"].dt.day

    # Create lag features
    print("Creating lag features...")

    for lag in [7, 14, 28]:
        sample[f"lag_{lag}"] = (
            sample.groupby("id")["sales"]
            .shift(lag)
        )

    # Create rolling average features
    print("Creating rolling features...")

    for window in [7, 28]:
        sample[f"rolling_mean_{window}"] = (
            sample.groupby("id")["sales"]
            .transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
        )

    # Remove rows without enough historical data
    print("Removing rows with missing features...")

    sample = sample.dropna().copy()

    # Save feature dataset
    output_path = (
        PROCESSED_DIR / "features_sample.parquet"
    )

    sample.to_parquet(
        output_path,
        index=False,
    )

    print("\n--- Feature Dataset Summary ---")
    print("Final shape:", sample.shape)
    print("Number of products/stores:", sample["id"].nunique())

    print("\nFeature columns:")
    print(sample.columns.tolist())

    print("\nFirst five rows:")
    print(sample.head())

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
