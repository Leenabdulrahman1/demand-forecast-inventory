
import pandas as pd

print("Loading final test predictions...")
predictions = pd.read_parquet(
    "data/processed/model_predictions.parquet"
)

print("Loading inventory data...")
inventory = pd.read_parquet(
    "data/processed/inventory_data.parquet",
    columns=[
        "id",
        "date",
        "sell_price",
        "estimated_unit_cost",
        "estimated_unit_profit",
    ],
)

print("Converting dates...")
predictions["date"] = pd.to_datetime(predictions["date"])
inventory["date"] = pd.to_datetime(inventory["date"])

print("Joining predictions with prices and costs...")

optimizer_data = predictions.merge(
    inventory,
    on=["id", "date"],
    how="left",
    validate="one_to_one",
)

print("\n--- Join Results ---")
print("Rows:", len(optimizer_data))
print(
    "Missing selling prices:",
    optimizer_data["sell_price"].isna().sum(),
)
print(
    "Missing estimated costs:",
    optimizer_data["estimated_unit_cost"].isna().sum(),
)

if optimizer_data["sell_price"].isna().any():
    raise ValueError("Some predictions have no matching selling price.")

if optimizer_data["estimated_unit_cost"].isna().any():
    raise ValueError("Some predictions have no matching estimated cost.")

output_path = "data/processed/optimizer_test_data.parquet"

optimizer_data.to_parquet(
    output_path,
    index=False,
)

print("\n--- Final Dataset ---")
print(optimizer_data.head().to_string(index=False))
print("\nSaved:", output_path)
print("Preparation completed!")
