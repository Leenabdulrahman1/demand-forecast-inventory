
import pandas as pd

files = {
    "Inventory data": "data/processed/inventory_data.parquet",
    "Final test predictions": "data/processed/model_predictions.parquet",
}

for name, path in files.items():
    print(f"\n--- {name} ---")

    df = pd.read_parquet(path)

    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
