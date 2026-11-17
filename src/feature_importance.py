
from pathlib import Path

import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "docs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading trained LightGBM model...")

model = lgb.Booster(
    model_file=str(DATA_DIR / "lightgbm_model.txt")
)

importance = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(
        importance_type="gain"
    ),
})

importance = importance.sort_values(
    "importance",
    ascending=True
)

print("\n--- Feature Importance ---")
print(importance.sort_values(
    "importance",
    ascending=False
).to_string(index=False))

plt.figure(figsize=(10, 6))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.xlabel("Importance (Gain)")
plt.ylabel("Feature")
plt.title("LightGBM Feature Importance")

plt.tight_layout()

output_path = FIGURES_DIR / "feature_importance.png"
plt.savefig(output_path, dpi=150)
plt.close()

print("\nChart saved to:", output_path)
