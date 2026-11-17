
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

print("Loading feature dataset...")

df = pd.read_parquet(DATA_DIR / "features_sample.parquet")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "id"]).reset_index(drop=True)

# Model features
feature_columns = [
    "day_of_week",
    "month",
    "day_of_month",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

target_column = "sales"

# Chronological 70 / 15 / 15 split
unique_dates = np.sort(df["date"].unique())

train_end = int(len(unique_dates) * 0.70)
validation_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
validation_dates = unique_dates[train_end:validation_end]
test_dates = unique_dates[validation_end:]

train = df[df["date"].isin(train_dates)].copy()
validation = df[df["date"].isin(validation_dates)].copy()
test = df[df["date"].isin(test_dates)].copy()

X_train = train[feature_columns]
y_train = train[target_column]

X_validation = validation[feature_columns]
y_validation = validation[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

print("\n--- Data Split ---")
print(
    f"Training: {train['date'].min()} to "
    f"{train['date'].max()}"
)
print(
    f"Validation: {validation['date'].min()} to "
    f"{validation['date'].max()}"
)
print(
    f"Final test: {test['date'].min()} to "
    f"{test['date'].max()}"
)

print("\n--- Number of Rows ---")
print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Final test rows: {len(test):,}")

# Train LightGBM
print("\nTraining LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    eval_X=X_validation,
    eval_y=y_validation,
    callbacks=[
        lgb.early_stopping(30, verbose=False)
    ],
)

print("\nBest iteration:", model.best_iteration_)

# Evaluate validation performance
validation_predictions = np.maximum(
    model.predict(X_validation),
    0
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)

# Evaluate final test performance
test_predictions = np.maximum(
    model.predict(X_test),
    0
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

# Seasonal naive baseline on final test
baseline_predictions = test["lag_7"].clip(lower=0)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

print("\n--- Validation Results ---")
print(f"MAE: {validation_mae:.4f}")
print(f"RMSE: {validation_rmse:.4f}")

print("\n--- Final Test Results ---")
print(f"LightGBM MAE: {test_mae:.4f}")
print(f"LightGBM RMSE: {test_rmse:.4f}")

print("\n--- Final Test Model Comparison ---")
print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10}")
print("-" * 42)
print(
    f"{'Seasonal Naive':<20} "
    f"{baseline_mae:>10.4f} {baseline_rmse:>10.4f}"
)
print(
    f"{'LightGBM':<20} "
    f"{test_mae:>10.4f} {test_rmse:>10.4f}"
)

# Save validation predictions for uncertainty estimation
validation_results = validation[
    ["id", "date", "sales"]
].copy()

validation_results["prediction"] = validation_predictions
validation_results["error"] = (
    validation_results["sales"]
    - validation_results["prediction"]
)

validation_results.to_parquet(
    DATA_DIR / "validation_predictions.parquet",
    index=False
)

# Save final test predictions
test_results = test[
    ["id", "date", "sales"]
].copy()

test_results["baseline_prediction"] = (
    baseline_predictions.to_numpy()
)

test_results["lightgbm_prediction"] = test_predictions

test_results.to_parquet(
    DATA_DIR / "model_predictions.parquet",
    index=False
)

# Save trained model
model.booster_.save_model(
    str(DATA_DIR / "lightgbm_model.txt")
)

print("\n--- Saved Files ---")
print("Validation predictions saved.")
print("Final test predictions saved.")
print("Trained model saved.")

print("\nTraining and evaluation completed!")
