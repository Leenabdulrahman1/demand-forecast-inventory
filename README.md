# Demand Forecasting for Inventory Decisions

## Overview

A machine learning project that forecasts retail product demand and uses the predictions to support inventory replenishment decisions.

The project combines demand forecasting, uncertainty estimation, and inventory optimization to investigate how forecasting approaches affect simulated inventory costs.

## Objectives

- Explore and preprocess historical retail sales data.
- Build a seasonal naive forecasting baseline.
- Train a global machine learning forecasting model.
- Evaluate forecasts using chronological training, validation, and test splits.
- Estimate inventory decisions using forecast uncertainty.
- Compare simulated inventory costs under different forecasting strategies.
- Analyze how different gross margin assumptions affect inventory decisions.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- Statsmodels
- Matplotlib
- Streamlit
- Pytest

## Project Status

**Completed — ready for demonstration.**

The project includes:

- Data exploration and preprocessing using the M5 retail dataset.
- Feature engineering and LightGBM demand forecasting.
- Chronological model evaluation.
- Inventory optimization using forecast predictions and uncertainty estimates.
- Sensitivity analysis under different gross margin assumptions.
- An interactive Streamlit dashboard for forecasting and inventory analysis.

## Key Results

Compared with the seasonal naive baseline, the LightGBM model achieved:

- **19.2% lower MAE.**
- **27.9% lower RMSE.**

The uncertainty-adjusted inventory strategy achieved:

- **10.8% lower simulated inventory cost** than the seasonal naive strategy under the selected assumptions.

## Evaluation Metrics

### Forecasting Performance
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Inventory Performance
- Simulated total inventory cost
- Estimated holding cost
- Estimated stockout cost
- Fill rate
- Unfulfilled demand
- Leftover inventory

## Scope and Assumptions

- Model evaluation uses an initial sample of 100 product-store series from the M5 dataset.
- Selling prices are based on the dataset; product costs and gross margins are illustrative assumptions, not Walmart's actual procurement costs.
- Inventory optimization uses a daily single-period model without inventory carry-over or replenishment lead times.
- Forecast evaluation uses a one-step-ahead setup with actual historical lag values.

