
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Demand Forecasting & Inventory Optimization")

st.markdown(
    """
    An interactive analytics application for evaluating
    demand forecasts and inventory ordering strategies.
    """
)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    predictions = pd.read_parquet(
        "data/processed/model_predictions.parquet"
    )

    sensitivity = pd.read_csv(
        "data/processed/inventory_sensitivity.csv"
    )

    strategies = pd.read_csv(
        "data/processed/inventory_strategy_comparison.csv"
    )

    return predictions, sensitivity, strategies


predictions, sensitivity, strategies = load_data()

# --------------------------------------------------
# 1. FORECAST PERFORMANCE
# --------------------------------------------------

st.header("1. Forecast Performance")

actual = predictions["sales"]
baseline = predictions["baseline_prediction"]
lightgbm = predictions["lightgbm_prediction"]

baseline_mae = (actual - baseline).abs().mean()
lightgbm_mae = (actual - lightgbm).abs().mean()

baseline_rmse = ((actual - baseline) ** 2).mean() ** 0.5
lightgbm_rmse = ((actual - lightgbm) ** 2).mean() ** 0.5

mae_improvement = (
    (baseline_mae - lightgbm_mae) / baseline_mae
) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Baseline MAE", f"{baseline_mae:.3f}")

col2.metric(
    "LightGBM MAE",
    f"{lightgbm_mae:.3f}",
    f"{mae_improvement:.1f}% improvement"
)

col3.metric("Baseline RMSE", f"{baseline_rmse:.3f}")

col4.metric("LightGBM RMSE", f"{lightgbm_rmse:.3f}")

st.caption(
    "Final test evaluation using 100 product-store series."
)

# --------------------------------------------------
# FORECAST COMPARISON
# --------------------------------------------------

st.subheader("Actual vs. Predicted Sales")

chart_data = predictions.head(100).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    chart_data["sales"],
    label="Actual Sales",
    linewidth=1.8
)

ax.plot(
    chart_data["baseline_prediction"],
    label="Seasonal Naive",
    alpha=0.8
)

ax.plot(
    chart_data["lightgbm_prediction"],
    label="LightGBM",
    alpha=0.8
)

ax.set_xlabel("Test Observations")
ax.set_ylabel("Units Sold")
ax.set_title("Forecast Comparison")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------------------------
# 2. INVENTORY STRATEGY COMPARISON
# --------------------------------------------------

st.header("2. Inventory Strategy Comparison")

st.markdown(
    """
    Compare three inventory ordering strategies using
    simulated inventory costs and customer demand fulfillment.
    """
)

# Strategy comparison table

display_strategies = strategies.copy()

display_strategies["Demand Fill Rate"] = (
    display_strategies["Demand Fill Rate"] * 100
)

st.dataframe(
    display_strategies,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# STRATEGY METRICS
# --------------------------------------------------

st.subheader("Strategy Performance")

best_cost_strategy = strategies.loc[
    strategies["Total Inventory Cost"].idxmin()
]

best_fill_strategy = strategies.loc[
    strategies["Demand Fill Rate"].idxmax()
]

col1, col2 = st.columns(2)

col1.metric(
    "Lowest Inventory Cost",
    best_cost_strategy["Strategy"],
    f"{best_cost_strategy['Total Inventory Cost']:,.2f}"
)

col2.metric(
    "Highest Demand Fill Rate",
    best_fill_strategy["Strategy"],
    f"{best_fill_strategy['Demand Fill Rate'] * 100:.2f}%"
)

# --------------------------------------------------
# TOTAL INVENTORY COST CHART
# --------------------------------------------------

st.subheader("Total Inventory Cost by Strategy")

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    strategies["Strategy"],
    strategies["Total Inventory Cost"]
)

ax.set_ylabel("Total Inventory Cost")
ax.set_xlabel("Inventory Strategy")
ax.set_title("Inventory Cost Comparison")
ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", alpha=0.2)

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# FILL RATE CHART
# --------------------------------------------------

st.subheader("Demand Fill Rate by Strategy")

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    strategies["Strategy"],
    strategies["Demand Fill Rate"] * 100
)

ax.set_ylabel("Demand Fill Rate (%)")
ax.set_xlabel("Inventory Strategy")
ax.set_title("Customer Demand Fulfillment")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", alpha=0.2)

st.pyplot(fig)
plt.close(fig)

st.divider()

# --------------------------------------------------
# 3. INVENTORY SENSITIVITY ANALYSIS
# --------------------------------------------------

st.header("3. Inventory Sensitivity Analysis")

st.markdown(
    """
    Explore how different gross margin assumptions affect
    inventory ordering decisions and simulated business costs.
    """
)

margin_options = sorted(
    sensitivity["Gross Margin"].unique()
)

selected_margin = st.selectbox(
    "Select Gross Margin Scenario",
    margin_options,
    format_func=lambda x: f"{x:.0%}"
)

selected = sensitivity[
    sensitivity["Gross Margin"] == selected_margin
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Units Ordered",
    f"{selected['Units Ordered']:,.0f}"
)

col2.metric(
    "Total Inventory Cost",
    f"{selected['Total Cost']:,.2f}"
)

col3.metric(
    "Demand Fill Rate",
    f"{selected['Fill Rate'] * 100:.2f}%"
)

# --------------------------------------------------
# SENSITIVITY CHART
# --------------------------------------------------

st.subheader("Total Cost by Gross Margin")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    sensitivity["Gross Margin"] * 100,
    sensitivity["Total Cost"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Gross Margin (%)")
ax.set_ylabel("Total Inventory Cost")
ax.set_title("Inventory Cost Sensitivity")
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# SENSITIVITY TABLE
# --------------------------------------------------

with st.expander("View Detailed Sensitivity Results"):

    display_data = sensitivity.copy()

    display_data["Gross Margin"] = (
        display_data["Gross Margin"].map(
            lambda x: f"{x:.0%}"
        )
    )

    display_data["Fill Rate"] = (
        display_data["Fill Rate"].map(
            lambda x: f"{x * 100:.2f}%"
        )
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# --------------------------------------------------
# 4. PROJECT INFORMATION
# --------------------------------------------------

st.header("4. Project Information")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        **Forecasting Model**

        - LightGBM
        - Seasonal Naive baseline
        - Chronological evaluation

        **Dataset**

        - M5 Walmart Sales Dataset
        - 100 product-store series in the initial model evaluation
        """
    )

with col2:

    st.markdown(
        """
        **Inventory Optimization**

        - Seasonal Naive ordering
        - LightGBM ordering
        - Uncertainty-adjusted ordering
        - Gross margin sensitivity analysis
        """
    )

st.subheader("Important Assumptions")

st.markdown(
    """
    - Product gross margins are illustrative assumptions.
    - Inventory costs are simulated, not actual Walmart costs.
    - The inventory simulation uses a simplified single-period decision.
    - The forecasting evaluation covers an initial sample of 100 product-store series.
    """
)

st.caption(
    "Demand Forecasting & Inventory Optimization"
)
