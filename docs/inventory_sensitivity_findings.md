# Inventory Sensitivity Analysis

## Objective
Evaluate how different assumed gross margins affect inventory ordering decisions and simulated inventory costs.

## Scenarios
Four gross margin assumptions were tested: 20%, 30%, 40%, and 50%. The corresponding demand quantiles were used to adjust the LightGBM forecasts.

## Results

| Gross Margin | Units Ordered | Unfulfilled Units | Leftover Units | Total Cost | Fill Rate |
|---|---:|---:|---:|---:|---:|
| 20% | 19,000 | 24,167 | 8,061 | 58,068.13 | 31.16% |
| 30% | 25,893 | 20,676 | 11,463 | 68,712.98 | 41.10% |
| 40% | 33,159 | 17,481 | 15,534 | 75,079.22 | 50.21% |
| 50% | 39,569 | 15,041 | 19,504 | 77,432.97 | 57.16% |

## Key Findings

1. Higher assumed gross margins resulted in larger inventory orders.
2. Fill rate increased from 31.16% to 57.16% across the tested scenarios.
3. Leftover inventory increased as the ordering quantity increased.
4. Total simulated inventory cost was lowest in the 20% scenario and highest in the 50% scenario.

## Interpretation

The results demonstrate a trade-off between inventory cost and customer demand fulfillment. Lower order quantities reduce leftover inventory costs but leave more demand unfulfilled. Higher order quantities improve fulfillment but increase the cost of excess inventory.

## Limitations

- Product margins and unit costs are illustrative assumptions, not actual Walmart financial data.
- The simulation represents a simplified single-period inventory decision.
- Replenishment lead times, inventory carryover, and salvage value are not modeled.
- The forecasting model was developed using an initial sample of 100 product-store series, not the entire M5 dataset.

## Conclusion

The sensitivity analysis demonstrates how inventory decisions depend on business priorities and cost assumptions. The best ordering strategy should balance the cost of excess inventory against the cost of unmet demand.
