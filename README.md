# Freight Rate Prediction - ML Assessment

This repository contains an end-to-end machine learning pipeline for forecasting freight load rates, developed as part of a technical assessment.

## Project Structure
- `ML1.py` / `ML1.ipynb`: The core machine learning pipeline, featuring exploratory data analysis, robust feature engineering, model training using `HistGradientBoostingRegressor`, and validation.
- `validation_predictions.csv`: Final verified predictions for the 12,000 validation loads.
- `december_predictions.csv`: Daily rate projections for December 2025.
- `score.py`: Official scoring script provided to evaluate predictions.
- `scorer_results/`: Contains the generated pricing visualization (`candidate_december.png`).

## Methodology
1. **Validation Strategy:** Implemented a rigorous time-based expanding-window approach to prevent data leakage and mimic real-world forecasting.
2. **Feature Engineering:** Addressed unseen cities in the validation set (12%) by utilizing geographical coordinates and Haversine distance, combined with log target transformation ($\log(\text{Rate / Distance})$) and recency sample weighting.
3. **Model:** Deployed `HistGradientBoostingRegressor` with a robust two-pass error cleaning mechanism to handle target anomalies.
