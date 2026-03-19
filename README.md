# North Wales Crime Forecast

This project builds a one-step-ahead machine learning system to predict next-month crime counts at LSOA level using Police.uk street crime data from the North Wales Police dataset.

## Project goal

The goal was to build an end-to-end forecasting system for monthly crime counts.
For each LSOA and month t, the model uses only information available up to month t to predict the crime count in month t+1.

## Data

Source: Police.uk street crime data from the North Wales Police dataset.

The dataset was transformed into a monthly panel with:
- one row per LSOA and month
- missing LSOA-month combinations filled with 0
- lag and rolling features built from past values only
- a next-month target for supervised learning

Note:
The project keeps both E-codes and W-codes. This is intentional because the North Wales Police dataset includes LSOAs beyond Wales only.

## Features

Main features:
- LSOA code
- month number
- month_sin
- month_cos
- lag_1
- lag_2
- lag_3
- roll_mean_3
- roll_mean_6

Target:
- next-month crime count per LSOA

## Models

Models tested:
- Naive baseline using lag_1
- Poisson Regressor
- Tweedie Regressor

Final selected model:
- poisson_a_0.0001

## Evaluation

Metrics:
- MAE
- RMSE

Validation was used for model selection.
The final test set used December 2025 feature rows, which correspond to forecasts for January 2026.

## Streamlit app

The Streamlit app loads saved artifacts and provides:
- next-month crime forecast for a selected LSOA
- naive baseline
- recent crime history
- input feature view
- model comparison table
- forecast and residual plots
- how-it-works explanation
- hotspot choropleth map of predicted next-month crime counts by LSOA

## Files

- `app.py` runs the Streamlit app
- `artifacts/model.joblib` trained model
- `artifacts/latest_features.csv` latest input rows for prediction
- `artifacts/meta.json` project metadata
- `artifacts/results.csv` model results
- `artifacts/lsoa_lookup.csv` lookup table for areas
- `artifacts/forecast_map.geojson` hotspot map boundaries with predictions
- `artifacts/forecast_map_table.csv` table used for map generation

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
