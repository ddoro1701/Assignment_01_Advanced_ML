# North Wales Crime Forecast

This project predicts next-month crime counts for each LSOA in North Wales using Police.uk street crime data.

## Project goal

Build an end-to-end machine learning system for a real forecasting problem.
The model uses past monthly crime counts and time features to predict the count for the next month.

## Data

Source: Police.uk street crime data for North Wales.

Data was converted into a monthly panel:
- one row per LSOA and month
- missing months filled with 0
- lag and rolling features built from past values only

## Model

Target:
- next-month crime count per LSOA

Main features:
- LSOA code
- month features
- lag_1, lag_2, lag_3
- roll_mean_3, roll_mean_6

Models tested:
- Naive baseline using lag_1
- Poisson Regressor
- Tweedie Regressor

Evaluation:
- MAE
- RMSE

## Streamlit app

The app loads saved artifacts and shows:
- predicted next-month crime count
- naive baseline
- model comparison table
- test plots

## Files

- `app.py` runs the Streamlit app
- `artifacts/model.joblib` trained model
- `artifacts/latest_features.csv` latest input rows for prediction
- `artifacts/meta.json` project metadata
- `artifacts/results.csv` model results
- `artifacts/lsoa_lookup.csv` lookup table for areas

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
