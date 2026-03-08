# Load libraries
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# Set page
st.set_page_config(page_title="North Wales Crime Forecast", layout="wide")

# Set paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Load model once
@st.cache_resource
def load_model():
    return joblib.load(ARTIFACT_DIR / "model.joblib")

# Load tables once
@st.cache_data
def load_data():
    with open(ARTIFACT_DIR / "meta.json", "r") as f:
        meta = json.load(f)

    latest_features = pd.read_csv(ARTIFACT_DIR / "latest_features.csv")
    latest_features["Month"] = pd.to_datetime(latest_features["Month"])
    latest_features["forecast_month"] = pd.to_datetime(latest_features["forecast_month"])

    results = pd.read_csv(ARTIFACT_DIR / "results.csv")
    lsoa_lookup = pd.read_csv(ARTIFACT_DIR / "lsoa_lookup.csv")

    return meta, latest_features, results, lsoa_lookup

# Read files
model = load_model()
meta, latest_features, results, lsoa_lookup = load_data()

# Title
st.title("North Wales Crime Forecast")
st.write("Next-month crime count forecast by LSOA.")

# Build labels
latest_features["label"] = (
    latest_features["LSOA name"].astype(str)
    + " | "
    + latest_features["LSOA code"].astype(str)
)

labels = sorted(latest_features["label"].unique().tolist())

# Pick one LSOA
selected_label = st.selectbox("Select LSOA", labels)

# Get one row
row = latest_features.loc[latest_features["label"] == selected_label].copy().iloc[[0]]

# Build model input
feature_cols = meta["feature_cols"]
X_row = row[feature_cols].copy()

# Run prediction
pred = float(model.predict(X_row)[0])
pred = max(0.0, pred)

# Baseline
baseline = float(row["lag_1"].iloc[0])

# Read dates
observed_month = pd.to_datetime(row["Month"].iloc[0]).strftime("%Y-%m")
forecast_month = pd.to_datetime(row["forecast_month"].iloc[0]).strftime("%Y-%m")

# Main metrics
col1, col2, col3 = st.columns(3)
col1.metric("Forecast month", forecast_month)
col2.metric("Model prediction", f"{pred:.2f}")
col3.metric("Naive baseline", f"{baseline:.2f}", delta=f"{pred - baseline:.2f}")

# Area details
st.subheader("Selected area")
st.write(f"LSOA name: {row['LSOA name'].iloc[0]}")
st.write(f"LSOA code: {row['LSOA code'].iloc[0]}")
st.write(f"Latest observed month: {observed_month}")

# Feature view
st.subheader("Input features")
show_cols = [
    "lag_1",
    "lag_2",
    "lag_3",
    "roll_mean_3",
    "roll_mean_6",
    "month_num",
    "month_sin",
    "month_cos",
]
st.dataframe(row[show_cols].T.rename(columns={row.index[0]: "value"}))

# Model table
st.subheader("Model comparison")
st.dataframe(results)

# Saved plots
plot1 = ARTIFACT_DIR / "forecast_vs_actual_test.png"
plot2 = ARTIFACT_DIR / "residuals_test.png"

if plot1.exists():
    st.subheader("Forecast vs actual")
    st.image(str(plot1))

if plot2.exists():
    st.subheader("Residuals")
    st.image(str(plot2))
