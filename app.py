import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crime type prediction", layout="centered")

ARTIFACTS_DIR = Path(".")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "meta.json"
LSOA_PATH = ARTIFACTS_DIR / "lsoa_lookup.csv"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Missing file: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if not META_PATH.exists():
        st.error(f"Missing file: {META_PATH}")
        st.stop()
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required = ["defaults", "feature_columns", "cat_cols", "num_cols", "data_months"]
    missing = [k for k in required if k not in meta]
    if missing:
        st.error(f"meta.json missing keys: {missing}")
        st.stop()

    return meta

@st.cache_data
def load_lsoa_lookup():
    if not LSOA_PATH.exists():
        st.error(f"Missing file: {LSOA_PATH}")
        st.stop()

    lsoa_df = pd.read_csv(LSOA_PATH)
    need = ["LSOA code", "LSOA name", "lat_med", "lon_med"]
    miss = [c for c in need if c not in lsoa_df.columns]
    if miss:
        st.error(f"lsoa_lookup.csv missing columns: {miss}")
        st.stop()

    lsoa_df = lsoa_df.dropna(subset=["LSOA code", "LSOA name"]).reset_index(drop=True)
    if len(lsoa_df) == 0:
        st.error("lsoa_lookup.csv has no valid rows.")
        st.stop()

    return lsoa_df

def safe_float(x, fallback=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(fallback)
    except Exception:
        return float(fallback)

model = load_model()
meta = load_meta()
lsoa_df = load_lsoa_lookup()

st.title("Crime type prediction")
st.write("The model predicts one of the top crime types plus Other.")

defaults = dict(meta["defaults"])
feature_columns = list(meta["feature_columns"])
cat_cols = list(meta["cat_cols"])
num_cols = list(meta["num_cols"])

data_months = [str(x) for x in meta["data_months"] if isinstance(x, str) and "-" in x]
year_to_months = {}
for ym in data_months:
    y, m = ym.split("-")
    year_to_months.setdefault(int(y), set()).add(int(m))

years = sorted(year_to_months.keys())
if not years:
    st.error("No valid data_months found in meta.json")
    st.stop()

def fmt(i):
    return f"{lsoa_df.loc[i,'LSOA code']} | {lsoa_df.loc[i,'LSOA name']}"

st.subheader("Inputs")

with st.form("predict_form"):
    year = st.selectbox("Year", years, index=len(years) - 1)
    months_available = sorted(year_to_months.get(int(year), set()))
    month_num = st.selectbox("Month", months_available, index=len(months_available) - 1)

    lsoa_idx = st.selectbox("LSOA", lsoa_df.index.tolist(), format_func=fmt, index=0)
    sel = lsoa_df.loc[lsoa_idx]

    row = defaults.copy()
    row["year"] = int(year)
    row["month_num"] = int(month_num)

    row["LSOA code"] = str(sel["LSOA code"])
    row["LSOA name"] = str(sel["LSOA name"])
    row["Latitude"] = safe_float(sel["lat_med"], fallback=safe_float(row.get("Latitude", 0.0), 0.0))
    row["Longitude"] = safe_float(sel["lon_med"], fallback=safe_float(row.get("Longitude", 0.0), 0.0))

    st.write("Selected LSOA code:", row["LSOA code"])
    st.write("Selected LSOA name:", row["LSOA name"])
    st.write("Latitude:", row["Latitude"])
    st.write("Longitude:", row["Longitude"])

    for c in cat_cols:
        if c in ["LSOA code", "LSOA name"]:
            continue
        row[c] = st.text_input(c, value=str(row.get(c, "")))

    for c in num_cols:
        if c in ["year", "month_num", "Latitude", "Longitude"]:
            continue
        row[c] = safe_float(
            st.number_input(c, value=safe_float(row.get(c, 0.0), 0.0)),
            fallback=safe_float(row.get(c, 0.0), 0.0),
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([row]).reindex(columns=feature_columns)

    pred = model.predict(X)[0]
    st.write("Prediction:", pred)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        out = (
            pd.DataFrame({"label": model.classes_, "prob": proba})
            .sort_values("prob", ascending=False)
            .head(5)
        )
        st.dataframe(out, use_container_width=True)
