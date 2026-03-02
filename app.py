import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crime type prediction", layout="centered")

BASE_DIR = Path(".")
MODEL_PATH = BASE_DIR / "model.joblib"
META_PATH = BASE_DIR / "meta.json"
LSOA_PATH = BASE_DIR / "lsoa_lookup.csv"

@st.cache_resource
def load_model(model_mtime: float):
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta(meta_mtime: float):
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_lsoa_lookup(lsoa_mtime: float):
    df = pd.read_csv(LSOA_PATH)
    need = ["LSOA code", "LSOA name", "lat_med", "lon_med"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"lsoa_lookup.csv missing column: {c}")
    df = df.dropna(subset=["LSOA code", "LSOA name"]).reset_index(drop=True)
    if "n" in df.columns:
        df = df.sort_values("n", ascending=False)
    df = df.head(3000).reset_index(drop=True)
    return df

def safe_float(x, fallback=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(fallback)
    except Exception:
        return float(fallback)

st.title("Crime type prediction")
st.write("The model predicts one of the top crime types plus Other.")

try:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing file: {META_PATH}")
    if not LSOA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {LSOA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_PATH}")

    meta = load_meta(META_PATH.stat().st_mtime)
    lsoa_df = load_lsoa_lookup(LSOA_PATH.stat().st_mtime)
except Exception as e:
    st.error(str(e))
    st.stop()

required = ["defaults", "feature_columns", "cat_cols", "num_cols", "data_months"]
missing = [k for k in required if k not in meta]
if missing:
    st.error(f"meta.json missing keys: {missing}")
    st.stop()

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
    try:
        model = load_model(MODEL_PATH.stat().st_mtime)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

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
