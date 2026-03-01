import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crime type prediction", layout="centered")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "meta.json"

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

st.subheader("Inputs")

with st.form("predict_form"):
    year = st.selectbox("Year", years, index=len(years) - 1)
    months_available = sorted(year_to_months.get(int(year), set()))
    if not months_available:
        months_available = list(range(1, 13))
    month_num = st.selectbox("Month", months_available, index=len(months_available) - 1)

    row = defaults.copy()
    row["year"] = int(year)
    row["month_num"] = int(month_num)

    for c in cat_cols:
        row[c] = st.text_input(c, value=str(row.get(c, "")))

    for c in num_cols:
        if c in ["year", "month_num"]:
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
