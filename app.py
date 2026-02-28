import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crime type prediction", layout="centered")

model = joblib.load("artifacts/model.joblib")
meta = json.load(open("artifacts/meta.json", "r"))

st.title("Crime type prediction")
st.write("The model predicts one of the top crime types plus Other.")

defaults = meta["defaults"]
feature_columns = meta["feature_columns"]
cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]

st.subheader("Inputs")

year = st.selectbox("Year", sorted(list(set([int(x.split("-")[0]) for x in meta["data_months"]]))))
month_num = st.selectbox("Month", list(range(1, 13)))

row = defaults.copy()
row["year"] = int(year)
row["month_num"] = int(month_num)

for c in cat_cols:
    row[c] = st.text_input(c, value=str(row.get(c, "")))

for c in num_cols:
    if c in ["year", "month_num"]:
        continue
    row[c] = float(st.number_input(c, value=float(row.get(c, 0.0))))

X = pd.DataFrame([row]).reindex(columns=feature_columns)

if st.button("Predict"):
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