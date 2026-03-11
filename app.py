# Import libraries
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import altair as alt

# Set up the page
st.set_page_config(
    page_title="North Wales Crime Forecast",
    page_icon="📈",
    layout="wide"
)


# Define file paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


# Load the trained model once
@st.cache_resource
def load_model():
    try:
        return joblib.load(ARTIFACT_DIR / "model.joblib")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()


# Load all saved data once
@st.cache_data
def load_artifacts():
    try:
        with open(ARTIFACT_DIR / "meta.json", "r") as f:
            meta = json.load(f)

        latest_features = pd.read_csv(ARTIFACT_DIR / "latest_features.csv")
        latest_features["Month"] = pd.to_datetime(latest_features["Month"], errors="coerce")
        latest_features["forecast_month"] = pd.to_datetime(latest_features["forecast_month"], errors="coerce")

        results = pd.read_csv(ARTIFACT_DIR / "results.csv")
        lsoa_lookup = pd.read_csv(ARTIFACT_DIR / "lsoa_lookup.csv")

        history = None
        history_path = ARTIFACT_DIR / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            history["Month"] = pd.to_datetime(history["Month"], errors="coerce")

        centroids = None
        centroids_path = ARTIFACT_DIR / "lsoa_centroids.csv"
        if centroids_path.exists():
            centroids = pd.read_csv(centroids_path)

        return meta, latest_features, results, lsoa_lookup, history, centroids

    except Exception as e:
        st.error(f"Artifact load failed: {e}")
        st.stop()


# Format metric values
def fmt_value(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


# Rename technical features for normal users
def build_feature_table(row):
    return pd.DataFrame({
        "Feature": [
            "Previous month crime count",
            "Crime count two months ago",
            "Crime count three months ago",
            "Average over previous 3 months",
            "Average over previous 6 months",
            "Month number"
        ],
        "Value": [
            row["lag_1"].iloc[0],
            row["lag_2"].iloc[0],
            row["lag_3"].iloc[0],
            row["roll_mean_3"].iloc[0],
            row["roll_mean_6"].iloc[0],
            row["month_num"].iloc[0]
        ]
    })


# Load model and artifacts
model = load_model()
meta, latest_features, results, lsoa_lookup, history, centroids = load_artifacts()


# Merge map coordinates if available
if centroids is not None:
    latest_features = latest_features.merge(
        centroids,
        on=["LSOA code", "LSOA name"],
        how="left"
    )


# Create a readable label for selection
latest_features["label"] = (
    latest_features["LSOA name"].astype(str)
    + " | "
    + latest_features["LSOA code"].astype(str)
)

labels = sorted(latest_features["label"].dropna().unique().tolist())


# Show app title and short explanation
st.title("North Wales Crime Forecast")
st.write(
    "This app predicts the number of recorded crimes for the next month in a selected "
    "Lower Layer Super Output Area (LSOA) in North Wales."
)
st.write(
    "The prediction is based on historical monthly crime counts and time-based features. "
    "A simple baseline is also shown for comparison."
)


# Show basic project information
best_model_name = meta.get("best_model", "Unknown")
latest_observed_month = meta.get("latest_observed_month", "Unknown")
app_forecast_month = meta.get("app_forecast_month", "Unknown")

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Final selected model", best_model_name)
info_col2.metric("Latest observed month", latest_observed_month)
info_col3.metric("Forecast month", app_forecast_month)


# Let the user select an area
selected_label = st.selectbox("Select an LSOA", labels)

row = latest_features.loc[latest_features["label"] == selected_label].copy().iloc[[0]]
feature_cols = meta["feature_cols"]
X_row = row[feature_cols].copy()


# Run the model prediction
pred = float(model.predict(X_row)[0])
pred = max(0.0, pred)

baseline = float(row["lag_1"].iloc[0])

observed_month = row["Month"].iloc[0]
forecast_month = row["forecast_month"].iloc[0]

observed_month_str = observed_month.strftime("%Y-%m") if pd.notna(observed_month) else "Unknown"
forecast_month_str = forecast_month.strftime("%Y-%m") if pd.notna(forecast_month) else "Unknown"

selected_lsoa_name = row["LSOA name"].iloc[0]
selected_lsoa_code = row["LSOA code"].iloc[0]


# Create tabs for a clearer layout
tab1, tab2, tab3 = st.tabs(["Forecast", "Model comparison", "How it works"])


# Show the main forecast
with tab1:
    st.subheader("Next-month forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast month", forecast_month_str)
    col2.metric("Model prediction", fmt_value(pred))
    col3.metric("Naive baseline", fmt_value(baseline), delta=fmt_value(pred - baseline))

    st.write(
        f"The model predicts approximately {fmt_value(pred)} crimes for "
        f"{selected_lsoa_name} ({selected_lsoa_code}) in {forecast_month_str}."
    )

    st.info(
        "The naive baseline uses the previous month's crime count as the forecast. "
        "This helps show whether the trained model adds value beyond a simple rule."
    )

    st.subheader("Selected area")
    details_df = pd.DataFrame({
        "Field": ["LSOA name", "LSOA code", "Latest observed month"],
        "Value": [selected_lsoa_name, selected_lsoa_code, observed_month_str]
    })
    st.dataframe(details_df, use_container_width=True, hide_index=True)

    st.subheader("Approximate location")
    st.write(
        "The forecast is made at LSOA level. The map below shows an approximate point for the selected area, not the full boundary."
    )

    if "lat_med" in row.columns and "lon_med" in row.columns:
        lat_val = row["lat_med"].iloc[0]
        lon_val = row["lon_med"].iloc[0]

        if pd.notna(lat_val) and pd.notna(lon_val):
            map_df = pd.DataFrame({
                "lat": [lat_val],
                "lon": [lon_val]
            })
            st.map(map_df, zoom=11)
        else:
            st.write("No map coordinates are available for this LSOA.")
    else:
        st.write("No map coordinates file was found.")

    st.subheader("Input features used for prediction")
    feature_display = build_feature_table(row)
    st.dataframe(feature_display, use_container_width=True, hide_index=True)

    st.subheader("Interpretation")
    if pred > baseline:
        st.write(
            "The model forecast is higher than the naive baseline. "
            "This suggests the model expects crime levels to increase relative to the previous month."
        )
    elif pred < baseline:
        st.write(
            "The model forecast is lower than the naive baseline. "
            "This suggests the model expects crime levels to decrease relative to the previous month."
        )
    else:
        st.write(
            "The model forecast is equal to the naive baseline. "
            "This suggests little expected change relative to the previous month."
        )

    # Show recent history with visible forecast and baseline points
if history is not None:
    hist_area = history.loc[history["LSOA code"] == selected_lsoa_code].copy()
    hist_area = hist_area.sort_values("Month")

    if not hist_area.empty:
        st.subheader("Recent crime history for this LSOA")

        # Build observed series
        observed_df = hist_area[["Month", "crime_count"]].copy()
        observed_df["Series"] = "Observed"

        # Get the last observed point
        last_month = observed_df["Month"].max()
        last_value = float(observed_df.loc[observed_df["Month"] == last_month, "crime_count"].iloc[0])

        # Build model forecast segment from last observed point
        model_df = pd.DataFrame({
            "Month": [last_month, forecast_month],
            "crime_count": [last_value, pred],
            "Series": ["Model prediction", "Model prediction"]
        })

        # Build baseline forecast segment from last observed point
        baseline_df = pd.DataFrame({
            "Month": [last_month, forecast_month],
            "crime_count": [last_value, baseline],
            "Series": ["Naive baseline", "Naive baseline"]
        })

        # Combine all series
        chart_df = pd.concat([observed_df, model_df, baseline_df], ignore_index=True)

        # Base chart settings
        base = alt.Chart(chart_df).encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("crime_count:Q", title="Crime count"),
            color=alt.Color("Series:N", title="Series")
        )

        # Observed history as one continuous line
        observed_line = base.transform_filter(
            alt.datum.Series == "Observed"
        ).mark_line()

        observed_points = base.transform_filter(
            alt.datum.Series == "Observed"
        ).mark_point(filled=True, size=45)

        # Model prediction as a connected line from the last observed point
        model_line = base.transform_filter(
            alt.datum.Series == "Model prediction"
        ).mark_line(strokeDash=[6, 3])

        model_points = base.transform_filter(
            alt.datum.Series == "Model prediction"
        ).mark_point(filled=True, size=90)

        # Naive baseline as a connected line from the last observed point
        baseline_line = base.transform_filter(
            alt.datum.Series == "Naive baseline"
        ).mark_line(strokeDash=[2, 2])

        baseline_points = base.transform_filter(
            alt.datum.Series == "Naive baseline"
        ).mark_point(filled=True, size=90)

        # Add a vertical rule at the forecast month
        rule_df = pd.DataFrame({"Month": [forecast_month]})
        forecast_rule = alt.Chart(rule_df).mark_rule().encode(x="Month:T")

        # Combine layers
        chart = (
            alt.layer(
                observed_line,
                observed_points,
                model_line,
                model_points,
                baseline_line,
                baseline_points,
                forecast_rule
            )
            .properties(height=380)
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "The solid line shows observed monthly crime counts. "
            "The two projected lines show the next-month model prediction and the naive baseline."
        )


# Show model comparison and saved evaluation outputs
with tab2:
    st.subheader("Model comparison")

    st.write(
        f"The final selected model is {best_model_name}. "
        "Model selection was based on validation performance."
    )

    st.write(
        "Lower Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) indicate better forecasting accuracy."
    )

    st.dataframe(results, use_container_width=True, hide_index=True)

    if "model" in results.columns:
        best_row = results.loc[results["model"] == best_model_name]
        if not best_row.empty:
            st.subheader("Selected model summary")
            st.dataframe(best_row, use_container_width=True, hide_index=True)

    plot1 = ARTIFACT_DIR / "forecast_vs_actual_test.png"
    plot2 = ARTIFACT_DIR / "residuals_test.png"

    if plot1.exists():
        st.subheader("Forecast versus actual values on the test set")
        st.image(str(plot1))

    if plot2.exists():
        st.subheader("Residual distribution on the test set")
        st.image(str(plot2))


# Explain the workflow in simple language
with tab3:
    st.subheader("What this system does")
    st.write(
        "The system forecasts next-month crime counts for each LSOA in North Wales. "
        "It does not retrain during app use. Instead, it loads a saved model and saved input features."
    )

    st.subheader("How the data was prepared")
    st.write("1. Police.uk street crime records were collected from monthly CSV files.")
    st.write("2. Individual incidents were aggregated into monthly crime counts for each LSOA.")
    st.write("3. Missing LSOA-month combinations were filled with zero.")
    st.write("4. Historical features were created from past crime counts.")
    st.write("5. A trained model was selected and exported for deployment.")

    st.subheader("Feature explanation")
    explain_df = pd.DataFrame({
        "Feature": [
            "lag_1",
            "lag_2",
            "lag_3",
            "roll_mean_3",
            "roll_mean_6",
            "month_num"
        ],
        "Meaning": [
            "Crime count in the previous month",
            "Crime count two months earlier",
            "Crime count three months earlier",
            "Average crime count over the previous 3 months",
            "Average crime count over the previous 6 months",
            "Numeric month of the year"
        ]
    })
    st.dataframe(explain_df, use_container_width=True, hide_index=True)

    st.subheader("Model output")
    st.write(
        "The model returns a predicted count for the next month. "
        "This value is shown together with a simple baseline based on the previous month."
    )

    st.subheader("Deployment artifacts")
    artifact_df = pd.DataFrame({
        "File": [
            "model.joblib",
            "latest_features.csv",
            "meta.json",
            "results.csv",
            "lsoa_lookup.csv",
            "lsoa_centroids.csv"
        ],
        "Purpose": [
            "Saved trained model pipeline",
            "Latest input rows used for prediction",
            "Project metadata and selected feature names",
            "Model comparison results",
            "Lookup information for LSOA names and codes",
            "Approximate map point for each LSOA"
        ]
    })
    st.dataframe(artifact_df, use_container_width=True, hide_index=True)

