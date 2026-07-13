import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

import config

from smoglens.config import ACTIVITY_LIMITS, HORIZONS, SHIBUYA_LAT, SHIBUYA_LON
from smoglens.data import DataProcessor
from smoglens.features import FeatureGenerator
from smoglens.inference import SimplePredictor
from smoglens.spatial import SpatialImputer
from smoglens.visualization import create_activity_prediction_chart

st.set_page_config(page_title="SmogLens", layout="wide")

TOKYO_TZ = ZoneInfo("Asia/Tokyo")


@st.cache_resource
def load_components():
    data_processor = DataProcessor(data_path=config.DATA_PATH, cache_dir=config.DATA_DIR)
    data_processor.load_data()
    data_processor.find_nearby_hexagons()

    feature_generator = FeatureGenerator()
    model_predictor = SimplePredictor(model_dir=config.MODEL_DIR)
    spatial_imputer = SpatialImputer(data_processor)

    return data_processor, feature_generator, model_predictor, spatial_imputer


data_processor, feature_generator, model_predictor, spatial_imputer = load_components()

st.title("SmogLens")
st.markdown("Real-time PM2.5 predictions for outdoor activities in Shibuya")

with st.sidebar:
    st.subheader("Select Date & Time")

    use_current_time = st.checkbox("Use current time", value=True)

    if use_current_time:
        selected_datetime = datetime.now(TOKYO_TZ)
    else:
        selected_date = st.date_input(
            "Date",
            min_value=datetime(2023, 7, 14),
            max_value=datetime(2025, 7, 26),
            value=datetime(2024, 6, 1),
        )

        selected_time = st.time_input("Time", value=datetime.now().time())

        selected_datetime = datetime.combine(selected_date, selected_time, tzinfo=TOKYO_TZ)

    st.info(f"Selected: {selected_datetime.strftime('%Y-%m-%d %H:%M')} JST")

col1, col2 = st.columns([2, 1])

with col1:
    map_data = pd.DataFrame({"lat": [SHIBUYA_LAT], "lon": [SHIBUYA_LON]})
    st.map(map_data, zoom=13, use_container_width=True)
    st.caption("Shibuya Station - 500m radius coverage")

with col2:
    st.metric("Tokyo Time", selected_datetime.strftime("%H:%M"))
    st.caption(selected_datetime.strftime("%A, %B %d, %Y"))

    current_data = data_processor.get_data_at_time(selected_datetime)

    if current_data is not None:
        current_pm25 = (
            current_data["pm25_ugm3_mean"]
            if "pm25_ugm3_mean" in current_data
            else current_data.get("pm25", np.nan)
        )

        if not pd.isna(current_pm25):
            st.metric("Current PM2.5", f"{current_pm25:.1f} μg/m³")
        else:
            st.metric("Current PM2.5", "No data (using imputation)")
    else:
        st.metric("Current PM2.5", "Data pending...")

st.divider()

activities = {
    "Running": ("Running", "High intensity exercise"),
    "Walking with Baby": ("Baby", "Infants are more vulnerable"),
    "Walking": ("Walking", "Moderate outdoor activity"),
    "Sitting Outside": ("Sitting", "Low intensity, stationary"),
    "Driving": ("Car", "Filtered air in vehicle"),
}

activity = st.selectbox(
    "Select Your Activity:",
    options=list(activities.keys()),
    index=2,
    help="Different activities have different safe PM2.5 thresholds based on exposure and breathing rate",
)

activity_key = activities[activity][0]
threshold = ACTIVITY_LIMITS[activity_key]

st.caption(f"Threshold: {threshold} μg/m³ - {activities[activity][1]}")

st.divider()
st.subheader("6-Hour PM2.5 Forecast")

with st.spinner("Generating predictions..."):
    try:
        historical_data = data_processor.get_historical_window(selected_datetime, hours_back=168)

        if historical_data is None or len(historical_data) < 24:
            st.warning("Insufficient historical data. Using spatial imputation from nearby areas...")
            historical_data = spatial_imputer.impute_dataframe(
                pd.DataFrame([current_data]),
                columns=["pm25_ugm3_mean", "temperature_c_mean", "humidity_pct_mean"],
            )

        features_df = feature_generator.generate_features(historical_data)

        if len(features_df) > 0:
            last_features = features_df.iloc[-1:].copy()

            predictions, confidence_intervals = model_predictor.predict_all(last_features)

            predictions_df = pd.DataFrame(
                {
                    "time": [selected_datetime + timedelta(hours=int(horizon[:-1])) for horizon in HORIZONS],
                    "pm25": [predictions[horizon] for horizon in HORIZONS],
                    "lower": [confidence_intervals[horizon][0] for horizon in HORIZONS],
                    "upper": [confidence_intervals[horizon][1] for horizon in HORIZONS],
                }
            )

            fig = create_activity_prediction_chart(predictions_df, activity_threshold=threshold)

            st.plotly_chart(fig, use_container_width=True)

            safe_hours = predictions_df[predictions_df["pm25"] < threshold]

            col1, col2 = st.columns(2)
            col1.metric("Safe Hours", f"{len(safe_hours)}/6")

            if len(safe_hours) > 0:
                best_hour = safe_hours.loc[safe_hours["pm25"].idxmin()]
                col2.metric("Best Time", best_hour["time"].strftime("%H:%M"))
                st.success(f"Good conditions for {activity} - {len(safe_hours)}/6 hours within safe limits")
            else:
                st.warning(f"Not recommended for {activity} in the next 6 hours")

            with st.expander("Prediction Details"):
                st.dataframe(
                    predictions_df.rename(
                        columns={
                            "time": "Time",
                            "pm25": "PM2.5 (μg/m³)",
                            "lower": "95% CI Lower",
                            "upper": "95% CI Upper",
                        }
                    ).set_index("Time")
                )

    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        st.info("Please try selecting a different date/time within the available data range.")

st.divider()
st.caption("Data source: PM2.5 measurements from OpenAQ, weather from OpenMeteo, traffic from JARTIC")
st.caption("Models: Ensemble of LightGBM, XGBoost, and CatBoost trained on 2023-2025 data")
