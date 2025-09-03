from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import pytz
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.data_processor import DataProcessor
from utils.feature_generator import FeatureGenerator
from utils.inference import SimplePredictor
from utils.spatial_imputer import SpatialImputer
from utils.visualization import create_activity_prediction_chart
import config

st.set_page_config(
    page_title="SmogLens",
    page_icon="🏃",
    layout="wide"
)

@st.cache_resource
def load_components():
    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.find_nearby_hexagons()
    
    feature_generator = FeatureGenerator()
    model_predictor = SimplePredictor()
    spatial_imputer = SpatialImputer(data_processor)
    
    return data_processor, feature_generator, model_predictor, spatial_imputer

data_processor, feature_generator, model_predictor, spatial_imputer = load_components()

st.title("SmogLens")
st.markdown("Real-time PM2.5 predictions for outdoor activities in Shibuya")

with st.sidebar:
    st.subheader("📅 Select Date & Time")
    
    use_current_time = st.checkbox("Use current time", value=True)
    
    if use_current_time:
        tokyo_tz = pytz.timezone('Asia/Tokyo')
        selected_datetime = datetime.now(tokyo_tz)
    else:
        selected_date = st.date_input(
            "Date",
            min_value=datetime(2023, 7, 14),
            max_value=datetime(2025, 7, 26),
            value=datetime(2024, 6, 1)
        )
        
        selected_time = st.time_input(
            "Time",
            value=datetime.now().time()
        )
        
        selected_datetime = datetime.combine(selected_date, selected_time)
        tokyo_tz = pytz.timezone('Asia/Tokyo')
        selected_datetime = tokyo_tz.localize(selected_datetime)
    
    st.info(f"Selected: {selected_datetime.strftime('%Y-%m-%d %H:%M')} JST")

col1, col2 = st.columns([2, 1])

with col1:
    m = folium.Map(
        location=[config.SHIBUYA_LAT, config.SHIBUYA_LON],
        zoom_start=13,
        tiles='CartoDB positron',
        width='100%',
        height='100%'
    )
    
    folium.Marker(
        [config.SHIBUYA_LAT, config.SHIBUYA_LON],
        popup='Shibuya Station, Tokyo',
        tooltip='📍 PM2.5 Monitoring Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    folium.Circle(
        location=[config.SHIBUYA_LAT, config.SHIBUYA_LON],
        radius=500,
        color='#6366F1',
        fill=True,
        fillColor='#6366F1',
        fillOpacity=0.2,
        popup='500m radius coverage area'
    ).add_to(m)
    
    st_folium(m, height=250, returned_objects=[], key='tokyo_map')

with col2:
    st.metric("🕐 Tokyo Time", selected_datetime.strftime("%H:%M"))
    st.caption(selected_datetime.strftime("%A, %B %d, %Y"))
    
    current_data = data_processor.get_data_at_time(selected_datetime)
    
    if current_data is not None:
        current_pm25 = current_data['pm25_ugm3_mean'] if 'pm25_ugm3_mean' in current_data else current_data.get('pm25', np.nan)
        
        if not pd.isna(current_pm25):
            COLOR = "🟢" if current_pm25 < 12 else "🟡" if current_pm25 < 35 else "🟠" if current_pm25 < 55 else "🔴"
            st.metric(f"{COLOR} Current PM2.5", f"{current_pm25:.1f} μg/m³")
        else:
            st.metric("Current PM2.5", "No data (using imputation)")
    else:
        st.metric("Current PM2.5", "Data pending...")

st.divider()

activities = {
    '🏃 Running': ('Running', 'High intensity exercise'),
    '👶 Walking with Baby': ('Baby', 'Infants are more vulnerable'),
    '🚶 Walking': ('Walking', 'Moderate outdoor activity'),
    '🪑 Sitting Outside': ('Sitting', 'Low intensity, stationary'),
    '🚗 Driving': ('Car', 'Filtered air in vehicle')
}

activity = st.selectbox(
    "Select Your Activity:",
    options=list(activities.keys()),
    index=2,
    help="Different activities have different safe PM2.5 thresholds based on exposure and breathing rate"
)

activity_key = activities[activity][0]
threshold = config.ACTIVITY_LIMITS[activity_key]

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
                columns=['pm25_ugm3_mean', 'temperature_c_mean', 'humidity_pct_mean']
            )
        
        features_df = feature_generator.generate_features(historical_data)
        
        if len(features_df) > 0:
            last_features = features_df.iloc[-1:].copy()
            
            predictions, confidence_intervals = model_predictor.predict_all(last_features)
            
            prediction_times = []
            pm25_values = []
            lower_bounds = []
            upper_bounds = []
            
            for i, horizon in enumerate(config.HORIZONS):
                hours_ahead = int(horizon[:-1])
                pred_time = selected_datetime + timedelta(hours=hours_ahead)
                prediction_times.append(pred_time)
                pm25_values.append(predictions[horizon])
                lower_bounds.append(confidence_intervals[horizon][0])
                upper_bounds.append(confidence_intervals[horizon][1])
            
            predictions_df = pd.DataFrame({
                'time': prediction_times,
                'pm25': pm25_values,
                'lower': lower_bounds,
                'upper': upper_bounds
            })
            
            fig = create_activity_prediction_chart(
                predictions_df,
                activity_threshold=threshold,
                activity_name=activity_key
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            safe_hours = predictions_df[predictions_df['pm25'] < threshold]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Safe Hours", f"{len(safe_hours)}/6")
            with col2:
                if len(safe_hours) > 0:
                    best_hour = safe_hours.loc[safe_hours['pm25'].idxmin()]
                    st.metric("Best Time", best_hour['time'].strftime("%H:%M"))
            
            if len(safe_hours) > 0:
                st.success(f"✅ Good conditions for {activity} - "
                          f"{len(safe_hours)}/6 hours within safe limits")
            else:
                st.warning(f"⚠️ Not recommended for {activity} in the next 6 hours")
            
            with st.expander("📊 Prediction Details"):
                st.dataframe(
                    predictions_df.rename(columns={
                        'time': 'Time',
                        'pm25': 'PM2.5 (μg/m³)',
                        'lower': '95% CI Lower',
                        'upper': '95% CI Upper'
                    }).set_index('Time')
                )
            
            with st.expander("🗺️ Spatial Coverage"):
                coverage = spatial_imputer.get_spatial_coverage(selected_datetime)
                st.metric("Data Coverage", f"{coverage['coverage_percentage']:.1f}%")
                st.caption(f"{coverage['hexagons_with_data']}/{coverage['total_hexagons']} hexagons with data")
                if coverage['average_distance'] > 0:
                    st.caption(f"Average distance: {coverage['average_distance']:.1f} km")
                    
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        st.info("Please try selecting a different date/time within the available data range.")

st.divider()
st.caption("Data source: PM2.5 measurements from OpenAQ, weather from OpenMeteo, traffic from JARTIC")
st.caption("Models: Ensemble of LightGBM, XGBoost, and CatBoost trained on 2023-2025 data")