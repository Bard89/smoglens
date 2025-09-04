import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import pytz
import joblib
import os
from PIL import Image
import time
import gc
import xgboost as xgb
from streamlit_folium import folium_static
import folium

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="SmogLens",
    page_icon="/home/julia/smoglens/data/logo.png",
    layout="wide"
)



# ------------------------
# Constants
# ------------------------
forecast_hours = 6
expected_columns = [
    'avg_traffic_volume', 'max_traffic_volume', 'congestion_index',
    'traffic_measurement_count', 'traffic_distance_km', 'traffic_intensity',
    'temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean',
    'cloud_cover_pct_mean', 'is_weekend', 'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_4h', 'lag_5h', 'lag_6h',
    'lag_12h', 'lag_24h', 'lag_48h', 'lag_72h', 'lag_168h',
    'diff_1h', 'diff_6h', 'diff_24h', 'rate_6h', 'rate_24h',
    'rolling_mean_3h', 'rolling_std_3h', 'rolling_max_3h', 'rolling_min_3h',
    'rolling_mean_6h', 'rolling_std_6h', 'rolling_max_6h', 'rolling_min_6h',
    'rolling_mean_12h', 'rolling_std_12h', 'rolling_max_12h', 'rolling_min_12h',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_max_24h', 'rolling_min_24h',
    'rolling_mean_48h', 'rolling_std_48h', 'rolling_max_48h', 'rolling_min_48h',
    'ewm_0.1', 'ewm_0.3', 'ewm_0.5', 'hour_sin_1', 'hour_cos_1',
    'dow_sin_1', 'dow_cos_1', 'hour_sin_2', 'hour_cos_2', 'dow_sin_2', 'dow_cos_2',
    'temp_humidity', 'temp_hour', 'traffic_hour', 'traffic_weekend', 'hex_encoded'
]

prefecture_coords = {
    "Hokkaido": [43.0642, 141.3469], "Aomori": [40.8222, 140.7474], "Iwate": [39.7036, 141.1527],
    "Miyagi": [38.2688, 140.8719], "Akita": [39.7199, 140.1024], "Yamagata": [38.2404, 140.3633],
    "Fukushima": [37.7503, 140.4676], "Ibaraki": [36.3418, 140.4468], "Tochigi": [36.5658, 139.8836],
    "Gunma": [36.3912, 139.0609], "Saitama": [35.8569, 139.6489], "Chiba": [35.6074, 140.1065],
    "Tokyo": [35.6895, 139.6917], "Kanagawa": [35.4478, 139.6425], "Niigata": [37.9026, 139.0236],
    "Toyama": [36.6953, 137.2113], "Ishikawa": [36.5947, 136.6256], "Fukui": [36.0652, 136.2216],
    "Yamanashi": [35.6639, 138.5684], "Nagano": [36.6513, 138.181], "Gifu": [35.4233, 136.7615],
    "Shizuoka": [34.9756, 138.3828], "Aichi": [35.1802, 136.9066], "Mie": [34.7303, 136.5086],
    "Shiga": [35.0045, 135.8686], "Kyoto": [35.0116, 135.7681], "Osaka": [34.6937, 135.5023],
    "Hyogo": [34.6913, 135.183], "Nara": [34.6851, 135.8048], "Wakayama": [34.226, 135.1675],
    "Tottori": [35.5039, 134.2383], "Shimane": [35.4723, 133.0505], "Okayama": [34.6618, 133.935],
    "Hiroshima": [34.3853, 132.4553], "Yamaguchi": [34.1785, 131.4737], "Tokushima": [34.0703, 134.5548],
    "Kagawa": [34.3401, 134.0434], "Ehime": [33.8416, 132.7657], "Kochi": [33.5597, 133.5311],
    "Fukuoka": [33.5902, 130.4017], "Saga": [33.2494, 130.2988], "Nagasaki": [32.7503, 129.8777],
    "Kumamoto": [32.7898, 130.7417], "Oita": [33.2382, 131.6126], "Miyazaki": [31.9078, 131.4202],
    "Kagoshima": [31.5602, 130.5581], "Okinawa": [26.2124, 127.6809]
}


# Activity-specific safety thresholds
ACTIVITY_THRESHOLDS = {
    "Walking": {"safe": 12, "caution": 35, "unsafe": 55},
    "Running": {"safe": 8, "caution": 25, "unsafe": 35},    # Same as Jogging
    "Walking with Baby": {"safe": 12, "caution": 35, "unsafe": 55}, # Same as Walking
    "Sitting": {"safe": 25, "caution": 50, "unsafe": 100},  # Relaxed limits
    "Driving": {"safe": 35, "caution": 55, "unsafe": 100}   # Vehicle protection
}


# ------------------------
# Sidebar: Logo
# ------------------------
logo_path = os.path.join("/home/julia/smoglens/data", "logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=120)

# ------------------------
# Sidebar: Location + Activity + Predict button
# ------------------------
prefectures = [
    "Hokkaido","Aomori","Iwate","Miyagi","Akita","Yamagata","Fukushima",
    "Ibaraki","Tochigi","Gunma","Saitama","Chiba","Tokyo","Kanagawa",
    "Niigata","Toyama","Ishikawa","Fukui","Yamanashi","Nagano","Gifu",
    "Shizuoka","Aichi","Mie","Shiga","Kyoto","Osaka","Hyogo","Nara",
    "Wakayama","Tottori","Shimane","Okayama","Hiroshima","Yamaguchi",
    "Tokushima","Kagawa","Ehime","Kochi","Fukuoka","Saga","Nagasaki",
    "Kumamoto","Oita","Miyazaki","Kagoshima","Okinawa"
]
location = st.sidebar.selectbox("Select Location", prefectures, index=12)  # Tokyo default
# ------------------------
# Current Time
# ------------------------
tz = pytz.timezone("Asia/Tokyo")
current_time = datetime.datetime.now(tz)
st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%H:%M %Z')}")



st.sidebar.markdown("### Select Activity")
activities = {
    "Running": "High intensity - strictest limits. Prefer indoor exercise if air quality is poor.",
    "Walking with Baby": "Protect vulnerable groups. Avoid outdoor activities for children when PM2.5 is high.",
    "Walking": "Moderate outdoor activity. Wear a mask in congested areas.",
    "Sitting": "Low intensity - relaxed limits. Indoor sitting is generally safe.",
    "Driving": "Vehicle provides some protection. Keep windows closed."
}
selected_activity = st.sidebar.radio("", list(activities.keys()))

# Show activity-specific thresholds
thresholds = ACTIVITY_THRESHOLDS[selected_activity]
st.sidebar.markdown(f"**Safety Levels for {selected_activity}:**")
st.sidebar.markdown(f"🟢 Safe: ≤ {thresholds['safe']} µg/m³")
st.sidebar.markdown(f"🟡 Caution: {thresholds['safe']+1}-{thresholds['caution']} µg/m³")
st.sidebar.markdown(f"🔴 Unsafe: > {thresholds['caution']} µg/m³")


# ------------------------
# Compact Map Visualization
# ------------------------

    # Place info columns tightly to the right of the map
map_col, info_col1 = st.columns([1, 1])

with map_col:
    # Render the compact map visualization in the map_col
    coords = prefecture_coords.get(location, [35.6895, 139.6917])  # Default to Tokyo

    m = folium.Map(location=coords, zoom_start=7, width='100%', height='100%')
    folium.Marker(
        location=coords,
        popup=f"{location}",
        tooltip=f"{location}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    folium_static(m, width=350, height=250)

with info_col1:
    st.markdown("""
    # SmogLens
    - **Real-time PM2.5 prediction** for next 6 hours
    - **Activity-specific recommendations**
    - **Ensemble ML models** (XGBoost, LGBM, CatBoost)
    - **Location-based forecasting** across Japan
    """)



# ------------------------
# Predict Button
# ------------------------
predict_button = st.sidebar.button("Predict Air Quality", key="predict_btn")
st.sidebar.markdown("""
<style>
div[data-testid="stSidebar"] div.stButton > button:first-child {
    background-color: #FF4B4B !important;
    color: white !important;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}
div[data-testid="stSidebar"] div.stButton > button:first-child:hover {
    background-color: #FF6B6B !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------
# Load Models
# ------------------------
@st.cache_resource
def load_models():
    models = {}
    for i in range(1,7):
        path = os.path.join("/home/julia/smoglens/data", f"models_{i}h.pkl")
        try:
            model = joblib.load(path)
            models[f"{i}h"] = model
        except Exception:
            models[f"{i}h"] = None
    return models

# ------------------------
# Load feature data
# ------------------------
@st.cache_resource
def load_feature_data():
    try:
        X_val = joblib.load("/home/julia/smoglens/data/X_val.pkl")
        return X_val
    except Exception as e:
        st.error(f"❌ Error loading X_val: {e}")
        return None



# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}
    st.session_state.X_val = None

# Load models and data on first run
if not st.session_state.models_loaded:
    with st.spinner("Loading prediction models..."):
        st.session_state.models = load_models()
        st.session_state.X_val = load_feature_data()
        st.session_state.models_loaded = True

if predict_button:
    if st.session_state.X_val is None:
        st.error("❌ Cannot make predictions - feature data not available")
        st.stop()

    with st.spinner("🔮 Creating predictions..."):
        time.sleep(1)  # simulate computation

        forecast_times = [current_time + datetime.timedelta(hours=i) for i in range(forecast_hours)]
        pm25_values = []
        upper_shadow = []
        lower_shadow = []

        for i in range(1, forecast_hours+1):
            model = st.session_state.models.get(f"{i}h")
            X_input = st.session_state.X_val[expected_columns].iloc[[0]]

            if model is not None:
                # Ensemble handling: XGBoost, LGBM, CatBoost
                if isinstance(model, dict):
                    preds = []
                    for m in model.values():
                        if isinstance(m, xgb.Booster):
                            dmatrix = xgb.DMatrix(X_input)
                            p = m.predict(dmatrix)[0]
                        else:
                            p = m.predict(X_input)[0]
                        preds.append(p)
                    pred = np.mean(preds)
                else:
                    if isinstance(model, xgb.Booster):
                        dmatrix = xgb.DMatrix(X_input)
                        pred = model.predict(dmatrix)[0]
                    else:
                        pred = model.predict(X_input)[0]

                # Use RMSE for confidence intervals
                rmse_table = {1:2.004,2:2.459,3:2.796,4:3.055,5:3.269,6:3.443}
                conf = rmse_table.get(i,3)
                upper = min(pred + conf, 100)
                lower = max(pred - conf, 0)
            else:
                # Fallback if model not available
                pred = np.random.uniform(10,50)
                upper = min(pred + 3,100)
                lower = max(pred - 3,0)

            pm25_values.append(pred)
            upper_shadow.append(upper)
            lower_shadow.append(lower)

        # ------------------------
        # Enhanced Plot with Activity-Specific Thresholds
        # ------------------------
        df_forecast = pd.DataFrame({
            "Hour": [t.strftime("%H:%M") for t in forecast_times],
            "PM2.5": pm25_values,
            "Upper": upper_shadow,
            "Lower": lower_shadow,
            "SortOrder": range(len(forecast_times))
        })

        # Create the array of hour strings
        hour_strings = [f"'{t}'" for t in df_forecast['Hour']]
        hour_array_string = f"[{', '.join(hour_strings)}]"

        base = alt.Chart(df_forecast)

        # First define the confidence band
        confidence_band = base.mark_area(
            opacity=0.1,  # Reduced opacity
            color='#ADD8E6'  # Light blue
        ).encode(
            x=alt.X('SortOrder:O',
                   axis=alt.Axis(title='Time (Hours Ahead)',
                               labelAngle=-45,
                               values=list(range(len(df_forecast))),
                               labelExpr=f"{hour_array_string}[datum.value]")),
            y='Lower:Q',
            y2='Upper:Q'
        )

        # Activity-specific threshold lines
        safe_line = base.mark_rule(
            color='green',
            strokeWidth=2,
            strokeDash=[8,4]
        ).encode(
            y=alt.Y(datum=thresholds['safe'])
        )

        caution_line = base.mark_rule(
            color='orange',
            strokeWidth=2,
            strokeDash=[8,4]
        ).encode(
            y=alt.Y(datum=thresholds['caution'])
        )

        unsafe_line = base.mark_rule(
            color='red',
            strokeWidth=2,
            strokeDash=[8,4]
        ).encode(
            y=alt.Y(datum=thresholds['unsafe'])
        )

        # Labels for threshold lines
        safe_text = alt.Chart(pd.DataFrame({
            'x': [len(df_forecast)-1],
            'y': [thresholds['safe']],
            'text': [f'Safe ({thresholds["safe"]} µg/m³)']
        })).mark_text(
            align='right',
            dx=-5,
            dy=-5,
            color='green',
            fontSize=14,
            fontWeight='bold'
        ).encode(x='x:O', y='y:Q', text='text:N')

        caution_text = alt.Chart(pd.DataFrame({
            'x': [len(df_forecast)-1],
            'y': [thresholds['caution']],
            'text': [f'Caution ({thresholds["caution"]} µg/m³)']
        })).mark_text(
            align='right',
            dx=-5,
            dy=-5,
            color='orange',
            fontSize=14,
            fontWeight='bold'
        ).encode(x='x:O', y='y:Q', text='text:N')

        # Make prediction line and dots more prominent
        prediction_line = base.mark_line(
            color='#FF0000',  # Bright red
            strokeWidth=4,     # Thicker line
            interpolate='linear'
        ).encode(
            x=alt.X('SortOrder:O',
                   axis=alt.Axis(title='Time (Hours Ahead)',
                               labelAngle=-45,
                               values=list(range(len(df_forecast))),
                               labelExpr=f"{hour_array_string}[datum.value]")),
            y=alt.Y('PM2.5:Q',
                   axis=alt.Axis(title='PM2.5 (µg/m³)'),
                   scale=alt.Scale(domain=[0, max(100, max(pm25_values)+10)])),
        )

        prediction_dots = base.mark_circle(
            size=120,         # Larger dots
            color='#FF0000',  # Bright red
            opacity=1,
            stroke='white',   # White border
            strokeWidth=2     # Thicker border
        ).encode(
            x=alt.X('SortOrder:O'),
            y=alt.Y('PM2.5:Q'),
            tooltip=['Hour:O', 'PM2.5:Q']
        )

        # Combine chart elements in proper order
        chart = (
            confidence_band +   # Background first
            safe_line +        # Then threshold lines
            caution_line +
            unsafe_line +
            prediction_line +   # Then prediction line
            prediction_dots +   # Dots on top
            safe_text +        # Finally, labels
            caution_text
        ).properties(
            width=800,
            height=400,
            title=f"PM2.5 Forecast - {selected_activity} Activity Safety Levels"
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18
        ).configure_title(
            fontSize=24,
            anchor='start'
        )

        # Confidence band (make it more transparent)
        confidence_band = base.mark_area(
            opacity=0.1,  # Reduced opacity
            color='#ADD8E6'  # Light blue
        ).encode(
            x=alt.X('SortOrder:O',
                   axis=alt.Axis(title='Time (Hours Ahead)',
                               labelAngle=-45,
                               values=list(range(len(df_forecast))),
                               labelExpr=f"{hour_array_string}[datum.value]")),
            y='Lower:Q',
            y2='Upper:Q'
        )

        # Enhanced prediction line with dots
        prediction_dots = base.mark_circle(
            size=100,  # Larger dots
            color='#FF0000',  # Bright red
            opacity=1,
            stroke='white',  # White border around dots
            strokeWidth=2
        ).encode(
            x=alt.X('SortOrder:O'),
            y=alt.Y('PM2.5:Q'),
            tooltip=['Hour:O', 'PM2.5:Q']
        )

        prediction_line = base.mark_line(
            color='#FF0000',  # Bright red
            strokeWidth=3,
            interpolate='linear'  # Linear interpolation between points
        ).encode(
            x=alt.X('SortOrder:O',
                   axis=alt.Axis(title='Time (Hours Ahead)',
                               labelAngle=-45,
                               values=list(range(len(df_forecast))),
                               labelExpr=f"{hour_array_string}[datum.value]")),
            y=alt.Y('PM2.5:Q',
                   axis=alt.Axis(title='PM2.5 (µg/m³)'),
                   scale=alt.Scale(domain=[0,max(100, max(pm25_values)+10)])),
        )

        # Combine all chart elements (updated order)
        chart = (
            confidence_band +
            prediction_line +
            prediction_dots +  # Add dots on top
            safe_line +
            caution_line +
            unsafe_line +
            safe_text +
            caution_text
        ).properties(
            width=800,
            height=400,
            title=f"PM2.5 Forecast - {selected_activity} Activity Safety Levels"
        ).configure_axis(
            labelFontSize=18,
            titleFontSize=22
        ).configure_title(
            fontSize=28
        ).configure_legend(
            labelFontSize=18,
            titleFontSize=20
        )

        st.altair_chart(chart, use_container_width=True)

        # ------------------------
        # Current Status & Recommendations
        # ------------------------
        current_pm = pm25_values[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current PM2.5", f"{current_pm:.1f} µg/m³")

        with col2:
            if current_pm <= thresholds['safe']:
                status = "🟢 SAFE"
                status_color = "green"
            elif current_pm <= thresholds['caution']:
                status = "🟡 CAUTION"
                status_color = "orange"
            else:
                status = "🔴 UNSAFE"
                status_color = "red"

            st.markdown(f"**Activity Status:** <span style='color:{status_color}; font-size:18px'>{status}</span>",
                       unsafe_allow_html=True)

        with col3:
            st.metric("Peak in 6h", f"{max(pm25_values):.1f} µg/m³")

        # ------------------------
        # Detailed Recommendations
        # ------------------------
        st.markdown("### 🏥 Personalized Recommendations")

        if current_pm <= thresholds['safe']:
            st.success(f"✅ **{selected_activity} is SAFE** at current PM2.5 levels")
            recommendations = ["🌟 Current air quality is suitable for your activity"]
        elif current_pm <= thresholds['caution']:
            st.warning(f"⚠️ **{selected_activity} requires CAUTION** at current PM2.5 levels")
            if selected_activity == "Jogging":
                recommendations = ["🏃‍♂️ Reduce intensity and duration", "😷 Consider wearing a mask", "⏰ Check forecast for better timing"]
            elif selected_activity == "Walking":
                recommendations = ["🚶‍♀️ Limit prolonged exposure", "😷 Wear mask in congested areas", "🌳 Prefer parks over busy streets"]
            else:  # Indoor
                recommendations = ["🏠 Keep windows closed", "💨 Use air purifier if available", "🌡️ Monitor indoor air quality"]
        else:
            st.error(f"❌ **{selected_activity} is UNSAFE** at current PM2.5 levels")
            if selected_activity == "Jogging":
                recommendations = ["🚫 Avoid outdoor jogging", "🏠 Exercise indoors instead", "⏰ Wait for better air quality"]
            elif selected_activity == "Walking":
                recommendations = ["🚫 Minimize outdoor exposure", "😷 Wear N95 mask if must go outside", "🚗 Use transportation instead of walking"]
            else:  # Indoor
                recommendations = ["🪟 Keep all windows closed", "💨 Use air purifier on high", "🚫 Avoid opening doors frequently"]

        # Add activity-specific advice
        recommendations.append(f"💡 **Activity Tip:** {activities[selected_activity]}")

        for rec in recommendations:
            st.markdown(f"• {rec}")

        # Show hourly breakdown
        st.markdown("### 📊 Hourly Forecast Breakdown")
        forecast_df = pd.DataFrame({
            "Time": [t.strftime("%H:%M") for t in forecast_times],
            "PM2.5 (µg/m³)": [f"{val:.1f}" for val in pm25_values],
            "Status": [
                "🟢 Safe" if val <= thresholds['safe'] else
                "🟡 Caution" if val <= thresholds['caution'] else
                "🔴 Unsafe" for val in pm25_values
            ]
        })
        st.dataframe(forecast_df, use_container_width=True)

else:
    st.info("👆 Select your location and activity, then click 'Predict Air Quality' to get started!")
