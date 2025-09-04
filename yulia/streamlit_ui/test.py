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

# ------------------------
# Sidebar: Logo
# ------------------------
logo_path = os.path.join("data", "logo.png")
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
location = st.sidebar.selectbox("Select Location", prefectures, index=11)

st.sidebar.markdown("### Select Activity")
activities = {
    "Indoor": "Close windows or turn on air purifier",
    "Walking": "Wear mask or good to go outside",
    "Jogging": "Better later, max 30 mins outside"
}
selected_activity = st.sidebar.radio("", list(activities.keys()))

# ------------------------
# Black Predict Button
# ------------------------
predict_button = st.sidebar.button("Predict", key="predict_btn")
st.sidebar.markdown("""
<style>
div[data-testid="stSidebar"] div.stButton > button:first-child {
    background-color: black !important;
    color: white !important;
    height: 3em;
    width: 100%;
    border-radius: 5px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Current Time
# ------------------------
tz = pytz.timezone("Asia/Tokyo")
current_time = datetime.datetime.now(tz)
st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%H:%M %Z')}")

# ------------------------
# Load Models
# ------------------------
@st.cache_resource
def load_models():
    models = {}
    for i in range(1,7):
        path = os.path.join("data", f"models_{i}h.pkl")
        try:
            models[f"{i}h"] = joblib.load(path)
        except:
            models[f"{i}h"] = None
    return models

models = load_models()

# ------------------------
# Load feature data
# ------------------------
X_val = joblib.load("/home/julia/smoglens/data/X_val.pkl")   # <--- replace with actual path
X_test = joblib.load("/home/julia/smoglens/data/X_test.pkl") # <--- replace with actual path

# ------------------------
# Main App: Wait for Predict
# ------------------------
if predict_button:
    with st.spinner("Creating predictions..."):
        time.sleep(1)  # simulate computation

        forecast_times = [current_time + datetime.timedelta(hours=i) for i in range(forecast_hours)]
        pm25_values = []
        upper_shadow = []
        lower_shadow = []

        for i in range(1, forecast_hours+1):
            model = models.get(f"{i}h")
            X_input = X_val[expected_columns].iloc[[0]]  # <--- replace with proper current row/input
            if model is not None:
                # ------------------------
                # Ensemble handling: XGBoost, LGBM, CatBoost
                # ------------------------
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

                # ------------------------
                # Use RMSE for shadows (replace with real RMSE per horizon)
                rmse_table = {1:2.004,2:2.459,3:2.796,4:3.055,5:3.269,6:3.443}
                conf = rmse_table.get(i,3)
                upper = min(pred + conf, 50)
                lower = max(pred - conf, 0)
            else:
                pred = np.random.uniform(10,50)
                upper = min(pred + 3,50)
                lower = max(pred - 3,0)

            pm25_values.append(pred)
            upper_shadow.append(upper)
            lower_shadow.append(lower)

        # ------------------------
        # Plot Forecast
        # ------------------------
        df_forecast = pd.DataFrame({
            "Hour": [t.strftime("%H:%M") for t in forecast_times],
            "PM2.5": pm25_values,
            "Upper": upper_shadow,
            "Lower": lower_shadow
        })

        base = alt.Chart(df_forecast)
        confidence_band = base.mark_area(opacity=0.2, color='lightblue').encode(
            x=alt.X('Hour:O', axis=alt.Axis(title='Time', labelAngle=-45)),
            y='Lower:Q',
            y2='Upper:Q'
        )

        prediction_line = base.mark_line(color='#3498DB', strokeWidth=3, point=True).encode(
            x='Hour:O',
            y=alt.Y('PM2.5:Q', axis=alt.Axis(title='PM2.5 (µg/m³)'), scale=alt.Scale(domain=[0,50])),
            tooltip=['Hour:O','PM2.5:Q']
        )

        good_line = alt.Chart(pd.DataFrame({'y':[12]})).mark_rule(color='green', strokeDash=[5,5]).encode(y='y:Q')
        moderate_line = alt.Chart(pd.DataFrame({'y':[35]})).mark_rule(color='orange', strokeDash=[5,5]).encode(y='y:Q')
        unhealthy_line = alt.Chart(pd.DataFrame({'y':[50]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')

        chart = (confidence_band + prediction_line + good_line + moderate_line + unhealthy_line).properties(
            width=700, height=350, title="PM2.5 Forecast (Next 6 Hours)"
        )

        st.altair_chart(chart, use_container_width=True)

        # ------------------------
        # Current PM2.5
        # ------------------------
        current_pm = pm25_values[0]
        st.markdown(f"**Current PM2.5:** {current_pm:.1f} µg/m³")

        # ------------------------
        # Health Advisory
        # ------------------------
        st.markdown("### 🏥 Air Quality & Recommendations")
        if current_pm <= 12:
            health_category = "Good ✅"
            alert_color = "green"
            recommendations = ["🌟 Air is healthy. Minimal precautions needed."]
        elif current_pm <= 35:
            health_category = "Moderate ⚠️"
            alert_color = "orange"
            recommendations = ["⚡ Reduce outdoor activity intensity","🚶 Limit prolonged exposure outdoors"]
        else:
            health_category = "Unhealthy ❌"
            alert_color = "red"
            recommendations = ["⚠️ Avoid outdoor activities","🏠 Stay indoors and use air purifier","😷 Wear mask if you must go outside"]

        # Add activity advice
        recommendations.append(f"💡 Activity advice: {activities[selected_activity]}")

        st.markdown(f"**Air Quality:** <span style='color:{alert_color}'>{health_category}</span>", unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f"• {rec}")
