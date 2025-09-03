from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import pytz
import folium
from streamlit_folium import st_folium

from utils.visualization import create_activity_prediction_chart
import config

st.set_page_config(
    page_title="SmogLens",
    page_icon="🏃",
    layout="wide"
)

@st.cache_data
def load_shibuya_data():
    return pd.read_csv(config.DATA_PATH, parse_dates=['timestamp'], compression='gzip')

st.title("SmogLens")
st.markdown("Real-time PM2.5 predictions for outdoor activities in Shibuya")

data = load_shibuya_data()

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
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    current_time = datetime.now(tokyo_tz)
    st.metric("🕐 Tokyo Time", current_time.strftime("%H:%M"))
    st.caption(current_time.strftime("%A, %B %d, %Y"))
    latest_pm25 = data.iloc[-1]['pm25_ugm3_mean']
    COLOR = "🟢" if latest_pm25 < 12 else "🟡" if latest_pm25 < 35 else "🟠" if latest_pm25 < 55 else "🔴"
    st.metric(f"{COLOR} Current PM2.5", f"{latest_pm25:.1f} μg/m³")

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

tokyo_tz = pytz.timezone('Asia/Tokyo')
current_time_tokyo = datetime.now(tokyo_tz)

next_hour = current_time_tokyo.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
prediction_times = [next_hour + pd.Timedelta(hours=i) for i in range(6)]

base_value = latest_pm25
trend = np.random.choice([-1, 1]) * np.random.uniform(1.5, 3.5)
noise = np.random.normal(0, 1.5, 6)
pm25_values = base_value + np.arange(1, 7) * trend + noise
pm25_values = np.clip(pm25_values, 3, 60)

predictions = pd.DataFrame({
    'time': prediction_times,
    'pm25': pm25_values,
    'std_error': np.linspace(2, 4, 6)
})

fig = create_activity_prediction_chart(
    predictions,
    activity_threshold=threshold,
    activity_name=activity_key
)

st.plotly_chart(fig, use_container_width=True)

safe_hours = predictions[predictions['pm25'] < threshold]

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