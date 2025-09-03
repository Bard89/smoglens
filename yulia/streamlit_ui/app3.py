import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import folium
from streamlit_folium import folium_static
import requests
import pytz

# --- Page Config ---
st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.big-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0;
    color: #2E86C1;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    margin-bottom: 20px;
    color: #5D6D7E;
}
.prediction-card {
    background-color: #F8F9FA;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #3498DB;
    margin: 10px 0;
}
.activity-selector {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #E5E7E9;
    margin: 10px 0;
}
.health-alert {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: bold;
}
.alert-good { background-color: #D5F4E6; color: #27AE60; }
.alert-moderate { background-color: #FCF3CF; color: #F39C12; }
.alert-unhealthy { background-color: #FADBD8; color: #E74C3C; }
.metric-box {
    text-align: center;
    padding: 10px;
    border-radius: 5px;
    background-color: #F8F9FA;
    margin: 5px;
}
.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 30px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# --- Get User's Location and Time ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_location_time():
    try:
        # Get user's IP and location
        response = requests.get('http://ip-api.com/json/', timeout=5)
        data = response.json()

        if data['status'] == 'success':
            timezone = data.get('timezone', 'Asia/Tokyo')
            country = data.get('country', 'Japan')
            region = data.get('regionName', 'Tokyo')

            # Get current time in user's timezone
            user_tz = pytz.timezone(timezone)
            current_time = datetime.datetime.now(user_tz)

            return {
                'timezone': timezone,
                'country': country,
                'region': region,
                'current_time': current_time,
                'lat': data.get('lat', 35.6895),
                'lon': data.get('lon', 139.6917)
            }
        else:
            raise Exception("IP API failed")
    except:
        # Fallback to Tokyo time
        tokyo_tz = pytz.timezone('Asia/Tokyo')
        return {
            'timezone': 'Asia/Tokyo',
            'country': 'Japan',
            'region': 'Tokyo',
            'current_time': datetime.datetime.now(tokyo_tz),
            'lat': 35.6895,
            'lon': 139.6917
        }

# Get user's location and time
user_info = get_user_location_time()
current_time = user_info['current_time']

# --- Title Section ---
st.markdown('<p class="big-title">🌬️ Air Quality Forecast Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">PM₂.₅ Predictions • {current_time.strftime("%A, %B %d, %Y")} • Your Location: {user_info["region"]}</p>', unsafe_allow_html=True)

# --- Main Layout: Left side controls, Right side content ---
left_col, main_col = st.columns([1, 3])

# --- Left Column: User Controls ---
with left_col:
    st.markdown("### Location")

    # Pin to predicted/detected location
    st.info(f"Detected: **{user_info['region']}, {user_info['country']}**")

    prefectures = [
        "Tokyo", "Kanagawa", "Saitama", "Chiba", "Aichi", "Osaka", "Hyogo", "Hokkaido",
        "Fukuoka", "Shizuoka", "Ibaraki", "Hiroshima", "Kyoto", "Miyagi", "Niigata",
        "Nagano", "Gunma", "Tochigi", "Okayama", "Fukushima", "Gifu", "Mie", "Kumamoto",
        "Kagoshima", "Okinawa", "Aomori", "Ehime", "Yamaguchi", "Nagasaki", "Iwate",
        "Ishikawa", "Oita", "Akita", "Yamagata", "Toyama", "Wakayama", "Yamanashi",
        "Fukui", "Kochi", "Shiga", "Nara", "Tottori", "Shimane", "Saga", "Tokushima",
        "Kagawa"
    ]

    # Default to detected region if it's in the list
    default_idx = 0
    if user_info['region'] in prefectures:
        default_idx = prefectures.index(user_info['region'])

    location = st.selectbox("Override Location", prefectures, index=default_idx)

    st.markdown("### ⚙️ Settings")
    forecast_hours = st.slider("Forecast Duration (Hours)", 1, 24, 12)

    st.markdown("### 🏃 Select Activity")
    st.markdown('<div class="activity-selector">', unsafe_allow_html=True)

    activities = {
        "🏠 Staying Indoor": {"icon": "🏠", "risk_factor": 0.1},
        "🚶 Walking Outside": {"icon": "🚶", "risk_factor": 1.0},
        "🏃 Jogging/Running": {"icon": "🏃", "risk_factor": 2.0},
        "🚴 Cycling": {"icon": "🚴", "risk_factor": 1.8},
        "⚽ Outdoor Sports": {"icon": "⚽", "risk_factor": 2.5}
    }

    selected_activity = st.radio("", list(activities.keys()), index=1)
    activity_info = activities[selected_activity]

    st.markdown('</div>', unsafe_allow_html=True)

    # User profile for personalized advice
    st.markdown("### 👤 Personal Profile")
    children = st.checkbox("👶 Children present")
    elderly = st.checkbox("👴 Elderly present")
    allergies = st.checkbox("🤧 Respiratory sensitivities")
    heart_condition = st.checkbox("❤️ Heart conditions")

# --- Main Column: Map and Graph ---
with main_col:
    # --- Top: Map View ---
    st.markdown("### 🗺️ Location & Air Quality Map")

    # Create map centered on selected location
    city_coordinates = {
        "Tokyo": [35.6895, 139.6917], "Kanagawa": [35.4478, 139.6425], "Saitama": [35.8517, 139.6455],
        "Chiba": [35.6047, 140.1233], "Aichi": [35.1815, 136.9066], "Osaka": [34.6937, 135.5023],
        "Hyogo": [34.6864, 135.1974], "Hokkaido": [43.0618, 141.3545], "Fukuoka": [33.5904, 130.4017],
        "Shizuoka": [34.9769, 138.3830], "Ibaraki": [36.3418, 140.4468], "Hiroshima": [34.3853, 132.4553],
        "Kyoto": [35.0116, 135.7681], "Miyagi": [38.2682, 140.8694], "Niigata": [37.9026, 139.0236],
        "Nagano": [36.6528, 138.1812], "Gunma": [36.3912, 139.0609], "Tochigi": [36.5658, 139.8836],
        "Okayama": [34.6617, 133.5577], "Fukushima": [37.7503, 140.4678], "Gifu": [35.3912, 136.7222],
        "Mie": [34.7303, 136.5083], "Kumamoto": [32.7894, 130.7417], "Kagoshima": [31.5602, 130.5580],
        "Okinawa": [26.2124, 127.6811], "Aomori": [40.8243, 140.7400], "Ehime": [33.8416, 132.7661],
        "Yamaguchi": [34.1859, 131.4714], "Nagasaki": [32.7447, 129.8737], "Iwate": [39.7036, 141.1527],
        "Ishikawa": [36.5946, 136.6255], "Oita": [33.2494, 131.6125], "Akita": [39.7213, 140.1025],
        "Yamagata": [38.2592, 140.3624], "Toyama": [36.6983, 137.2117], "Wakayama": [34.2260, 135.1675],
        "Yamanashi": [35.6639, 138.5683], "Fukui": [36.0652, 136.2219], "Kochi": [33.5597, 133.5311],
        "Shiga": [35.0045, 135.8683], "Nara": [34.6851, 135.8048], "Tottori": [35.5062, 134.2383],
        "Shimane": [35.4720, 133.0505], "Saga": [33.2494, 130.3009], "Tokushima": [34.0658, 134.5596],
        "Kagawa": [34.3401, 134.0435]
    }

    map_coords = city_coordinates.get(location, [35.6895, 139.6917])
    m = folium.Map(location=map_coords, zoom_start=10, tiles='OpenStreetMap')

    # Add marker for selected location with color based on air quality
    current_pm = 20 + np.random.randn() * 5  # Simulated current PM2.5
    marker_color = 'green' if current_pm <= 15 else 'orange' if current_pm <= 35 else 'red'

    folium.Marker(
        map_coords,
        popup=f"{location}<br>PM2.5: {current_pm:.1f} µg/m³",
        tooltip=f"{location}",
        icon=folium.Icon(color=marker_color, icon='info-sign')
    ).add_to(m)

    folium_static(m, width=700, height=300)

# --- Bottom: Forecast Graph ---
with main_col:
    st.markdown("### 📈 PM2.5 Forecast - Predicted Values")
    st.markdown(f"**Current Time:** {current_time.strftime('%H:%M %Z')} | **Location:** {location}")

    # Generate forecast starting from +1 hour from current time
    start_hour = current_time + datetime.timedelta(hours=1)
    forecast_times = [start_hour + datetime.timedelta(hours=i) for i in range(forecast_hours)]

    # Generate realistic forecast data with daily patterns
    hours_from_midnight = [(t.hour + t.minute/60) for t in forecast_times]

    # Daily pattern: lower at night, higher during traffic hours
    daily_pattern = 15 + 10 * np.sin(2 * np.pi * (np.array(hours_from_midnight) - 6) / 24) + \
                   5 * np.sin(2 * np.pi * (np.array(hours_from_midnight) - 8) / 12)  # Rush hour pattern

    # Add some realistic noise and trend
    noise = np.random.randn(forecast_hours) * 3
    trend = np.linspace(0, 2, forecast_hours)  # Slight upward trend
    pm25_values = np.clip(daily_pattern + noise + trend, 5, 60)

    # Calculate confidence intervals (decreases with time)
    base_uncertainty = 3
    time_uncertainty = np.linspace(2, 8, forecast_hours)  # Uncertainty increases with time
    upper_bound = pm25_values + time_uncertainty
    lower_bound = pm25_values - time_uncertainty

    # Create DataFrame for plotting
    df_forecast = pd.DataFrame({
        'Hour': [f"{t.strftime('%H:%M')}" for t in forecast_times],
        'Time': [t.hour + t.minute/60 for t in forecast_times],
        'PM2.5': pm25_values,
        'Upper': upper_bound,
        'Lower': lower_bound,
        'Day_Period': ['Night' if (6 <= t.hour <= 18) else 'Day' for t in forecast_times]
    })

    # Create Altair chart
    base = alt.Chart(df_forecast).add_selection(
        alt.selection_interval(bind='scales')
    )

    # Confidence band
    confidence_band = base.mark_area(
        opacity=0.3,
        color='lightblue'
    ).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Time (Hour:Minute)', labelAngle=-45)),
        y=alt.Y('Lower:Q', scale=alt.Scale(domain=[0, max(upper_bound) + 5])),
        y2='Upper:Q'
    )

    # Main prediction line
    prediction_line = base.mark_line(
        color='#3498DB',
        strokeWidth=3,
        point=alt.OverlayMarkDef(color='#2980B9', size=60)
    ).encode(
        x='Hour:O',
        y=alt.Y('PM2.5:Q', axis=alt.Axis(title='PM2.5 (µg/m³)')),
        tooltip=['Hour:O', 'PM2.5:Q']
    )

    # Health threshold lines
    good_line = alt.Chart(pd.DataFrame({'y': [15]})).mark_rule(
        color='green', strokeDash=[5, 5]
    ).encode(y='y:Q')

    moderate_line = alt.Chart(pd.DataFrame({'y': [35]})).mark_rule(
        color='orange', strokeDash=[5, 5]
    ).encode(y='y:Q')

    # Combine chart
    chart = (confidence_band + prediction_line + good_line + moderate_line).resolve_scale(
        y='independent'
    ).properties(
        width=700,
        height=400,
        title=f"PM2.5 Forecast for {location} (Starting {start_hour.strftime('%H:%M %Z')})"
    )

    st.altair_chart(chart, use_container_width=True)

    # Show confidence score
    avg_uncertainty = time_uncertainty.mean()
    confidence_score = max(50, 100 - (avg_uncertainty / 8) * 50)
    st.info(f"🎯 **Forecast Confidence:** {confidence_score:.1f}% (decreases over time)")

# --- Health Advisory Section ---
with main_col:
    st.markdown("### 🏥 Personalized Health Advisory")

    # Get max PM2.5 in forecast period for risk assessment
    max_pm_forecast = max(pm25_values)
    avg_pm_forecast = np.mean(pm25_values)

    # Determine health category
    if max_pm_forecast <= 15:
        health_category = "Good ✅"
        alert_class = "alert-good"
    elif max_pm_forecast <= 35:
        health_category = "Moderate ⚠️"
        alert_class = "alert-moderate"
    else:
        health_category = "Unhealthy ❗"
        alert_class = "alert-unhealthy"

    # Risk adjustment based on activity and personal factors
    risk_multiplier = activity_info["risk_factor"]
    personal_risk = 1.0

    if children:
        personal_risk *= 1.3
    if elderly:
        personal_risk *= 1.2
    if allergies:
        personal_risk *= 1.4
    if heart_condition:
        personal_risk *= 1.5

    adjusted_risk_pm = avg_pm_forecast * risk_multiplier * personal_risk

    # Display health advisory
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Current PM2.5", f"{current_pm:.1f} µg/m³")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Avg Forecast", f"{avg_pm_forecast:.1f} µg/m³")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Activity Risk", f"{adjusted_risk_pm:.1f} µg/m³",
                 f"{((adjusted_risk_pm/avg_pm_forecast - 1) * 100):+.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Personalized recommendations
    st.markdown(f'<div class="health-alert {alert_class}">🏥 Air Quality: {health_category}</div>',
                unsafe_allow_html=True)

    recommendations = []

    if adjusted_risk_pm > 35:
        recommendations.extend([
            f"⚠️ **Avoid {selected_activity.lower()}** during peak hours (7-9 AM, 5-7 PM)",
            "🏠 Consider staying indoors when possible",
            "😷 Wear N95/P2 masks if going outside"
        ])

        if children:
            recommendations.append("👶 Keep children indoors, especially during outdoor play time")
        if elderly:
            recommendations.append("👴 Elderly should minimize outdoor exposure")
        if allergies or heart_condition:
            recommendations.append("💊 Have medication readily available")

    elif adjusted_risk_pm > 15:
        recommendations.extend([
            f"⚡ **Reduce intensity** of {selected_activity.lower()}",
            "🕐 Choose early morning (5-7 AM) or late evening (8-10 PM) for outdoor activities",
            "🏢 Consider indoor alternatives when possible"
        ])

        if allergies:
            recommendations.append("🤧 Monitor symptoms and consider wearing a mask")

    else:
        recommendations.extend([
            f"✅ **Safe for {selected_activity.lower()}**",
            "🌟 Great time for outdoor activities!",
            "🚶 No special precautions needed"
        ])

    for rec in recommendations:
        st.markdown(f"• {rec}")

# --- Footer ---
st.markdown("---")
st.markdown(f'<p class="footer">🕒 Last updated: {current_time.strftime("%H:%M %Z")} • Built by Voi & Yulia</p>', unsafe_allow_html=True)
