import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt
import folium
from streamlit_folium import folium_static

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
}
.sub-title {
    font-size: 18px;
    text-align: center;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 30px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<p class="big-title">Air Quality Forecast Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">PM₂.₅ Trends · Personalized Health Advice</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Personalize")
    location = st.selectbox("Select Location", ["Tokyo", "Osaka", "Kyoto"], index=0)
    forecast_hour = st.slider("Forecast Next (Hours)", 1, 24, 12)
    children = st.checkbox("Children at home?")
    allergies = st.checkbox("Allergies?")
    activity = st.radio("Planned Activity", ["Indoor Only", "Light Outdoor", "Jogging"], index=1)

# --- Layout ---
col1, col2 = st.columns([2, 1])

# --- Column 1: Forecast Graph ---
with col1:
    now = datetime.datetime.now().strftime("%A, %d %B %Y · %H:%M")
    st.write(f"**Location:** {location} | **Updated:** {now}")

    # Generate forecast data with variance
    hours = np.arange(1, forecast_hour + 1)
    base = 20 + np.sin(hours / 2) * 15
    noise = np.random.randn(forecast_hour) * 2
    pm25 = np.clip(base + noise, 5, 100)

    # Confidence intervals
    variance = np.random.uniform(3, 8, size=forecast_hour)
    upper = pm25 + variance
    lower = pm25 - variance

    df = pd.DataFrame({
        "Hour": hours,
        "PM2.5": pm25,
        "Upper": upper,
        "Lower": lower
    })

    # Compute confidence score (lower variance → higher confidence)
    confidence = max(50, 100 - (variance.mean() / variance.max()) * 40)

    # Altair chart with confidence band
    base_chart = alt.Chart(df).encode(x='Hour')
    band = base_chart.mark_area(opacity=0.2, color='lightblue').encode(
        y='Lower',
        y2='Upper'
    )
    line = base_chart.mark_line(color='blue', strokeWidth=3).encode(
        y='PM2.5'
    )

    st.altair_chart(band + line, use_container_width=True)
    st.write(f"**Forecast Confidence:** {confidence:.1f}%")

# --- Column 2: Map + Health Advisory ---
with col2:
    st.subheader("Map View")
    city_coordinates = {
        "Tokyo": [35.6895, 139.6917],
        "Osaka": [34.6937, 135.5023],
        "Kyoto": [35.0116, 135.7681]
    }

    m = folium.Map(location=[36.2048, 138.2529], zoom_start=5, tiles='OpenStreetMap')
    for city, coords in city_coordinates.items():
        folium.Marker(coords, popup=city, tooltip=city).add_to(m)
    folium_static(m, width=250, height=250)

    st.subheader("Health Advisory")
    latest_pm = pm25[-1]

    # Determine color category
    if latest_pm <= 15:
        category = "Good ✅"
        color = "green"
    elif latest_pm <= 35:
        category = "Moderate ⚠"
        color = "yellow"
    elif latest_pm <= 55:
        category = "Unhealthy for Sensitive Groups ❗"
        color = "orange"
    else:
        category = "Unhealthy ❌"
        color = "red"

    advice = f"**Air Quality:** <span style='color:{color}'>{category}</span><br>"
    if latest_pm > 35:
        advice += "Reduce outdoor activities. "
        if children:
            advice += "Keep children indoors. "
        if allergies:
            advice += "Wear masks and close windows. "
        if activity == "Jogging":
            advice += "Avoid jogging outside."
    else:
        advice += "Air is safe for most activities."

    st.write(f"**Current PM2.5:** {latest_pm:.1f} µg/m³")
    st.markdown(advice, unsafe_allow_html=True)

# --- Footer ---
st.markdown('<p class="footer">Built by ... Should i include it here?</p>', unsafe_allow_html=True)
