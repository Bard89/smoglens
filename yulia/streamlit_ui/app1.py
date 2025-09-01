# import pickle
# model = pickle.load(open("models/air_quality_model.pkl", "rb"))

import streamlit as st
import pandas as pd
import numpy as np
import datetime

# --- Style tweaks for mobile widget feel ---
st.set_page_config(page_title="AQI Widget", layout="centered")

st.markdown("""
<style>
.small-widget {
    background-color: #f0f9ff;
    border-radius: 15px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# Widget Container
st.markdown('<div class="small-widget">', unsafe_allow_html=True)

# Title & Time/Weather placeholders
now = datetime.datetime.now().strftime("%a %H:%M")
st.write(f"**Air Quality Forecast**  ·  {now}")
st.write("**Weather:** ☀ 25°C")  # placeholder

# Location input
location = st.selectbox("Select Location", ["Tokyo", "Osaka", "Kyoto"], index=0) #Tokyo as default

# Forecast time
forecast_hour = st.slider("Forecast for next (hours)", 1, 24, 1)

# Extra options for personalization
children = st.checkbox("Children at home?")
allergies = st.checkbox("Allergies?")
jogging = st.checkbox("Planning to jog outside?")

# Forecast graph (dummy data)
hours = np.arange(1, 13)
pm25 = np.clip(20 + np.sin(hours / 2) * 15 + np.random.randn(12) * 2, 5, 100)
df = pd.DataFrame({"Hour": hours, "PM2.5": pm25})
chart = st.line_chart(df.set_index("Hour"))

# Health advisory metadata
st.subheader("Health Advisory")

# Get latest PM2.5
latest_pm = pm25[-1]
advice = "All clear—safe for everyone!  "
sensitive_group = children or allergies

if latest_pm > 35:
    advice = "⚠ High PM2.5 for sensitive groups. "
    if children:
        advice += "Keep children indoors. "
    if allergies:
        advice += "Limit outdoor activities if you have allergies. "
    if not sensitive_group:
        advice = "High PM2.5. Limit prolonged outdoor exertion."

st.write(f"**Current PM2.5:** {latest_pm:.1f} µg/m³")
st.write(advice)

st.markdown('</div>', unsafe_allow_html=True)
