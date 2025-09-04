# import pickle
# model = pickle.load(open("models/air_quality_model.pkl", "rb"))

import streamlit as st
import datetime
import random

# Title
st.title("🌤 Air Quality Forecast")

# Location input
location = st.selectbox("Select Location", ["Tokyo", "Osaka", "Kyoto"])

# Forecast time
forecast_hour = st.slider("Forecast for next (hours)", 1, 24, 1)

# Extra options for personalization
children = st.checkbox("Children at home?")
allergies = st.checkbox("Allergies?")
jogging = st.checkbox("Planning to jog outside?")

# Predict button
if st.button("Get Prediction"):
    # Dummy prediction
    aqi = random.randint(10, 150)
    pm25 = round(random.uniform(5, 50), 2)
    pm10 = round(random.uniform(10, 80), 2)

    st.subheader(f"📍 Location: {location}")
    st.write(f"⏳ Forecast for next {forecast_hour} hours")

    st.metric("Air Quality Index (AQI)", aqi)
    st.metric("PM2.5", f"{pm25} µg/m³")
    st.metric("PM10", f"{pm10} µg/m³")

    # Health advisory
    advice = "✅ Safe to go outside." if aqi < 50 else "⚠ Stay indoors, consider using a mask."
    if allergies and aqi > 50:
        advice += " Allergies may worsen."
    if children and aqi > 50:
        advice += " Limit outdoor play for children."
    if jogging and aqi > 50:
        advice += " Postpone jogging or wear a mask."

    st.write(f"**Health Advice:** {advice}")

# Footer
st.caption("Powered by Streamlit • Demo App")
