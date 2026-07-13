# SmogLens V2 - PM2.5 Prediction System

Real-time PM2.5 predictions for outdoor activities in Shibuya, Tokyo.

## Features
- 6-hour PM2.5 forecasts using ensemble models
- Activity-based safety thresholds
- K-NN spatial imputation from nearby areas
- Historical data analysis

## Deployment

### Streamlit Cloud
1. Fork/clone this repository
2. Deploy on [share.streamlit.io](https://share.streamlit.io)
3. Set app path: `voi/V4_streamlit/V2_working/app.py`
4. Add secret: `STREAMLIT_CLOUD=true`

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models
Using ensemble models (LightGBM, XGBoost, CatBoost) trained on 2+ years of data.

## Data
Shibuya area data from 2024 onwards, updated hourly.