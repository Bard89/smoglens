# PM2.5 Prediction Dashboard - V1 Baseline

Interactive Streamlit dashboard for visualizing PM2.5 predictions across Japan using the V1 baseline model.

## Features

- **Pin-Based Map**: Click anywhere on Japan map to select location
- **Automatic Data Finding**: Finds nearest data station with sufficient coverage
- **Smart Date Filtering**: Only shows dates with available data for predictions
- **Real-time Predictions**: Generate forecasts for 1h, 6h, 12h, and 24h horizons
- **Comparison View**: See predicted vs actual values when available
- **Performance Metrics**: MAE, RMSE, R² scores
- **Export**: Download predictions as CSV

## Installation

```bash
cd voi/V4_streamlit/V1_baseline
pip install -r requirements.txt
```

## Usage

```bash
# Navigate to the app directory
cd /Users/vojtech/Code/Bard89/smoglens/voi/V4_streamlit/V1_baseline

# Run the Streamlit app
streamlit run app.py
```

Then open your browser to http://localhost:8501

## How to Use

1. **Click on Map**: Click anywhere in Japan to place a pin
2. **Automatic Station Selection**: App finds nearest data station with good coverage
3. **Choose Time**: Date picker shows only valid dates for that station
4. **Generate Predictions**: Click the prediction button
5. **View Results**: Examine charts, metrics, and comparison table
6. **Export Data**: Download predictions as CSV if needed

## Model Information

- **Model Type**: LightGBM with horizon-specific training
- **Horizons**: 1, 6, 12, 24 hours
- **Coverage**: High-coverage hexagons only (≥90%)
- **Features**: 30 engineered features including lags, rolling stats, temporal encodings

## Data Requirements

The app expects:
- Enriched PM2.5 dataset at `/Users/vojtech/Code/Bard89/smoglens-data/`
- Pre-trained models in the models directory
- Hexagon lookup table for spatial mapping

## Performance

- **1h predictions**: R² ~0.84, MAE ~2.1 μg/m³
- **6h predictions**: R² ~0.55, MAE ~3.6 μg/m³
- **12h predictions**: R² ~0.38, MAE ~4.3 μg/m³
- **24h predictions**: R² ~0.20, MAE ~4.9 μg/m³