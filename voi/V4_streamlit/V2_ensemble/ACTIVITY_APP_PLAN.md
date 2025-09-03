# Activity-Based PM2.5 Prediction App - Implementation Plan

## Overview
Create a simplified, activity-focused PM2.5 prediction app for Shibuya, Tokyo with preprocessed data for easy deployment.

## Design Requirements (from user sketch)
- Fixed location: Shibuya, Tokyo with pin on map
- Current time display
- Activity selector: Walking vs Running
- 6-hour prediction graph
- Activity-specific PM2.5 thresholds
- Confidence intervals based on MSE
- Small deployable dataset

## Project Structure
```
/Users/vojtech/Code/Bard89/smoglens/voi/V4_streamlit/V3_activity/
├── app.py                      # Main activity-based app
├── config.py                   # Configuration with thresholds
├── preprocess_shibuya_data.py  # Data preprocessing script
├── combine_models.py           # Model optimization script
├── requirements.txt            # Dependencies
├── data/
│   └── shibuya_pm25_data.csv  # Preprocessed data (5-10MB)
├── models/
│   └── ensemble_models.pkl    # Combined models for 1-6h only
└── utils/
    ├── __init__.py
    ├── data_loader.py         # Load preprocessed data
    ├── prediction.py          # Generate predictions
    └── visualization.py       # Activity-based charts
```

## Implementation Steps

### Step 1: Data Preprocessing Script
**File: `preprocess_shibuya_data.py`**

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SHIBUYA_HEXAGON = '872e44d04ffffff'
SOURCE_DATA = '/Users/vojtech/Code/Bard89/smoglens-data/pm25_enriched_2023_2025_v4_20250830_222050.csv'
OUTPUT_PATH = 'data/shibuya_pm25_data.csv'

def preprocess_shibuya_data():
    # 1. Load full dataset
    print("Loading original dataset...")
    df = pd.read_csv(SOURCE_DATA, parse_dates=['timestamp'])
    
    # 2. Filter Shibuya hexagon only
    print(f"Filtering for Shibuya hexagon: {SHIBUYA_HEXAGON}")
    shibuya_data = df[df['hex7_id'] == SHIBUYA_HEXAGON].copy()
    
    # 3. Set timestamp as index and ensure hourly frequency
    shibuya_data = shibuya_data.set_index('timestamp').sort_index()
    shibuya_data = shibuya_data.resample('1H').mean()
    
    # 4. Fill gaps in PM2.5 data
    print("Filling data gaps...")
    shibuya_data['pm25_ugm3_mean'] = shibuya_data['pm25_ugm3_mean'].interpolate(
        method='linear', limit=3
    ).fillna(method='bfill').fillna(method='ffill')
    
    # 5. Add essential features for predictions
    print("Creating features...")
    # Lag features (1-6 hours needed for 6h predictions)
    for lag in [1, 2, 3, 4, 5, 6]:
        shibuya_data[f'pm25_lag_{lag}h'] = shibuya_data['pm25_ugm3_mean'].shift(lag)
    
    # Rolling statistics
    shibuya_data['pm25_rolling_mean_3h'] = shibuya_data['pm25_ugm3_mean'].rolling(3).mean()
    shibuya_data['pm25_rolling_std_3h'] = shibuya_data['pm25_ugm3_mean'].rolling(3).std()
    
    # Temporal features
    shibuya_data['hour'] = shibuya_data.index.hour
    shibuya_data['day_of_week'] = shibuya_data.index.dayofweek
    shibuya_data['is_weekend'] = (shibuya_data['day_of_week'] >= 5).astype(int)
    
    # 6. Keep only last 2 years of data
    cutoff_date = datetime.now() - timedelta(days=730)
    shibuya_data = shibuya_data[shibuya_data.index >= cutoff_date]
    
    # 7. Remove any remaining NaN values
    shibuya_data = shibuya_data.dropna()
    
    # 8. Reset index and save
    shibuya_data = shibuya_data.reset_index()
    shibuya_data['hex7_id'] = SHIBUYA_HEXAGON
    
    print(f"Saving preprocessed data ({len(shibuya_data)} rows)...")
    shibuya_data.to_csv(OUTPUT_PATH, index=False, compression='gzip')
    
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"✅ Saved to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    
    return shibuya_data
```

### Step 2: Model Combination Script
**File: `combine_models.py`**

```python
import joblib
import pickle

def combine_models_for_deployment():
    """Combine only the models needed for 1-6h predictions"""
    
    combined = {}
    model_dir = '/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained'
    
    # Load only 1-6h models
    for h in ['1h', '2h', '3h', '4h', '5h', '6h']:
        model_path = f'{model_dir}/models_{h}.pkl'
        print(f"Loading {h} model...")
        combined[h] = joblib.load(model_path)
    
    # Save as single compressed file
    output_path = 'models/ensemble_models.pkl'
    print(f"Saving combined models to {output_path}...")
    joblib.dump(combined, output_path, compress=3)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Combined model size: {file_size_mb:.1f} MB")
    
    return combined
```

### Step 3: Main App
**File: `app.py`**

```python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="SmogLens",
    page_icon="🗾",
    layout="wide"
)

# Fixed location
SHIBUYA_LAT = 35.6580
SHIBUYA_LON = 139.7016

# Activity thresholds (μg/m³)
ACTIVITY_LIMITS = {
    'Walking': 30,
    'Running': 10  # 3x stricter due to higher breathing rate
}

@st.cache_data
def load_shibuya_data():
    """Load preprocessed Shibuya data"""
    return pd.read_csv('data/shibuya_pm25_data.csv', 
                      parse_dates=['timestamp'],
                      compression='gzip')

@st.cache_resource
def load_models():
    """Load combined ensemble models"""
    return joblib.load('models/ensemble_models.pkl')

def create_tokyo_map():
    """Create map with Shibuya pin"""
    m = folium.Map(location=[SHIBUYA_LAT, SHIBUYA_LON], zoom_start=11)
    folium.Marker(
        [SHIBUYA_LAT, SHIBUYA_LON],
        popup="Shibuya, Tokyo",
        tooltip="📍 Prediction Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    return m

def generate_predictions(current_time, data, models):
    """Generate predictions for next 6 hours"""
    predictions = []
    
    # Get latest data point
    latest_data = data[data['timestamp'] <= current_time].iloc[-1]
    
    for h in range(1, 7):
        horizon = f'{h}h'
        if horizon in models:
            # Prepare features (simplified)
            features = prepare_features_for_hour(latest_data, h)
            
            # Get ensemble prediction
            pred = models[horizon].predict(features)[0]
            
            predictions.append({
                'hour': h,
                'time': current_time + timedelta(hours=h),
                'pm25': pred,
                'std_error': np.sqrt(MODEL_MSE[horizon])  # From config
            })
    
    return pd.DataFrame(predictions)

# Main App
st.title("🗾 Tokyo Air Quality - Activity Planner")
st.markdown("Real-time PM2.5 predictions for outdoor activities in Shibuya")

# Load data and models
data = load_shibuya_data()
models = load_models()

# Top section: Map and Current Time
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Location: Shibuya, Tokyo")
    tokyo_map = create_tokyo_map()
    st_folium(tokyo_map, height=200, width=None)

with col2:
    # Current time in Tokyo
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    current_time = datetime.now(tokyo_tz)
    
    st.subheader("🕐 Current Time")
    st.metric("Tokyo Time", current_time.strftime("%H:%M"))
    st.caption(current_time.strftime("%Y-%m-%d %A"))
    
    # Current PM2.5
    latest_pm25 = data.iloc[-1]['pm25_ugm3_mean']
    st.metric("Current PM2.5", f"{latest_pm25:.1f} μg/m³")

# Activity Selector
st.divider()
activity = st.radio(
    "Select Your Activity:",
    options=['🚶 Walking', '🏃 Running'],
    horizontal=True,
    index=0
)

selected_activity = activity.split()[1]  # Extract 'Walking' or 'Running'
threshold = ACTIVITY_LIMITS[selected_activity]

# Generate and display predictions
st.divider()
st.subheader(f"6-Hour Forecast for {activity}")

# Generate predictions
predictions = generate_predictions(current_time, data, models)

# Create activity chart
fig = create_activity_prediction_chart(
    predictions,
    activity_threshold=threshold,
    activity_name=selected_activity
)

st.plotly_chart(fig, use_container_width=True)

# Safety recommendation
safe_hours = predictions[predictions['pm25'] < threshold]
if len(safe_hours) > 0:
    st.success(f"✅ Safe hours for {selected_activity}: {len(safe_hours)} out of 6 hours")
    st.info(f"Best time: {safe_hours.iloc[0]['time'].strftime('%H:%M')} "
            f"(PM2.5: {safe_hours.iloc[0]['pm25']:.1f} μg/m³)")
else:
    st.warning(f"⚠️ Not recommended for {selected_activity} in the next 6 hours")
    st.info(f"Consider indoor activities or postponing your {selected_activity.lower()}")

# Footer
st.divider()
st.caption("Data source: Rural station ~400km from Tokyo • Model: Ensemble (LGB+XGB+CatBoost)")
st.caption("Thresholds based on WHO guidelines adjusted for activity intensity")
```

### Step 4: Visualization
**File: `utils/visualization.py`**

```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_activity_prediction_chart(predictions, activity_threshold, activity_name):
    """Create chart with predictions, threshold, and confidence intervals"""
    
    fig = go.Figure()
    
    # Add prediction points and line
    fig.add_trace(go.Scatter(
        x=predictions['time'],
        y=predictions['pm25'],
        mode='markers+lines',
        name='Predicted PM2.5',
        marker=dict(size=12, color='darkblue'),
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals (95% CI = ±1.96 * std_error)
    upper_bound = predictions['pm25'] + 1.96 * predictions['std_error']
    lower_bound = predictions['pm25'] - 1.96 * predictions['std_error']
    
    # Upper bound
    fig.add_trace(go.Scatter(
        x=predictions['time'],
        y=upper_bound,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,200,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=predictions['time'],
        y=lower_bound,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,200,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(0,100,200,0.2)'
    ))
    
    # Add activity threshold line
    fig.add_hline(
        y=activity_threshold,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"{activity_name} Safe Limit ({activity_threshold} μg/m³)",
        annotation_position="right"
    )
    
    # Add safe zone (green)
    fig.add_hrect(
        y0=0, 
        y1=activity_threshold,
        fillcolor="green", 
        opacity=0.1,
        layer="below", 
        line_width=0,
        annotation_text="Safe Zone",
        annotation_position="top left"
    )
    
    # Add unsafe zone (red)
    fig.add_hrect(
        y0=activity_threshold, 
        y1=50,
        fillcolor="red", 
        opacity=0.1,
        layer="below", 
        line_width=0,
        annotation_text="Not Recommended",
        annotation_position="bottom left"
    )
    
    # Update layout
    fig.update_layout(
        title=f"PM2.5 Forecast - {activity_name} Safety",
        xaxis_title="Time",
        yaxis_title="PM2.5 Concentration (μg/m³)",
        yaxis_range=[0, 50],
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Format x-axis to show hours
    fig.update_xaxis(
        tickformat="%H:%M",
        dtick=3600000  # 1 hour in milliseconds
    )
    
    return fig
```

### Step 5: Configuration
**File: `config.py`**

```python
# Activity thresholds (μg/m³)
ACTIVITY_LIMITS = {
    'Walking': 30,   # Moderate activity
    'Running': 10    # High intensity (3x stricter)
}

# Model performance metrics (MSE) for confidence intervals
MODEL_MSE = {
    '1h': 7.2,
    '2h': 10.5,
    '3h': 13.2,
    '4h': 15.8,
    '5h': 18.1,
    '6h': 20.3
}

# Location
SHIBUYA_LAT = 35.6580
SHIBUYA_LON = 139.7016
SHIBUYA_HEXAGON = '872e44d04ffffff'

# Data paths
DATA_PATH = 'data/shibuya_pm25_data.csv'
MODEL_PATH = 'models/ensemble_models.pkl'
```

## Deployment Strategy

### Size Optimization
- Original data: ~2GB → Preprocessed: ~5-10MB
- Original models: ~500MB → Combined 6h models: ~50MB
- Total deployment size: <100MB

### Deployment Options
1. **Streamlit Cloud**: Free tier, perfect for this size
2. **Heroku**: Free dyno sufficient
3. **GitHub Pages**: With Streamlit sharing
4. **Google Cloud Run**: Serverless option

### Next Steps
1. Run `preprocess_shibuya_data.py` to create dataset
2. Run `combine_models.py` to optimize models
3. Test app locally with `streamlit run app.py`
4. Deploy to chosen platform

## Key Features
- ✅ Real-time predictions (uses current time)
- ✅ Activity-specific thresholds
- ✅ Confidence intervals
- ✅ Visual safe/unsafe zones
- ✅ Small deployment footprint
- ✅ No user inputs needed (except activity)
- ✅ Mobile-friendly design
- ✅ Fast loading (preprocessed data)