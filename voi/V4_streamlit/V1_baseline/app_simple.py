import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

from utils.data_loader import load_enriched_data
from utils.feature_engineering import prepare_features
from utils.visualization import create_prediction_chart

st.set_page_config(
    page_title="PM2.5 Prediction - Shibuya",
    page_icon="🌫️",
    layout="centered"
)

st.title("🌫️ PM2.5 Prediction - Shibuya, Tokyo")
st.markdown("Air quality predictions for Shibuya area")

SHIBUYA_LAT = 35.6580
SHIBUYA_LON = 139.7016
GOOD_HEXAGON = '872e44d04ffffff'

@st.cache_data(ttl=3600)
def load_data():
    with st.spinner("Loading data..."):
        df = load_enriched_data()
        return df

@st.cache_resource
def load_models():
    models = {}
    for horizon in [1, 6, 12, 24]:
        path = f'/Users/vojtech/Code/Bard89/smoglens-data/models/baseline_v2_lgb_{horizon}h_20250901_024347.pkl'
        try:
            models[horizon] = joblib.load(path)
        except:
            pass
    
    for h in [2, 3, 4, 5]:
        if 1 in models:
            models[h] = models[1]
    
    metadata_path = '/Users/vojtech/Code/Bard89/smoglens-data/models/baseline_v2_metadata_20250901_024347.pkl'
    try:
        metadata = joblib.load(metadata_path)
        hex_encoder = metadata.get('hexagon_encoder', None)
    except:
        hex_encoder = None
    
    return models, hex_encoder, metadata

df = load_data()
models, hex_encoder, metadata = load_models()

if hex_encoder is None:
    hex_encoder = LabelEncoder()
    hex_encoder.fit(df['hex7_id'].unique())

hex_data = df[df['hex7_id'] == GOOD_HEXAGON].sort_values('timestamp')
hex_data['time_diff'] = hex_data['timestamp'].diff()
hex_data['new_segment'] = hex_data['time_diff'] > pd.Timedelta(days=7)
hex_data['segment'] = hex_data['new_segment'].cumsum()

segments = hex_data.groupby('segment').agg({
    'timestamp': ['min', 'max', 'count']
})
largest_segment = segments.loc[segments[('timestamp', 'count')].idxmax()]

min_valid_date = largest_segment[('timestamp', 'min')] + pd.Timedelta(hours=200)
max_valid_date = largest_segment[('timestamp', 'max')]

st.subheader("📍 Location: Shibuya, Tokyo")
st.info("Using nearest continuous data source (rural station ~400km away)")

st.subheader("⏰ Select Date and Time")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Date",
        value=max_valid_date.date() - timedelta(days=30),
        min_value=min_valid_date.date(),
        max_value=max_valid_date.date()
    )

with col2:
    selected_hour = st.selectbox(
        "Hour (UTC)",
        options=list(range(24)),
        index=12
    )

selected_timestamp = pd.Timestamp(
    datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour),
    tz='UTC'
)

st.info(f"Selected: {selected_timestamp.strftime('%Y-%m-%d %H:%M')} UTC")

if st.button("🔮 Generate PM2.5 Predictions", type="primary", use_container_width=True):
    with st.spinner("Calculating predictions..."):
        
        window_start = selected_timestamp - pd.Timedelta(hours=200)
        window_end = selected_timestamp + pd.Timedelta(hours=1)
        
        window_data = df[
            (df['hex7_id'] == GOOD_HEXAGON) & 
            (df['timestamp'] >= window_start) & 
            (df['timestamp'] <= window_end)
        ].copy()
        
        if window_data.empty:
            st.error("No data available for selected time")
        else:
            window_data, _ = prepare_features(window_data, selected_timestamp, GOOD_HEXAGON, hex_encoder)
            
            target_row = window_data[window_data['timestamp'] <= selected_timestamp].iloc[-1:] if not window_data.empty else None
            
            if target_row is not None and not target_row.empty:
                feature_cols = metadata['features']
                
                for feat in feature_cols:
                    if feat not in target_row.columns:
                        target_row[feat] = 0.0 if 'traffic' in feat or 'congestion' in feat else 1.0 if feat == 'data_completeness_score' else 0.0
                
                features = target_row[feature_cols]
                
                current_pm25 = float(target_row['pm25_current'].iloc[0]) if not target_row['pm25_current'].empty else 0.0
                st.info(f"Current PM2.5: {current_pm25:.1f} μg/m³")
                
                predictions = {}
                horizons = [1, 2, 3, 4, 5, 6, 12, 24]
                for horizon in horizons:
                    model_key = horizon if horizon in [1, 6, 12, 24] else 1
                    if model_key in models:
                        try:
                            pred_array = models[model_key].predict(features, num_iteration=models[model_key].best_iteration)
                            pred_value = float(pred_array[0]) if hasattr(pred_array, '__len__') else float(pred_array)
                            
                            predictions[horizon] = {
                                'timestamp': selected_timestamp + pd.Timedelta(hours=horizon),
                                'predicted': pred_value
                            }
                        except Exception as e:
                            st.warning(f"Could not predict {horizon}h: {str(e)}")
                
                if predictions:
                    st.success("✅ Predictions generated successfully!")
                    
                    future_data = df[
                        (df['hex7_id'] == GOOD_HEXAGON) & 
                        (df['timestamp'] > selected_timestamp) & 
                        (df['timestamp'] <= selected_timestamp + pd.Timedelta(hours=24))
                    ].copy()
                    
                    pred_data = []
                    for h, data in predictions.items():
                        actual_row = future_data[
                            (future_data['timestamp'] >= data['timestamp'] - pd.Timedelta(minutes=30)) &
                            (future_data['timestamp'] <= data['timestamp'] + pd.Timedelta(minutes=30))
                        ]
                        actual_val = actual_row['pm25_ugm3_mean'].iloc[0] if not actual_row.empty else None
                        
                        pred_data.append({
                            'horizon': h,
                            'timestamp': data['timestamp'],
                            'predicted': data['predicted'],
                            'actual': actual_val
                        })
                    
                    pred_df = pd.DataFrame(pred_data)
                    pred_df['error'] = pred_df.apply(
                        lambda x: x['predicted'] - x['actual'] if pd.notna(x['actual']) else None, 
                        axis=1
                    )
                    
                    st.subheader("📊 PM2.5 Predictions vs Actual")
                    
                    historical = df[
                        (df['hex7_id'] == GOOD_HEXAGON) & 
                        (df['timestamp'] >= selected_timestamp - pd.Timedelta(hours=24)) & 
                        (df['timestamp'] <= selected_timestamp)
                    ].copy()
                    
                    fig = create_prediction_chart(pred_df, historical)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📋 Comparison Table")
                    display_df = pred_df[['horizon', 'timestamp', 'predicted', 'actual', 'error']].copy()
                    display_df['predicted'] = display_df['predicted'].astype(float).round(1)
                    display_df['actual'] = display_df['actual'].astype(float).round(1) if display_df['actual'].notna().any() else display_df['actual']
                    display_df['error'] = display_df['error'].astype(float).round(1) if display_df['error'].notna().any() else display_df['error']
                    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M UTC')
                    display_df.columns = ['Hours', 'Time', 'Predicted', 'Actual', 'Error']
                    st.dataframe(display_df, hide_index=True)
                    
                    if pred_df['actual'].notna().any():
                        mae = pred_df['error'].abs().mean()
                        if pd.notna(mae):
                            st.metric("Mean Absolute Error", f"{mae:.2f} μg/m³")
                    
                    st.info("💡 Air Quality: Good (0-12), Moderate (12-35), Unhealthy for Sensitive (35-55)")
                else:
                    st.error("Could not generate predictions")
            else:
                st.error("Could not prepare features for prediction")

st.divider()
st.caption("Note: Using data from rural Japan station due to gaps in Tokyo data")
st.caption("Model: LightGBM baseline V2")