# SmogLens Streamlit Implementation Plan - Production Ready

## Analysis Summary

### Data Availability
- **Shibuya hexagon (872e44d04ffffff)**: 15,984 records available
- **Neighboring hexagons**: Limited direct neighbors, need wider search radius
- **Solution**: K-NN spatial imputation from nearby hexagons

### Existing Models (in /Users/vojtech/Code/Bard89/smoglens-02/)
- **Trained models**: Using 1h, 2h, 3h, 4h, 5h, 6h models only (per requirements)
- **Model types**: LightGBM, XGBoost, CatBoost ensemble
- **Required features**: 69 features including traffic, weather, temporal, and engineered features
- **Performance**: MAE beats baseline on all horizons

## Implementation Plan

### Phase 1: Data Pipeline (`data_processor.py`)
```python
# Load and prepare Shibuya area data - NO COMMENTS IN CODE
- Load PM2.5 enriched dataset (9.4M records)
- Filter for Shibuya hexagon (872e44d04ffffff)
- Find K nearest hexagons with data (based on H3 distance)
- Implement spatial imputation using weighted average by distance
- Cache processed data as parquet for speed
```

### Phase 2: Feature Engineering (`feature_generator.py`)
```python
# Generate all 69 required features matching training - NO COMMENTS IN CODE
- Lag features: [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168] hours
- Rolling statistics: mean, std, max, min for [3, 6, 12, 24, 48] hour windows
- EWM features: alpha [0.1, 0.3, 0.5]
- Temporal: hour_sin/cos, dow_sin/cos, month_sin/cos
- Interactions: temp_humidity, temp_hour, traffic_hour, traffic_weekend
- Differences: diff_1h, diff_6h, diff_24h, rate_6h, rate_24h
```

### Phase 3: Spatial Imputation (`spatial_imputer.py`)
```python
# K-NN imputation from nearby hexagons - NO COMMENTS IN CODE
- Find K=10 nearest hexagons with valid data
- Calculate H3 distance between hexagons
- Weight by inverse distance (1/distance)
- Fallback cascade: immediate neighbors → ring 2 → ring 3
- Cache neighbor relationships
```

### Phase 4: Model Interface (`model_predictor.py`)
```python
# Load models and predict 1-6 hours ahead - NO COMMENTS IN CODE
- Load only 1h-6h models from trained/ directory
- Implement batch prediction for efficiency
- Return predictions with confidence intervals
- Cache model objects for fast inference
```

### Phase 5: Streamlit Integration (`app.py`)
```python
# Update existing demo with real predictions - NO COMMENTS IN CODE
- ADD DATE/TIME SELECTOR for historical analysis
- Keep current UI design with enhancements
- Show 6-hour forecast from selected time
- Replace random predictions with model inference
- Add spatial imputation status indicator
- Activity thresholds for all 6 hours
```

## File Structure
```
voi/V4_streamlit/V2_working/
├── implementation_plan.md        # This plan
├── app.py                        # Updated Streamlit app
├── config.py                     # Configuration
├── data/
│   ├── shibuya_processed.parquet # Cached processed data
│   ├── neighbor_cache.pkl        # Cached neighbor relationships
│   └── feature_cache.pkl         # Cached features
├── models/                       # Symlink to trained models
│   └── -> /Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained/
└── utils/
    ├── data_processor.py         # Data loading and caching
    ├── feature_generator.py      # Feature engineering
    ├── spatial_imputer.py        # K-NN spatial imputation
    ├── model_predictor.py        # Model inference
    └── visualization.py          # Existing visualization
```

## Streamlit UI Enhancements

### Date/Time Selector Component
```python
with st.sidebar:
    st.subheader("Select Date & Time")
    
    selected_date = st.date_input(
        "Date",
        min_value=datetime(2023, 7, 14),
        max_value=datetime(2025, 7, 26),
        value=datetime.now()
    )
    
    selected_time = st.time_input(
        "Time",
        value=datetime.now().time()
    )
    
    selected_datetime = datetime.combine(selected_date, selected_time)
    
    use_current_time = st.checkbox("Use current time", value=True)
    
    if use_current_time:
        selected_datetime = datetime.now(pytz.timezone('Asia/Tokyo'))
```

## Technical Implementation Details

### Spatial Imputation Strategy (NO COMMENTS IN CODE)
```python
def impute_missing_data(timestamp, target_hex):
    if has_data(target_hex, timestamp):
        return get_data(target_hex, timestamp)
    
    nearby = find_nearest_with_data(target_hex, k=10)
    
    weighted_sum = 0
    weight_total = 0
    for hex_id, distance in nearby:
        weight = 1.0 / max(distance, 1)
        weighted_sum += get_data(hex_id, timestamp) * weight
        weight_total += weight
    
    return weighted_sum / weight_total
```

### Data Processing Pipeline
1. Load data based on selected datetime
2. Apply spatial imputation if Shibuya data missing
3. Generate features for selected time point
4. Make predictions for next 6 hours

### Feature Engineering (NO COMMENTS)
```python
def generate_features(df, timestamp):
    df = df.sort_values(['hex7_id', 'timestamp'])
    
    lag_hours = [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168]
    for lag in lag_hours:
        df[f'lag_{lag}h'] = df.groupby('hex7_id')['pm25'].shift(lag)
    
    windows = [3, 6, 12, 24, 48]
    for window in windows:
        df[f'rolling_mean_{window}h'] = df.groupby('hex7_id')['pm25'].transform(
            lambda x: x.rolling(window, min_periods=window//2).mean()
        )
    
    return df
```

### Model Loading (1-6 hours only)
```python
horizons = ['1h', '2h', '3h', '4h', '5h', '6h']
models = {}
for horizon in horizons:
    models[horizon] = joblib.load(f'models_{horizon}.pkl')
```

### Inference Pipeline with Time Selection
1. User selects date/time via Streamlit widgets
2. Get data for selected timestamp
3. Apply spatial imputation if needed
4. Compute all 69 required features
5. Make predictions for next 1-6 hours
6. Display with activity thresholds

## Code Style Requirements
- **NO COMMENTS** in any Python code
- Self-explanatory variable and function names
- Clean, concise implementation
- Type hints where helpful for clarity

## Critical Success Factors
- ✅ NO COMMENTS in code (clean, self-explanatory)
- ✅ Date/time selector for historical analysis
- ✅ K-NN spatial imputation from nearby hexagons
- ✅ Predict only 1-6 hours ahead
- ✅ Use existing trained models from smoglens-02
- ✅ Match exact feature engineering from training
- ✅ Real-time and historical prediction capability

## Next Steps
1. Create V2_working directory structure
2. Implement spatial imputation with K-NN (no comments)
3. Build data processor with neighbor caching
4. Create feature generator matching training
5. Add date/time selector to Streamlit UI
6. Load only 1-6h models for predictions
7. Integrate predictions based on selected time
8. Test with various dates in dataset range

This plan delivers a production-ready system with temporal flexibility and robust spatial imputation, all with clean, comment-free code.