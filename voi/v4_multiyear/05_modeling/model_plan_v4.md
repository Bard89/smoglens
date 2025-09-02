# PM2.5 Prediction Model Plan V4

## Model Architecture Recommendation

### Proposed Model: Ensemble Approach

For predicting PM2.5 levels for the next 24 hours across Japan, I recommend an **ensemble of three models**:

1. **LSTM with Attention** - Captures long-term temporal dependencies
2. **LightGBM** - Handles non-linear relationships and feature interactions  
3. **Graph Neural Network (optional)** - Captures spatial dependencies between hexagons

### Prediction Strategy: Direct Multi-Output

- Predict all 24 hours simultaneously (t+1 to t+24)
- More stable than recursive approach
- Avoids error accumulation

## Feature Selection Analysis

### Features to USE (22 base features)

#### Target Variable
- `pm25_ugm3_mean` (current value)

#### Lag Features (to be created)
- PM2.5 at t-1, t-2, t-3, t-6, t-12, t-24 hours
- PM2.5 at t-48 hours (2 days ago)
- PM2.5 at t-168 hours (1 week ago, same hour)
- Rolling means: 3h, 6h, 12h, 24h

#### Traffic Features (2 selected)
- `avg_traffic_volume` ✓
- `congestion_index` ✓
- ~~`avg_speed_kmh`~~ (correlates with congestion)

#### Weather Features (6 selected)
- `temperature_c_mean` ✓
- `humidity_pct_mean` ✓
- `pressure_hpa_mean` ✓
- `cloud_cover_pct_mean` ✓
- `solar_radiation_wm2_mean` ✓
- `is_raining` ✓ (binary, simpler than precipitation_mm)
- ~~`dew_point_c_mean`~~ (highly correlated with temperature+humidity)

#### Temporal Features (all cyclical)
- `hour_sin`, `hour_cos` ✓
- `dow_sin`, `dow_cos` ✓
- `month_sin`, `month_cos` ✓
- `is_weekend` ✓

#### Spatial Features
- `lat`, `lon` ✓ (or use hex7_id as categorical embedding)

#### Data Quality (optional but useful)
- `data_completeness_score` ✓

### Features to DROP

#### Redundant PM2.5 Features
- ❌ `pm25_ugm3_min`, `pm25_ugm3_max` (captured by mean)
- ❌ `pm25_range` (derived)
- ❌ `pm25_ugm3_count` (metadata)

#### Redundant Traffic Features
- ❌ `max_traffic_volume` (correlates with avg)
- ❌ `traffic_measurement_count`, `unique_links` (metadata)
- ❌ `traffic_intensity` (derived from avg_traffic_volume)
- ❌ `avg_speed_kmh` (inverse of congestion)

#### Redundant Weather Features
- ❌ `heat_index` (derived from temp+humidity)
- ❌ `precipitation_mm_mean` (using is_raining instead)
- ❌ `dew_point_c_mean` (function of temp+humidity)

#### Raw Temporal Features
- ❌ `hour`, `day_of_week`, `month`, `year` (using cyclical encodings)

#### Distance Features (unless modeling uncertainty)
- ❌ `traffic_distance_km`, `weather_distance_km`

## Feature Engineering Requirements

### Lag Features Creation
```python
lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
for lag in lag_hours:
    df[f'pm25_lag_{lag}h'] = df.groupby('hex7_id')['pm25_ugm3_mean'].shift(lag)
```

### Rolling Statistics
```python
rolling_windows = [3, 6, 12, 24]
for window in rolling_windows:
    df[f'pm25_rolling_mean_{window}h'] = df.groupby('hex7_id')['pm25_ugm3_mean'].rolling(window).mean()
    df[f'pm25_rolling_std_{window}h'] = df.groupby('hex7_id')['pm25_ugm3_mean'].rolling(window).std()
```

### Spatial Neighbor Features
- Average PM2.5 of K-nearest hexagons
- Distance-weighted average based on geographic proximity
- Consider hexagons within 10km radius

## Feature Importance Analysis Requirements

1. **Calculate VIF (Variance Inflation Factor)** to detect multicollinearity
2. **Perform mutual information analysis** between features and target
3. **Use SHAP values** from a baseline model to understand feature importance
4. **Create lag correlation plots** to determine optimal lag windows
5. **Analyze spatial autocorrelation** using Moran's I

## Final Model Pipeline

### 1. Feature Engineering
- Create lag features (t-1 to t-168)
- Calculate rolling statistics
- Add spatial neighbor features

### 2. Feature Selection (22 base + 12 engineered = 34 total)
- Current PM2.5
- 8 lag features
- 4 rolling statistics
- 2 traffic features
- 6 weather features
- 7 temporal features
- 2 spatial features
- 1 quality score

### 3. Model Training
- **Train/Val Split**: 2023-2024 for training, 2025 for validation
- **Cross-validation**: Use sliding window for time series CV
- **Ensemble Method**: Train ensemble with weighted averaging
- **Loss Function**: MSE with optional weighting for peak hours

### 4. Output
- 24 hourly predictions per hexagon
- Uncertainty estimates (prediction intervals)
- Feature importance scores
- Model performance metrics (MAE, RMSE, R²)

## Implementation Steps

1. **Data Preparation**
   - Load enriched dataset
   - Create lag and rolling features
   - Handle missing values with forward fill for lags
   - Split data temporally

2. **Model Development**
   - Implement LSTM with attention mechanism
   - Train LightGBM with early stopping
   - Optional: Implement GNN for spatial modeling
   - Create ensemble weights using validation set

3. **Evaluation**
   - Evaluate on multiple metrics (MAE, RMSE, MAPE)
   - Analyze performance by time horizon (1h, 6h, 12h, 24h)
   - Create error analysis by location and time
   - Compare with baseline models (persistence, ARIMA)

4. **Deployment**
   - Save trained models
   - Create prediction pipeline
   - Implement real-time feature engineering
   - Set up monitoring for model drift

## Model Specifications

### LSTM Architecture
```python
Input -> LSTM(128) -> LSTM(64) -> Attention -> Dense(24)
- Dropout: 0.2
- Learning rate: 0.001
- Batch size: 256
- Epochs: 100 with early stopping
```

### LightGBM Parameters
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_threads': 8
}
```

### Ensemble Weights
- Initial: Equal weights (0.33, 0.33, 0.34)
- Optimized: Based on validation performance
- Dynamic: Weight by prediction horizon

## Expected Performance

Based on similar studies and data characteristics:
- **1-hour ahead**: MAE ~2-3 μg/m³, R² ~0.85
- **6-hour ahead**: MAE ~4-5 μg/m³, R² ~0.70
- **12-hour ahead**: MAE ~5-7 μg/m³, R² ~0.60
- **24-hour ahead**: MAE ~6-8 μg/m³, R² ~0.50

## Notes

- This approach balances model complexity with interpretability
- Selected features avoid redundancy and multicollinearity
- Direct multi-output strategy is preferred for stability
- Ensemble approach leverages strengths of different model types
- Feature engineering is crucial for capturing temporal patterns