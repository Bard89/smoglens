# Implementation Context V4

## Dataset Overview

### Basic Statistics
- **Total Records**: 9,457,112
- **Time Period**: 2023-07-14 to 2025-07-26 (743 days)
- **Unique Hexagons**: 634
- **Unique Timestamps**: 16,004
- **Total Features**: 39
- **Memory Usage**: 3.56 GB

### Temporal Coverage
- **Years**: 2023, 2024, 2025
- **2023**: 2,049,356 records (partial year from July)
- **2024**: 4,700,875 records (complete year)
- **2025**: 2,706,881 records (partial year through July)
- **Months with data**: 25 months
- **Sampling interval**: 1 hour (consistent)

### Geographic Coverage
- **Total hexagons**: 634 (H3 resolution 7)
- **Latitude range**: 24.34° to 45.12°
- **Longitude range**: 124.16° to 144.37°
- **Coverage**: All of Japan including remote islands

## Data Quality Metrics

### Overall Completeness
- **PM2.5 available**: 7,617,538 records (80.5%)
- **Traffic available**: 9,105,653 records (96.3%)
- **Weather available**: 9,091,320 records (96.1%)
- **Complete records** (all three): 7,317,792 (77.4%)

### Missing Data Patterns
- **PM2.5**: 19.45% missing
- **Traffic features**: 3.72% missing
- **Weather features**: 3.87% missing
- **Solar radiation**: 4.19% missing (slightly higher)

### Data Source Quality
- **Local traffic data**: 3.9% of records
- **Local weather data**: 1.1% of records
- **Mean distance to traffic sensor**: 5.9 km
- **Mean distance to weather station**: 38.1 km

### Outlier Detection Results
- **PM2.5**: 286,226 outliers (3.76% of non-null)
- **Traffic volume**: 3,336 outliers (0.04%)
- **Temperature**: 301 outliers (0.003%)

## PM2.5 Statistics

### Distribution
- **Mean**: 9.00 μg/m³
- **Median**: 8.00 μg/m³
- **Std Dev**: 7.99 μg/m³
- **Min**: 0.00 μg/m³
- **Max**: 4,170.00 μg/m³ (extreme outlier)

### Percentiles
- **5th**: 2.00 μg/m³
- **25th**: 5.00 μg/m³
- **50th**: 8.00 μg/m³
- **75th**: 12.00 μg/m³
- **95th**: 21.00 μg/m³
- **99th**: 32.00 μg/m³

### Air Quality Standards
- **WHO Annual Guideline (5 μg/m³)**: 54.1% exceedance
- **WHO 24h Guideline (15 μg/m³)**: 10.4% exceedance
- **Japan Standard (35 μg/m³)**: 0.5% exceedance

### Air Quality Index Distribution
- **Good (0-12 μg/m³)**: 77.6%
- **Moderate (12-35.4 μg/m³)**: 21.8%
- **Unhealthy for Sensitive**: 0.6%
- **Unhealthy+**: <0.1%

## Temporal Patterns

### Monthly Patterns
- **Peak pollution month**: April (Month 4)
- **Lowest pollution month**: August (Month 8)
- **Seasonal trend**: Higher in spring, lower in summer

### Daily Patterns
- **Peak hour**: 8:00 AM (morning rush)
- **Lowest hour**: 5:00 AM (early morning)
- **Secondary peak**: 7:00 PM (evening)
- **Weekend effect**: Lower on weekends (8.5 vs 9.1 μg/m³)

### Year-over-Year Trends
- **2023 mean**: 8.33 μg/m³
- **2024 mean**: 8.51 μg/m³
- **2025 mean**: 10.34 μg/m³ (increasing trend)

## Feature Correlations with PM2.5

### Strong Correlations (|r| > 0.1)
- **pm25_range**: 0.12
- **hour**: -0.09
- **temperature_c_mean**: -0.08
- **solar_radiation**: -0.07

### Weak Correlations (|r| < 0.05)
- **avg_traffic_volume**: 0.03
- **congestion_index**: 0.02
- **pressure_hpa_mean**: 0.01
- **humidity_pct_mean**: -0.02

## Key Findings

### Data Characteristics
1. **High temporal resolution**: Hourly data with good consistency
2. **Spatial coverage**: Good coverage of urban areas, sparse in rural
3. **Seasonal variations**: Clear seasonal and diurnal patterns
4. **Traffic influence**: Weak direct correlation, likely non-linear
5. **Weather influence**: Temperature shows inverse relationship

### Challenges for Modeling
1. **Missing data**: 20% PM2.5 missing, needs imputation strategy
2. **Extreme outliers**: Max PM2.5 of 4170 μg/m³ needs handling
3. **Spatial sparsity**: Many hexagons rely on distant sensors
4. **Non-linear relationships**: Low linear correlations suggest complex interactions
5. **Increasing trend**: 2025 shows higher pollution levels

### Advantages for Modeling
1. **Large dataset**: 9.4M records provide robust training data
2. **Multi-year coverage**: Captures seasonal variations
3. **High completeness**: 77% records have all key features
4. **Consistent sampling**: Regular hourly intervals
5. **Rich features**: 39 features capture various influences

## Files and Resources

### Input Data
- **Enriched dataset**: `pm25_enriched_2023_2025_v4_[timestamp].csv`
- **Size**: ~3.5 GB CSV format
- **Hexagon lookup**: Included in dataset (hex7_id)

### Generated Files
- **EDA notebook**: `pm25_enriched_2023_2025_v4_eda.ipynb`
- **Summary statistics**: `v4_summary_statistics.csv`
- **Model plan**: `model_plan_v4.md`

### Required Processing
1. **Lag feature creation**: 8 lag features needed
2. **Rolling statistics**: 4 window sizes (3h, 6h, 12h, 24h)
3. **Missing value imputation**: Forward fill for time series
4. **Outlier handling**: Cap at 99.9th percentile
5. **Train/test split**: 2023-2024 train, 2025 test

## Implementation Recommendations

### Data Preprocessing
```python
# Handle outliers
pm25_cap = df['pm25_ugm3_mean'].quantile(0.999)
df['pm25_ugm3_mean'] = df['pm25_ugm3_mean'].clip(upper=pm25_cap)

# Create lag features
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df[f'pm25_lag_{lag}h'] = df.groupby('hex7_id')['pm25_ugm3_mean'].shift(lag)

# Forward fill missing values for lags
df = df.groupby('hex7_id').fillna(method='ffill', limit=24)
```

### Feature Scaling
- **Standardize**: PM2.5, traffic, weather features
- **Leave as-is**: Binary features (is_raining, is_weekend)
- **Already scaled**: Cyclical encodings (sin/cos features)

### Model Training Strategy
1. **Validation**: Time-based split, no random shuffling
2. **Early stopping**: Monitor validation MAE, patience=10
3. **Batch size**: 256 for LSTM, full data for LightGBM
4. **Learning rate**: Start with 0.001, reduce on plateau

### Performance Monitoring
- Track MAE by prediction horizon (1h, 6h, 12h, 24h)
- Monitor performance by location (urban vs rural)
- Analyze errors by time of day and season
- Compare with persistence baseline

## Next Steps

1. **Feature Engineering Notebook**: Create lag and rolling features
2. **Feature Selection Analysis**: Calculate VIF and mutual information
3. **Baseline Models**: Implement persistence and ARIMA
4. **Model Implementation**: Start with LightGBM, then LSTM
5. **Ensemble Creation**: Combine models with optimal weights
6. **Error Analysis**: Identify systematic biases
7. **Visualization Dashboard**: Create interactive predictions map