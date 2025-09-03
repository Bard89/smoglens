import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

SHIBUYA_HEXAGON = '872e44d04ffffff'
SOURCE_DATA = '/Users/vojtech/Code/Bard89/smoglens-data/pm25_enriched_2023_2025_v4_20250830_222050.csv'
OUTPUT_DIR = '/Users/vojtech/Code/Bard89/smoglens/voi/V4_streamlit/V2_ensemble/data'
OUTPUT_PATH = f'{OUTPUT_DIR}/shibuya_pm25_data.csv'

def preprocess_shibuya_data():
    print("="*50)
    print("PREPROCESSING SHIBUYA DATA FOR DEPLOYMENT")
    print("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n1. Loading original dataset from:\n   {SOURCE_DATA}")
    df = pd.read_csv(SOURCE_DATA, parse_dates=['timestamp'])
    print(f"   Loaded {len(df):,} rows, {df['hex7_id'].nunique()} hexagons")
    
    print(f"\n2. Filtering for Shibuya hexagon: {SHIBUYA_HEXAGON}")
    shibuya_data = df[df['hex7_id'] == SHIBUYA_HEXAGON].copy()
    print(f"   Found {len(shibuya_data):,} rows for Shibuya")
    
    print("\n3. Setting timestamp index and resampling to hourly")
    numeric_cols = ['pm25_ugm3_mean', 'temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean',
                    'avg_traffic_volume', 'congestion_index', 'lat', 'lon']
    numeric_cols = [col for col in numeric_cols if col in shibuya_data.columns]
    
    shibuya_data = shibuya_data.set_index('timestamp').sort_index()
    original_len = len(shibuya_data)
    shibuya_data = shibuya_data[numeric_cols].resample('1h').mean()
    print(f"   Resampled from {original_len:,} to {len(shibuya_data):,} hourly rows")
    
    print("\n4. Filling gaps in PM2.5 data")
    missing_before = shibuya_data['pm25_ugm3_mean'].isna().sum()
    shibuya_data['pm25_ugm3_mean'] = shibuya_data['pm25_ugm3_mean'].interpolate(
        method='linear', limit=3
    ).bfill().ffill()
    missing_after = shibuya_data['pm25_ugm3_mean'].isna().sum()
    print(f"   Filled {missing_before - missing_after} missing values")
    print(f"   Remaining missing: {missing_after}")
    
    print("\n5. Creating features for model inference")
    
    for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168]:
        shibuya_data[f'pm25_lag_{lag}h'] = shibuya_data['pm25_ugm3_mean'].shift(lag)
    print(f"   ✓ Created 11 lag features")
    
    for window in [3, 6, 12, 24]:
        shibuya_data[f'pm25_rolling_mean_{window}h'] = shibuya_data['pm25_ugm3_mean'].rolling(window, min_periods=1).mean()
        shibuya_data[f'pm25_rolling_std_{window}h'] = shibuya_data['pm25_ugm3_mean'].rolling(window, min_periods=1).std()
    print(f"   ✓ Created 8 rolling features")
    
    shibuya_data['hour'] = shibuya_data.index.hour
    shibuya_data['day_of_week'] = shibuya_data.index.dayofweek
    shibuya_data['month'] = shibuya_data.index.month
    shibuya_data['is_weekend'] = (shibuya_data['day_of_week'] >= 5).astype(int)
    
    shibuya_data['hour_sin'] = np.sin(2 * np.pi * shibuya_data['hour'] / 24)
    shibuya_data['hour_cos'] = np.cos(2 * np.pi * shibuya_data['hour'] / 24)
    shibuya_data['dow_sin'] = np.sin(2 * np.pi * shibuya_data['day_of_week'] / 7)
    shibuya_data['dow_cos'] = np.cos(2 * np.pi * shibuya_data['day_of_week'] / 7)
    shibuya_data['month_sin'] = np.sin(2 * np.pi * shibuya_data['month'] / 12)
    shibuya_data['month_cos'] = np.cos(2 * np.pi * shibuya_data['month'] / 12)
    print(f"   ✓ Created temporal features")
    
    weather_cols = ['temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean']
    for col in weather_cols:
        if col in df.columns:
            shibuya_data[col] = df[df['hex7_id'] == SHIBUYA_HEXAGON].set_index('timestamp')[col].resample('1h').mean()
            shibuya_data[col] = shibuya_data[col].fillna(shibuya_data[col].median())
    print(f"   ✓ Added weather features")
    
    print("\n6. Keeping last 2 years of data for context")
    cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=730)
    before_cutoff = len(shibuya_data)
    shibuya_data = shibuya_data[shibuya_data.index >= cutoff_date]
    print(f"   Reduced from {before_cutoff:,} to {len(shibuya_data):,} rows")
    
    print("\n7. Removing remaining NaN values")
    before_drop = len(shibuya_data)
    shibuya_data = shibuya_data.dropna()
    print(f"   Dropped {before_drop - len(shibuya_data)} rows with NaN")
    
    print("\n8. Resetting index and adding hexagon ID")
    shibuya_data = shibuya_data.reset_index()
    shibuya_data['hex7_id'] = SHIBUYA_HEXAGON
    
    print(f"\n9. Saving preprocessed data")
    shibuya_data.to_csv(OUTPUT_PATH, index=False, compression='gzip')
    
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n✅ SUCCESS!")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"   File size: {file_size_mb:.2f} MB (compressed)")
    print(f"   Total rows: {len(shibuya_data):,}")
    print(f"   Date range: {shibuya_data['timestamp'].min()} to {shibuya_data['timestamp'].max()}")
    print(f"   Features: {len(shibuya_data.columns)} columns")
    
    return shibuya_data

if __name__ == "__main__":
    data = preprocess_shibuya_data()
    print("\n" + "="*50)
    print("Sample of preprocessed data:")
    print(data[['timestamp', 'pm25_ugm3_mean', 'pm25_lag_1h', 'hour', 'is_weekend']].tail())
    print("="*50)