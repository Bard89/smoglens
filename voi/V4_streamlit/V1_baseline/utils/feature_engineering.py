import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def create_lag_features(df: pd.DataFrame, target_col: str = 'pm25') -> pd.DataFrame:
    df = df.sort_values(['hex7_id', 'timestamp']).reset_index(drop=True)
    
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lag_hours:
        df[f'pm25_lag_{lag}h'] = df.groupby('hex7_id')[target_col].shift(lag)
    
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str = 'pm25') -> pd.DataFrame:
    df['pm25_rolling_mean_3h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    df['pm25_rolling_mean_6h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(6, min_periods=3).mean()
    )
    df['pm25_rolling_mean_12h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(12, min_periods=6).mean()
    )
    df['pm25_rolling_mean_24h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(24, min_periods=12).mean()
    )
    df['pm25_rolling_std_3h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )
    df['pm25_rolling_std_6h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(6, min_periods=3).std()
    )
    df['pm25_rolling_std_12h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(12, min_periods=6).std()
    )
    df['pm25_rolling_std_24h'] = df.groupby('hex7_id')[target_col].transform(
        lambda x: x.rolling(24, min_periods=12).std()
    )
    
    return df

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.month
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def encode_hexagons(df: pd.DataFrame, hex_encoder: LabelEncoder = None) -> tuple:
    if hex_encoder is None:
        hex_encoder = LabelEncoder()
        df['hex7_encoded'] = hex_encoder.fit_transform(df['hex7_id'])
    else:
        df['hex7_encoded'] = hex_encoder.transform(df['hex7_id'])
    
    return df, hex_encoder

def prepare_features(
    df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    hex7_id: str,
    hex_encoder: LabelEncoder = None
) -> pd.DataFrame:
    
    pm25_cap = 74.0
    df['pm25_ugm3_mean'] = df['pm25_ugm3_mean'].clip(upper=pm25_cap)
    
    df['pm25_was_missing'] = df['pm25_ugm3_mean'].isna().astype(int)
    df['pm25'] = df.groupby('hex7_id')['pm25_ugm3_mean'].transform(
        lambda x: x.interpolate(method='linear', limit=3)
    )
    
    df = df.dropna(subset=['pm25'])
    df['pm25_current'] = df['pm25']
    
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_temporal_features(df)
    
    df, hex_encoder = encode_hexagons(df, hex_encoder)
    
    weather_features = ['temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean']
    traffic_features = ['avg_traffic_volume', 'congestion_index']
    
    for feat in weather_features:
        if feat not in df.columns:
            default_val = 15.0 if feat == 'temperature_c_mean' else 70.0 if feat == 'humidity_pct_mean' else 1013.0
            df[feat] = default_val
        else:
            df[feat] = df[feat].fillna(df[feat].median())
    
    for feat in traffic_features:
        df[feat] = df[feat].fillna(0.0) if feat in df.columns else 0.0
    
    if 'data_completeness_score' not in df.columns:
        df['data_completeness_score'] = 1.0
    
    return df, hex_encoder