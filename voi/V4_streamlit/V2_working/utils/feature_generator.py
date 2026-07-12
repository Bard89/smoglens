import numpy as np
import config

class FeatureGenerator:
    def __init__(self):
        self.feature_cols = None
        
    def generate_features(self, df):
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if 'pm25' not in df.columns:
            if 'pm25_ugm3_mean' in df.columns:
                df['pm25'] = df['pm25_ugm3_mean'].copy()
            else:
                df['pm25'] = 0.0
        
        df['pm25'] = df['pm25'].ffill(limit=3)
        df['pm25'] = df['pm25'].bfill(limit=3)
        
        for lag in config.LAG_HOURS:
            df[f'lag_{lag}h'] = df['pm25'].shift(lag).astype('float32')
        
        df['diff_1h'] = (df['pm25'] - df['lag_1h']).astype('float32')
        df['diff_6h'] = (df['pm25'] - df['lag_6h']).astype('float32')
        df['diff_24h'] = (df['pm25'] - df['lag_24h']).astype('float32')
        df['rate_6h'] = (df['diff_6h'] / 6).astype('float32')
        df['rate_24h'] = (df['diff_24h'] / 24).astype('float32')
        
        for window in config.ROLLING_WINDOWS:
            df[f'rolling_mean_{window}h'] = df['pm25'].rolling(
                window, min_periods=window//2).mean().astype('float32')
            df[f'rolling_std_{window}h'] = df['pm25'].rolling(
                window, min_periods=window//2).std().astype('float32')
            df[f'rolling_max_{window}h'] = df['pm25'].rolling(
                window, min_periods=window//2).max().astype('float32')
            df[f'rolling_min_{window}h'] = df['pm25'].rolling(
                window, min_periods=window//2).min().astype('float32')
        
        for alpha in config.EWM_ALPHAS:
            df[f'ewm_{alpha}'] = df['pm25'].ewm(
                alpha=alpha, adjust=False).mean().astype('float32')
        
        df['hour'] = df['timestamp'].dt.hour.astype('int8')
        df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
        df['month'] = df['timestamp'].dt.month.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).astype('float32')
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).astype('float32')
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype('float32')
        
        for k in [1, 2]:
            df[f'hour_sin_{k}'] = np.sin(2 * np.pi * k * df['hour'] / 24).astype('float32')
            df[f'hour_cos_{k}'] = np.cos(2 * np.pi * k * df['hour'] / 24).astype('float32')
            df[f'dow_sin_{k}'] = np.sin(2 * np.pi * k * df['day_of_week'] / 7).astype('float32')
            df[f'dow_cos_{k}'] = np.cos(2 * np.pi * k * df['day_of_week'] / 7).astype('float32')
        
        for col in config.WEATHER_COLS + config.TRAFFIC_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median()).astype('float32')
        
        if 'temperature_c_mean' in df.columns and 'humidity_pct_mean' in df.columns:
            df['temp_humidity'] = (df['temperature_c_mean'] * df['humidity_pct_mean'] / 100).astype('float32')
            df['temp_hour'] = (df['temperature_c_mean'] * df['hour_sin']).astype('float32')
        
        if 'avg_traffic_volume' in df.columns:
            df['traffic_hour'] = (df['avg_traffic_volume'] * df['hour_sin']).astype('float32')
            df['traffic_weekend'] = (df['avg_traffic_volume'] * df['is_weekend']).astype('float32')
        
        df['hex_encoded'] = 0
        
        self.feature_cols = self.get_feature_columns(df)
        
        return df
    
    def get_feature_columns(self, df):
        feature_cols = []
        
        for col in df.columns:
            if 'lag_' in col or 'rolling_' in col or 'ewm' in col:
                feature_cols.append(col)
            elif 'diff_' in col or 'rate_' in col:
                feature_cols.append(col)
            elif '_sin' in col or '_cos' in col:
                feature_cols.append(col)
            elif 'temp_' in col or 'traffic_' in col:
                feature_cols.append(col)
            elif col in ['hex_encoded', 'is_weekend']:
                feature_cols.append(col)
            elif col in config.WEATHER_COLS or col in config.TRAFFIC_COLS:
                if col in df.columns:
                    feature_cols.append(col)
        
        for col in ['avg_traffic_volume', 'max_traffic_volume', 'congestion_index',
                    'traffic_measurement_count', 'traffic_distance_km', 'traffic_intensity',
                    'temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean',
                    'cloud_cover_pct_mean']:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)
        
        return feature_cols