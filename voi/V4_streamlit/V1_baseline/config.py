from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = Path('/Users/vojtech/Code/Bard89/smoglens-data')
MODEL_DIR = DATA_DIR / 'models'

DATA_PATH = DATA_DIR / 'pm25_enriched_2023_2025_v4_20250830_222050.csv'

MODEL_PATHS = {
    1: MODEL_DIR / 'baseline_v2_lgb_1h_20250901_024347.pkl',
    6: MODEL_DIR / 'baseline_v2_lgb_6h_20250901_024347.pkl',
    12: MODEL_DIR / 'baseline_v2_lgb_12h_20250901_024347.pkl',
    24: MODEL_DIR / 'baseline_v2_lgb_24h_20250901_024347.pkl'
}

METADATA_PATH = MODEL_DIR / 'baseline_v2_metadata_20250901_024347.pkl'

COVERAGE_THRESHOLD = 0.90
MIN_DURATION_DAYS = 100

AVAILABLE_HORIZONS = [1, 6, 12, 24]

BASE_FEATURES = [
    'pm25_current', 'hex7_encoded',
    'pm25_lag_1h', 'pm25_lag_2h', 'pm25_lag_3h', 'pm25_lag_6h',
    'pm25_lag_12h', 'pm25_lag_24h', 'pm25_lag_48h', 'pm25_lag_168h',
    'pm25_rolling_mean_3h', 'pm25_rolling_mean_6h', 
    'pm25_rolling_mean_12h', 'pm25_rolling_mean_24h',
    'pm25_rolling_std_3h', 'pm25_rolling_std_6h',
    'pm25_rolling_std_12h', 'pm25_rolling_std_24h',
    'temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean',
    'avg_traffic_volume', 'congestion_index',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
    'is_weekend', 'month_sin', 'month_cos',
    'data_completeness_score', 'pm25_was_missing'
]