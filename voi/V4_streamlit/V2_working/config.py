from pathlib import Path
import os

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

SHIBUYA_LAT = 35.6580
SHIBUYA_LON = 139.7016
SHIBUYA_HEXAGON = '872e44d04ffffff'

if os.getenv('STREAMLIT_CLOUD'):
    DATA_PATH = BASE_DIR / 'data' / 'shibuya_2024.csv.gz'
    MODEL_DIR = BASE_DIR / 'models'
else:
    DATA_PATH = '/Users/vojtech/Code/Bard89/smoglens-02/data/pm25_enriched_2023_2025_v4_20250830_222050.csv'
    MODEL_DIR = Path('/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained')

ACTIVITY_LIMITS = {
    'Running': 10,
    'Baby': 15,
    'Walking': 30,
    'Sitting': 40,
    'Car': 55
}

HORIZONS = ['1h', '2h', '3h', '4h', '5h', '6h']

HORIZON_MAE = {
    '1h': 2.5,
    '2h': 2.8,
    '3h': 3.1,
    '4h': 3.4,
    '5h': 3.7,
    '6h': 4.0
}

LAG_HOURS = [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 48]
EWM_ALPHAS = [0.1, 0.3, 0.5]

WEATHER_COLS = ['temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean', 
                'cloud_cover_pct_mean']
TRAFFIC_COLS = ['avg_traffic_volume', 'congestion_index']

K_NEIGHBORS = 10
MAX_SEARCH_RADIUS = 3

PM25_CAP = 53.0