from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

SHIBUYA_LAT = 35.6580
SHIBUYA_LON = 139.7016
SHIBUYA_HEXAGON = '872e44d04ffffff'

DATA_PATH = '/Users/vojtech/Code/Bard89/smoglens-02/data/pm25_enriched_2023_2025_v4_20250830_222050.csv'

ACTIVITY_LIMITS = {
    'Running': 10,
    'Baby': 15,
    'Walking': 30,
    'Sitting': 40,
    'Car': 55
}

HORIZONS = ['1h', '2h', '3h', '4h', '5h', '6h']

LAG_HOURS = [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 48]
EWM_ALPHAS = [0.1, 0.3, 0.5]

WEATHER_COLS = ['temperature_c_mean', 'humidity_pct_mean', 'pressure_hpa_mean', 
                'cloud_cover_pct_mean']
TRAFFIC_COLS = ['avg_traffic_volume', 'congestion_index']

K_NEIGHBORS = 10
MAX_SEARCH_RADIUS = 3

PM25_CAP = 53.0