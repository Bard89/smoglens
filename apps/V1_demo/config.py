from pathlib import Path

SHIBUYA_LAT = 35.6580
SHIBUYA_LON = 139.7016
SHIBUYA_HEXAGON = '872e44d04ffffff'

ACTIVITY_LIMITS = {
    'Running': 10,
    'Baby': 15,
    'Walking': 30,
    'Sitting': 40,
    'Car': 55
}

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'shibuya_growth_period.csv.gz'