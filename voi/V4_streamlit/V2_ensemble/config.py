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

MODEL_MSE = {
    '1h': 6.0,
    '2h': 9.1,
    '3h': 11.7,
    '4h': 14.0,
    '5h': 16.0,
    '6h': 17.8
}

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'shibuya_growth_period.csv.gz'

BASELINE_MODEL_PATH = Path('/Users/vojtech/Code/Bard89/smoglens-data/models/baseline_v2_lgb_1h_20250901_024347.pkl')
ENSEMBLE_MODEL_DIR = Path('/Users/vojtech/Code/Bard89/smoglens-02/voi/v4_multiyear/05_modeling/02_advanced/01_ensemble_training/trained')

ENSEMBLE_MODELS = {
    '1h': ENSEMBLE_MODEL_DIR / 'models_1h.pkl',
    '2h': ENSEMBLE_MODEL_DIR / 'models_2h.pkl', 
    '3h': ENSEMBLE_MODEL_DIR / 'models_3h.pkl',
    '4h': ENSEMBLE_MODEL_DIR / 'models_4h.pkl',
    '5h': ENSEMBLE_MODEL_DIR / 'models_5h.pkl',
    '6h': ENSEMBLE_MODEL_DIR / 'models_6h.pkl',
    '12h': ENSEMBLE_MODEL_DIR / 'models_12h.pkl'
}
