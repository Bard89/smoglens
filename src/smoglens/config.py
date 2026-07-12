import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path(os.getenv("SMOGLENS_DATA_PATH", "/Users/vojtech/Code/Bard89/smoglens-data")).expanduser()

ENRICHED_DATASET = DATA_ROOT / "pm25_enriched_2023_2025_v4_20250830_222050.csv"
ENSEMBLE_MODEL_DIR = DATA_ROOT / "models" / "ensemble_v4"
CACHE_DIR = DATA_ROOT / "cache"
