import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

if os.getenv("STREAMLIT_CLOUD"):
    DATA_PATH = BASE_DIR / "data" / "shibuya_2024.csv.gz"
    MODEL_DIR = BASE_DIR / "models"
else:
    from smoglens.config import ENRICHED_DATASET, ENSEMBLE_MODEL_DIR

    DATA_PATH = ENRICHED_DATASET
    MODEL_DIR = ENSEMBLE_MODEL_DIR
