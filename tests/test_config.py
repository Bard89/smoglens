import hashlib
import json
import pickle
from pathlib import Path

import pytest

from smoglens.config import DATA_ROOT, ENRICHED_DATASET, ENSEMBLE_MODEL_DIR

FIXTURES = Path(__file__).parent / "fixtures"
HORIZONS = ["1h", "2h", "3h", "4h", "5h", "6h", "12h", "24h"]


@pytest.fixture(scope="module")
def data_root_available():
    if not DATA_ROOT.exists():
        pytest.skip("SMOGLENS_DATA_PATH not available")


def test_data_layout_present(data_root_available):
    assert ENRICHED_DATASET.exists()
    for horizon in HORIZONS:
        assert (ENSEMBLE_MODEL_DIR / f"models_{horizon}.pkl").exists(), horizon


def test_model_files_match_manifest(data_root_available):
    manifest = json.loads((FIXTURES / "model_manifest.json").read_text())
    for name, expected_sha256 in manifest.items():
        actual = hashlib.sha256((ENSEMBLE_MODEL_DIR / name).read_bytes()).hexdigest()
        assert actual == expected_sha256, name


def test_model_metadata_matches_golden_contract(data_root_available):
    with open(ENSEMBLE_MODEL_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    fixture = json.loads((FIXTURES / "golden_predictions.json").read_text())
    assert metadata["feature_cols"] == fixture["feature_cols"]
    assert metadata["target_cols"] == [f"target_{h}" for h in HORIZONS]
    assert metadata["categorical_features"] == ["hex_encoded"]
