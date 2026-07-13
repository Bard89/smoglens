from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from smoglens.config import ENRICHED_DATASET, ENSEMBLE_MODEL_DIR

REPO_ROOT = Path(__file__).parents[1]
APP_DIR = REPO_ROOT / "apps" / "streamlit"


@pytest.fixture(scope="module")
def local_data_available():
    cache_path = APP_DIR / "data" / "shibuya_processed.parquet"
    if not (ENRICHED_DATASET.exists() or cache_path.exists()):
        pytest.skip("local dataset not available")
    if not (ENSEMBLE_MODEL_DIR / "models_1h.pkl").exists():
        pytest.skip("trained models not available")


def test_app_runs_without_uncaught_exception(local_data_available):
    app_test = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=300)
    app_test.run()
    assert not app_test.exception


def test_app_prediction_path_renders_without_errors(local_data_available):
    app_test = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=300)
    app_test.run()
    app_test.checkbox[0].uncheck().run()
    app_test.date_input[0].set_value(date(2024, 6, 1))
    app_test.run()
    assert not app_test.exception
    assert not app_test.error
