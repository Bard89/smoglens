import numpy as np
import pandas as pd
import pytest

from smoglens.config import EWM_ALPHAS, LAG_HOURS, ROLLING_WINDOWS
from smoglens.features import FeatureGenerator


@pytest.fixture(scope="module")
def synthetic_features():
    timestamps = pd.date_range("2024-01-01 00:00", periods=200, freq="h", tz="Asia/Tokyo")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pm25": np.arange(200, dtype="float64"),
            "temperature_c_mean": np.full(200, 20.0),
            "humidity_pct_mean": np.full(200, 50.0),
            "avg_traffic_volume": np.full(200, 100.0),
        }
    )
    return FeatureGenerator().generate_features(frame)


def test_all_engineered_columns_present(synthetic_features):
    expected = [f"lag_{lag}h" for lag in LAG_HOURS]
    stats = ["mean", "std", "max", "min"]
    expected += [f"rolling_{stat}_{window}h" for window in ROLLING_WINDOWS for stat in stats]
    expected += [f"ewm_{alpha}" for alpha in EWM_ALPHAS]
    expected += ["diff_1h", "diff_6h", "diff_24h", "rate_6h", "rate_24h"]
    expected += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    expected += [f"{base}_{k}" for k in [1, 2] for base in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]]
    expected += ["is_weekend", "temp_humidity", "temp_hour", "traffic_hour", "traffic_weekend", "hex_encoded"]
    missing = [col for col in expected if col not in synthetic_features.columns]
    assert not missing, missing


def test_lags_shift_pm25(synthetic_features):
    for lag in LAG_HOURS:
        assert synthetic_features[f"lag_{lag}h"].iloc[199] == pytest.approx(199 - lag)
        assert pd.isna(synthetic_features[f"lag_{lag}h"].iloc[lag - 1])


def test_diffs_and_rates(synthetic_features):
    row = synthetic_features.iloc[199]
    assert row["diff_1h"] == pytest.approx(1.0)
    assert row["diff_24h"] == pytest.approx(24.0)
    assert row["rate_6h"] == pytest.approx(1.0)
    assert row["rate_24h"] == pytest.approx(1.0)


def test_rolling_stats_on_linear_ramp(synthetic_features):
    for window in ROLLING_WINDOWS:
        ramp_mean = 199 - (window - 1) / 2
        assert synthetic_features[f"rolling_mean_{window}h"].iloc[199] == pytest.approx(ramp_mean)
        assert synthetic_features[f"rolling_max_{window}h"].iloc[199] == pytest.approx(199)
        assert synthetic_features[f"rolling_min_{window}h"].iloc[199] == pytest.approx(199 - window + 1)


def test_ewm_recursion(synthetic_features):
    for alpha in EWM_ALPHAS:
        assert synthetic_features[f"ewm_{alpha}"].iloc[0] == pytest.approx(0.0)
        assert synthetic_features[f"ewm_{alpha}"].iloc[1] == pytest.approx(alpha)


def test_cyclical_encodings_on_unit_circle(synthetic_features):
    for base in ["hour", "dow", "month"]:
        norms = synthetic_features[f"{base}_sin"] ** 2 + synthetic_features[f"{base}_cos"] ** 2
        assert np.allclose(norms, 1.0, atol=1e-5)
    assert synthetic_features["hour_sin"].equals(synthetic_features["hour_sin_1"])


def test_calendar_flags(synthetic_features):
    monday = synthetic_features["timestamp"].dt.dayofweek == 0
    saturday = synthetic_features["timestamp"].dt.dayofweek == 5
    assert (synthetic_features.loc[monday, "is_weekend"] == 0).all()
    assert (synthetic_features.loc[saturday, "is_weekend"] == 1).all()


def test_interactions(synthetic_features):
    row = synthetic_features.iloc[199]
    assert row["temp_humidity"] == pytest.approx(20.0 * 50.0 / 100)
    assert row["temp_hour"] == pytest.approx(20.0 * row["hour_sin"], rel=1e-6)
    assert row["traffic_weekend"] == pytest.approx(100.0 * row["is_weekend"])
    assert (synthetic_features["hex_encoded"] == 0).all()
