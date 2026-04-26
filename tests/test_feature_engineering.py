import pandas as pd

from src.feature_engineering import add_engineered_features


def test_feature_engineering_adds_expected_columns():
    df = pd.DataFrame(
        {
            "humidity_pct": [80],
            "temperature_c": [30],
            "rainfall_mm": [5],
            "pollution_proxy": [40],
            "salinity_proxy": [30],
            "pressure_hpa": [1010],
            "vibration_mm_s": [1.2],
            "exposure_hours": [100],
        }
    )

    out = add_engineered_features(df)

    assert "humidity_temp_index" in out.columns
    assert "vibration_exposure_index" in out.columns


def test_feature_engineering_preserves_row_count(valid_dataset_df):
    out = add_engineered_features(valid_dataset_df)

    assert len(out) == len(valid_dataset_df)
    assert out["high_humidity_flag"].isin([0, 1]).all()
