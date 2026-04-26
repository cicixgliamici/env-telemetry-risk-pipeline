import pandas as pd

from src.preprocessing import clean_dataset, split_features_target


def test_clean_dataset_fills_missing_and_sorts():
    df = pd.DataFrame(
        {
            "timestamp": ["2025-01-02", "2025-01-01"],
            "humidity_pct": [50.0, None],
            "risk_level": [1, 1],
            "failure_risk": [1, 1],
        }
    )

    cleaned = clean_dataset(df)

    assert cleaned["humidity_pct"].isna().sum() == 0
    assert str(cleaned.iloc[0]["timestamp"]).startswith("2025-01-01")


def test_split_features_target_drops_leakage_columns(valid_dataset_df):
    X_train, X_test, y_train, y_test = split_features_target(
        valid_dataset_df,
        target_column="risk_level",
        test_size=0.25,
        random_state=42,
    )

    for frame in [X_train, X_test]:
        assert "timestamp" not in frame.columns
        assert "failure_risk" not in frame.columns
        assert "risk_level" not in frame.columns

    assert len(y_train) + len(y_test) == len(valid_dataset_df)
