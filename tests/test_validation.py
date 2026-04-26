import pandas as pd
import pytest

from src.validation import DataValidationError, validate_dataset_schema


def test_validate_dataset_schema_rejects_missing_columns():
    df = pd.DataFrame({"timestamp": ["2025-01-01"]})

    with pytest.raises(DataValidationError):
        validate_dataset_schema(df)


def test_validate_dataset_schema_normalizes_column_names(valid_dataset_df):
    df = valid_dataset_df.rename(
        columns={
            "timestamp": "Timestamp",
            "humidity_pct": "Humidity Pct",
            "temperature_c": "Temperature C",
            "rainfall_mm": "Rainfall Mm",
            "pollution_proxy": "Pollution Proxy",
            "salinity_proxy": "Salinity Proxy",
            "pressure_hpa": "Pressure Hpa",
            "vibration_mm_s": "Vibration Mm S",
            "exposure_hours": "Exposure Hours",
            "risk_level": "Risk Level",
            "failure_risk": "Failure Risk",
        }
    )

    out = validate_dataset_schema(df)

    assert "humidity_pct" in out.columns
    assert str(out["timestamp"].dtype).startswith("datetime64")


def test_validate_dataset_schema_rejects_single_class_target(valid_dataset_df):
    df = valid_dataset_df.copy()
    df["risk_level"] = 1

    with pytest.raises(DataValidationError):
        validate_dataset_schema(df)


def test_validate_dataset_schema_rejects_non_numeric_values(valid_dataset_df):
    df = valid_dataset_df.copy()
    df.loc[0, "humidity_pct"] = "not-a-number"

    with pytest.raises(DataValidationError, match="humidity_pct"):
        validate_dataset_schema(df)
