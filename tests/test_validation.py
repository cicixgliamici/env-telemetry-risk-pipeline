import numpy as np
import pandas as pd
import pytest

from src.config import NUMERIC_COLUMNS, TELEMETRY_COLUMNS
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
    df["humidity_pct"] = df["humidity_pct"].astype(object)
    df.loc[0, "humidity_pct"] = "not-a-number"

    with pytest.raises(DataValidationError, match="humidity_pct"):
        validate_dataset_schema(df)


def test_validate_dataset_schema_rejects_invalid_timestamps(valid_dataset_df):
    df = valid_dataset_df.copy()
    df.loc[0, "timestamp"] = "not-a-date"

    with pytest.raises(DataValidationError, match="timestamp"):
        validate_dataset_schema(df)


@pytest.mark.parametrize("purpose", ["prediction", "anomaly"])
@pytest.mark.parametrize(
    "labels", [[], ["risk_level"], ["failure_risk"], ["risk_level", "failure_risk"]]
)
def test_scoring_validation_accepts_single_row_and_optional_labels(
    valid_dataset_df, purpose, labels
):
    dataset = valid_dataset_df.iloc[[0]].drop(
        columns=list({"risk_level", "failure_risk"} - set(labels))
    )

    result = validate_dataset_schema(dataset, purpose=purpose)

    assert len(result) == 1
    assert set(result.columns) == set(dataset.columns)


def test_training_validation_does_not_require_failure_label(valid_dataset_df):
    result = validate_dataset_schema(valid_dataset_df.drop(columns="failure_risk"))

    assert "risk_level" in result.columns
    assert "failure_risk" not in result.columns


def test_training_validation_requires_target(valid_dataset_df):
    with pytest.raises(DataValidationError, match="risk_level"):
        validate_dataset_schema(valid_dataset_df.drop(columns="risk_level"))


def test_training_validation_rejects_missing_target(valid_dataset_df):
    valid_dataset_df.loc[0, "risk_level"] = np.nan

    with pytest.raises(DataValidationError, match="risk_level.*missing"):
        validate_dataset_schema(valid_dataset_df)


@pytest.mark.parametrize("column", sorted(NUMERIC_COLUMNS))
@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_validation_rejects_infinite_values(valid_dataset_df, column, infinity):
    dataset = valid_dataset_df.astype({column: float})
    dataset.loc[0, column] = infinity

    with pytest.raises(DataValidationError, match=f"{column}.*infinite"):
        validate_dataset_schema(dataset)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("humidity_pct", -0.1),
        ("humidity_pct", 100.1),
        ("temperature_c", -273.16),
        ("rainfall_mm", -0.1),
        ("pollution_proxy", -0.1),
        ("salinity_proxy", -0.1),
        ("pressure_hpa", 0),
        ("pressure_hpa", -1),
        ("vibration_mm_s", -0.1),
        ("exposure_hours", -0.1),
    ],
)
def test_validation_rejects_out_of_range_measurements(valid_dataset_df, column, value):
    dataset = valid_dataset_df.astype({column: float})
    dataset.loc[0, column] = value

    with pytest.raises(DataValidationError, match=column):
        validate_dataset_schema(dataset, purpose="prediction")


def test_validation_accepts_physical_boundaries_and_missing_measurements(valid_dataset_df):
    dataset = valid_dataset_df.astype({column: float for column in TELEMETRY_COLUMNS})
    dataset.loc[0, list(TELEMETRY_COLUMNS)] = np.nan
    dataset.loc[1, "humidity_pct"] = 0
    dataset.loc[2, "humidity_pct"] = 100
    dataset.loc[1, "temperature_c"] = -273.15
    dataset.loc[1, "rainfall_mm"] = 0
    dataset.loc[1, "vibration_mm_s"] = 0
    dataset.loc[1, "exposure_hours"] = 0
    dataset.loc[1, "pressure_hpa"] = 0.1
    dataset.loc[1, "pollution_proxy"] = 1000
    dataset.loc[1, "salinity_proxy"] = 1000

    result = validate_dataset_schema(dataset)

    assert result.loc[0, list(TELEMETRY_COLUMNS)].isna().all()
    assert result.loc[1, "pollution_proxy"] == 1000


@pytest.mark.parametrize("value", [-1, 0.5])
@pytest.mark.parametrize("column", ["risk_level", "failure_risk"])
def test_validation_rejects_invalid_label_classes(valid_dataset_df, column, value):
    dataset = valid_dataset_df.astype({column: float})
    dataset.loc[0, column] = value

    with pytest.raises(DataValidationError, match=f"{column}.*integer"):
        validate_dataset_schema(dataset)


def test_scoring_validation_allows_missing_optional_annotations(valid_dataset_df):
    valid_dataset_df.loc[0, "risk_level"] = np.nan

    result = validate_dataset_schema(valid_dataset_df, purpose="prediction")

    assert pd.isna(result.loc[0, "risk_level"])


def test_validation_rejects_missing_timestamps(valid_dataset_df):
    valid_dataset_df.loc[0, "timestamp"] = None

    with pytest.raises(DataValidationError, match="timestamp.*missing"):
        validate_dataset_schema(valid_dataset_df)


def test_validation_rejects_duplicate_normalized_headers(valid_dataset_df):
    valid_dataset_df["Humidity Pct"] = valid_dataset_df["humidity_pct"]

    with pytest.raises(DataValidationError, match="duplicate normalized"):
        validate_dataset_schema(valid_dataset_df)


def test_validation_rejects_unknown_purpose(valid_dataset_df):
    with pytest.raises(DataValidationError, match="Unknown validation purpose"):
        validate_dataset_schema(valid_dataset_df, purpose="unsupported")
