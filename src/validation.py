from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LABEL_COLUMNS,
    NUMERIC_COLUMNS,
    NUMERIC_RANGES,
    TARGET_COLUMN,
    TELEMETRY_COLUMNS,
)
from src.preprocessing import normalize_column_names


class ValidationPurpose(StrEnum):
    """Choose the label requirements for the operation that will consume the data."""

    TRAINING = "training"
    PREDICTION = "prediction"
    ANOMALY = "anomaly"


class DataValidationError(ValueError):
    """Raised when the input dataset is missing required structure."""


def validate_file_exists(path: str | Path) -> Path:
    """Validate that the input path points to an existing file."""

    resolved = Path(path)
    if not resolved.exists():
        raise DataValidationError(f"Dataset not found: {resolved}")
    if resolved.is_dir():
        raise DataValidationError(f"Expected a file but received a directory: {resolved}")
    return resolved


def _validate_numeric_columns(dataset: pd.DataFrame) -> None:
    """Convert known numeric columns and reject infinity while retaining missing inputs."""

    for column in sorted(NUMERIC_COLUMNS & set(dataset.columns)):
        try:
            dataset[column] = pd.to_numeric(dataset[column])
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"Column '{column}' must be numeric.") from exc
        # NaN inputs belong to the fitted imputer; infinity is never a missing value.
        if np.isinf(dataset[column]).any():
            raise DataValidationError(f"Column '{column}' must not contain infinite values.")


def _validate_physical_ranges(dataset: pd.DataFrame) -> None:
    """Enforce broad physical bounds without inventing sensor-specific limits."""

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = dataset[column].dropna()
        if values.lt(minimum).any():
            raise DataValidationError(
                f"Column '{column}' must be greater than or equal to {minimum}."
            )
        if maximum is not None and values.gt(maximum).any():
            raise DataValidationError(f"Column '{column}' must be less than or equal to {maximum}.")
    # Atmospheric pressure is strictly positive, unlike rainfall or exposure duration.
    if dataset["pressure_hpa"].le(0).any():
        raise DataValidationError("Column 'pressure_hpa' must be greater than 0.")


def _validate_timestamps(dataset: pd.DataFrame) -> None:
    """Reject invalid and missing timestamps instead of silently retaining NaT."""

    try:
        dataset["timestamp"] = pd.to_datetime(
            dataset["timestamp"],
            errors="raise",
            format="mixed",
        )
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Column 'timestamp' must contain valid datetimes.") from exc
    if dataset["timestamp"].isna().any():
        raise DataValidationError("Column 'timestamp' must not contain missing values.")


def _validate_labels(dataset: pd.DataFrame, purpose: ValidationPurpose) -> None:
    """Allow optional annotations in scoring, but never impute training targets."""

    for column in sorted(LABEL_COLUMNS & set(dataset.columns)):
        values = dataset[column].dropna()
        if values.lt(0).any() or values.mod(1).ne(0).any():
            raise DataValidationError(
                f"Column '{column}' must contain non-negative integer classes."
            )
    if purpose == ValidationPurpose.TRAINING:
        if dataset[TARGET_COLUMN].isna().any():
            raise DataValidationError(
                f"Target column '{TARGET_COLUMN}' must not contain missing values."
            )
        if dataset[TARGET_COLUMN].nunique() < 2:
            raise DataValidationError(
                f"Target column '{TARGET_COLUMN}' must contain at least two classes."
            )


def validate_dataset_schema(
    df: pd.DataFrame, *, purpose: str | ValidationPurpose = ValidationPurpose.TRAINING
) -> pd.DataFrame:
    """Validate telemetry plus the label requirements of the requested operation."""

    try:
        purpose = ValidationPurpose(purpose)
    except ValueError as exc:
        raise DataValidationError(f"Unknown validation purpose: {purpose}") from exc
    normalized = normalize_column_names(df)
    # Normalization must not turn distinct headers into ambiguous duplicate columns.
    if normalized.columns.duplicated().any():
        raise DataValidationError("Dataset contains duplicate normalized column names.")
    required = TELEMETRY_COLUMNS | {"timestamp"}
    if purpose == ValidationPurpose.TRAINING:
        required = required | {TARGET_COLUMN}
    missing_columns = sorted(required - set(normalized.columns))
    if missing_columns:
        raise DataValidationError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )
    if normalized.empty:
        raise DataValidationError("Dataset is empty.")
    _validate_numeric_columns(normalized)
    _validate_physical_ranges(normalized)
    _validate_timestamps(normalized)
    _validate_labels(normalized, purpose)
    return normalized
