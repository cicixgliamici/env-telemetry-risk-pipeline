from pathlib import Path

import pandas as pd

from src.config import NUMERIC_COLUMNS, REQUIRED_COLUMNS


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


def validate_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the dataset schema expected by the pipeline."""

    normalized = df.copy()
    normalized.columns = [column.strip().lower().replace(" ", "_") for column in normalized.columns]

    missing_columns = sorted(REQUIRED_COLUMNS - set(normalized.columns))
    if missing_columns:
        raise DataValidationError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    for column in sorted(NUMERIC_COLUMNS):
        try:
            normalized[column] = pd.to_numeric(normalized[column])
        except Exception as exc:
            raise DataValidationError(f"Column '{column}' must be numeric.") from exc

    try:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    except Exception as exc:
        raise DataValidationError("Column 'timestamp' must contain valid datetimes.") from exc

    if normalized.empty:
        raise DataValidationError("Dataset is empty.")

    # The supervised training pipeline requires at least two classes to stratify splits.
    if normalized["risk_level"].nunique() < 2:
        raise DataValidationError("Target column 'risk_level' must contain at least two classes.")

    return normalized
