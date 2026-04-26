from pathlib import Path

import pandas as pd

from src.validation import validate_dataset_schema, validate_file_exists


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a raw CSV dataset after validating the file and schema."""

    validated_path = validate_file_exists(path)
    df = pd.read_csv(validated_path)
    return validate_dataset_schema(df)
