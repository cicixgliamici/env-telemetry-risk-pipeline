from pathlib import Path

import pytest

from src.data_loader import load_csv
from src.validation import DataValidationError


def test_load_csv_raises_for_missing_file(tmp_path: Path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(DataValidationError, match="Dataset not found"):
        load_csv(missing_path)


def test_load_csv_reads_and_normalizes_dataset(valid_dataset_df, tmp_path: Path):
    path = tmp_path / "dataset.csv"
    valid_dataset_df.rename(columns={"humidity_pct": "Humidity Pct"}).to_csv(path, index=False)

    out = load_csv(path)

    assert "humidity_pct" in out.columns
    assert len(out) == len(valid_dataset_df)
