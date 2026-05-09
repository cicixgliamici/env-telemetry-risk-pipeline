import pandas as pd
import pytest
from pandas.api.types import is_string_dtype

from scripts.generate_synthetic_data import generate_synthetic_dataset, write_synthetic_dataset
from src.config import REQUIRED_COLUMNS
from src.validation import validate_dataset_schema


def test_generate_synthetic_dataset_matches_schema_and_validates():
    df = generate_synthetic_dataset(rows=120, seed=7)

    assert set(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 120
    assert df["risk_level"].nunique() == 2
    assert df["failure_risk"].nunique() == 2

    validated = validate_dataset_schema(df)

    assert len(validated) == 120


def test_generate_synthetic_dataset_is_reproducible():
    left = generate_synthetic_dataset(rows=50, seed=99)
    right = generate_synthetic_dataset(rows=50, seed=99)

    pd.testing.assert_frame_equal(left, right)


def test_generate_synthetic_dataset_rejects_tiny_outputs():
    with pytest.raises(ValueError, match="at least 20"):
        generate_synthetic_dataset(rows=5)


def test_write_synthetic_dataset_creates_csv(tmp_path):
    output_path = tmp_path / "synthetic.csv"

    written_path = write_synthetic_dataset(output_path, rows=30, seed=1)

    assert written_path == output_path
    assert output_path.exists()
    assert len(pd.read_csv(output_path)) == 30


def test_synthetic_dataset_has_no_formula_like_text_fields():
    df = generate_synthetic_dataset(rows=40, seed=2)
    text_columns = [column for column in df.columns if is_string_dtype(df[column])]

    for column in text_columns:
        assert not df[column].str.startswith(("=", "+", "-", "@")).any()
