import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize incoming column names to the internal snake_case format."""

    out = df.copy()
    out.columns = [column.strip().lower().replace(" ", "_") for column in out.columns]
    return out


def clean_dataset(df: pd.DataFrame, *, preserve_rows: bool = False) -> pd.DataFrame:
    """Clean values, optionally preserving every input row and its position."""

    out = normalize_column_names(df)
    # Inference needs one output per input row, including repeated observations.
    if not preserve_rows:
        out = out.drop_duplicates().copy()
    for column in out.select_dtypes(include="number").columns:
        out[column] = out[column].fillna(out[column].median())

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        if not preserve_rows:
            out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """Split the dataset into features and target while excluding leakage columns."""

    X = df.drop(columns=[target_column, "failure_risk", "timestamp"], errors="ignore")
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
