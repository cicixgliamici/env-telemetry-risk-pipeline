import pandas as pd
from sklearn.model_selection import train_test_split
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_column_names(df).drop_duplicates().copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].fillna(out[col].median())
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        out = out.sort_values("timestamp").reset_index(drop=True)
    return out
def split_features_target(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    X = df.drop(columns=[target_column, "failure_risk", "timestamp"], errors="ignore")
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
