import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import OUTPUT_DIR, RAW_DATA_PATH
from src.data_loader import load_csv
from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_dataset


def _load_feature_columns(metadata_path: str | Path | None) -> list[str] | None:
    if metadata_path is None:
        return None

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return list(metadata["feature_columns"])


def prepare_prediction_features(
    raw_data_path: str | Path = RAW_DATA_PATH,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare raw pipeline inputs without learning statistics or changing rows."""

    return _prepare_prediction_features(
        load_csv(raw_data_path, purpose="prediction"), feature_columns
    )


def _prepare_prediction_features(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None,
    *,
    legacy_preprocessing: bool = False,
) -> pd.DataFrame:
    # Keep row positions stable so labels and probabilities match the source CSV.
    featured = clean_dataset(dataset, preserve_rows=True, impute_numeric=legacy_preprocessing)
    # Bare classifiers from earlier versions still expect externally engineered inputs.
    if legacy_preprocessing:
        featured = add_engineered_features(featured)
    if feature_columns is None:
        return featured.drop(columns=["risk_level", "failure_risk", "timestamp"], errors="ignore")
    missing_columns = sorted(set(feature_columns) - set(featured.columns))
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing model feature columns: {', '.join(missing_columns)}"
        )
    return featured[feature_columns]


def run_batch_prediction(
    model_path: str | Path,
    raw_data_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = OUTPUT_DIR / "batch_predictions.csv",
    metadata_path: str | Path | None = OUTPUT_DIR / "model_metadata.json",
) -> pd.DataFrame:
    """Run batch predictions from a serialized model and save a CSV output."""

    model = joblib.load(model_path)
    is_pipeline = isinstance(model, Pipeline) and "feature_engineering" in model.named_steps
    # The fitted pipeline owns its schema; scoring does not require a metadata sidecar.
    feature_columns = (
        list(model.feature_names_in_) if is_pipeline else _load_feature_columns(metadata_path)
    )
    # Read once so features and output rows always come from the same input snapshot.
    result = load_csv(raw_data_path, purpose="prediction").copy()
    X = _prepare_prediction_features(result, feature_columns, legacy_preprocessing=not is_pipeline)
    predictions = model.predict(X)

    result["predicted_risk_level"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] > 1:
            result["predicted_risk_probability"] = probabilities[:, 1]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
