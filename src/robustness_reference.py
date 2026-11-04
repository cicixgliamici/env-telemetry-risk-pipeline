"""Recover an existing training run's held-out data and frozen classifier."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN, TELEMETRY_COLUMNS
from src.experiment import build_data_evidence
from src.validation import validate_dataset_schema


@dataclass(frozen=True)
class EvaluationReference:
    """Keep frozen predictions, row membership, and source identity together."""

    model: Pipeline
    training: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    baseline_predictions: np.ndarray
    identity: dict


def file_sha256(path: Path) -> str:
    """Fingerprint the exact artifact bytes used by this experiment."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_rows(dataset: pd.DataFrame, record: dict) -> pd.DataFrame:
    """Reject stale or ambiguous protocol membership before scoring any scenario."""

    indices = record["row_indices"]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(dataset)
            for index in indices
        )
    ):
        raise ValueError("Training run contains invalid evaluation row membership.")
    if len(indices) != record["rows"]:
        raise ValueError("Training run partition row count does not match its protocol.")
    return dataset.iloc[indices].copy()


def _load_model(path: Path, metadata: dict) -> Pipeline:
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or "feature_engineering" not in model.named_steps:
        raise ValueError(
            "Robustness experiments require a saved training-fitted telemetry pipeline."
        )
    if list(model.feature_names_in_) != metadata["feature_columns"]:
        raise ValueError("Model feature schema does not match the training run metadata.")
    if set(model.feature_names_in_) != TELEMETRY_COLUMNS:
        raise ValueError("Robustness experiments support the eight-column telemetry schema.")
    return model


def _check_baseline(test, predictions, source_dir):
    """Verify the recovered rows and model reproduce the original holdout predictions."""

    saved = pd.read_csv(source_dir / "outputs" / "predictions.csv")
    if not np.array_equal(saved["y_true"], test[TARGET_COLUMN]) or not np.array_equal(
        saved["y_pred"], predictions
    ):
        raise ValueError(
            "Source model or test rows do not reproduce the saved holdout predictions."
        )


def load_evaluation_reference(training_run: str | Path) -> EvaluationReference:
    """Load trusted local artifacts; never retrain, select a candidate, or recompute a split."""

    source_dir = Path(training_run).resolve()
    manifest = _read_json(source_dir / "run.json")
    if manifest["operation"] != "train" or manifest["status"] != "completed":
        raise ValueError("Robustness experiments require a completed supervised training run.")
    metadata = _read_json(source_dir / "outputs" / "model_metadata.json")
    if metadata["run_id"] != manifest["run_id"]:
        raise ValueError("Training run identity does not match its model metadata.")
    protocol_path = source_dir / "outputs" / "evaluation_protocol.json"
    protocol = _read_json(protocol_path)
    dataset_path = source_dir / "processed" / "cleaned_dataset.csv"
    # Round-trip parsing preserves the floats used in the normalized training fingerprint.
    dataset = validate_dataset_schema(pd.read_csv(dataset_path, float_precision="round_trip"))
    evidence = build_data_evidence(dataset)
    if evidence["validated_csv_sha256"] != metadata["data_evidence"]["validated_csv_sha256"]:
        raise ValueError(
            "Cleaned data does not match the training fingerprint; check data/environment."
        )
    train = _partition_rows(dataset, protocol["training"])
    test = _partition_rows(dataset, protocol["test"])
    if not train.index.intersection(test.index).empty:
        raise ValueError("Training and test membership must be disjoint.")
    # Resolve inside the chosen run so copied run folders remain usable.
    model_files = list((source_dir / "models").glob("*_model.pkl"))
    if len(model_files) != 1:
        raise ValueError("Expected exactly one selected classifier in the source run.")
    model = _load_model(model_files[0], metadata)
    columns = list(model.feature_names_in_)
    predictions = model.predict(test[columns])
    _check_baseline(test, predictions, source_dir)
    identity = {
        "source_run_id": manifest["run_id"],
        "source_run_dir": str(source_dir),
        "selected_model": metadata["selected_model"],
        "split_strategy": protocol["split_strategy"],
        "source_model_sha256": file_sha256(model_files[0]),
        "source_data_sha256": file_sha256(dataset_path),
        "source_protocol_sha256": file_sha256(protocol_path),
        "validated_csv_sha256": evidence["validated_csv_sha256"],
    }
    return EvaluationReference(model, train, test, columns, predictions, identity)
