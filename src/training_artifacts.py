"""Persist training evidence separately from fitting and model selection."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.artifacts import RunArtifacts
from src.config import RANDOM_STATE, RISK_LABEL, TARGET_COLUMN
from src.experiment import build_data_evidence, format_training_summary
from src.utils import save_json, write_text_file
from src.visualize import plot_confusion_matrix, plot_feature_importance


@dataclass(frozen=True)
class TrainingData:
    """Keep split membership and its audit protocol together."""

    cleaned: pd.DataFrame
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_target: pd.Series
    test_target: pd.Series
    cv_splits: list
    protocol: dict


def _save_featured_dataset(model, dataset: pd.DataFrame, output_path: Path) -> list[str]:
    """Export interpretable features using only the selected model's fitted imputer."""

    raw_features = dataset[list(model.feature_names_in_)]
    # Pipeline slicing also handles raw-only candidates with a passthrough feature step.
    engineered = model[:2].transform(raw_features)
    # Keep labels and timestamps in this audit export, outside the model inputs.
    featured = dataset.copy()
    featured[engineered.columns] = engineered
    featured.to_csv(output_path, index=False)
    return list(engineered.columns)


def _build_metadata(model, data, metrics, config, artifacts, model_feature_columns) -> dict:
    """Describe the fitted pipeline without changing its evaluation results."""

    cleaned = data.cleaned
    X_train, X_test = data.train_features, data.test_features
    best_model_name = metrics["selected_model"]
    cross_validation = metrics["cross_validation"]
    policy = metrics["decision_policy"]
    raw_data_path = config["raw_data_path"]
    candidates = config["candidate_models"]
    model_metadata = {
        **artifacts.identity(),
        "run_timestamp_utc": metrics["run_timestamp_utc"],
        "selected_model": best_model_name,
        "selection_metric": metrics["selection_metric"],
        "selection_dataset": metrics["selection_dataset"],
        "evaluation_dataset": metrics["evaluation_dataset"],
        "split_strategy": metrics["split_strategy"],
        "cv_strategy": metrics["cv_strategy"],
        "temporal_gap": config["temporal_gap"],
        "evaluation_protocol": str(artifacts.outputs / "evaluation_protocol.json"),
        "model_artifact": metrics["model_artifact"],
        "raw_data_path": str(raw_data_path),
        "target_column": TARGET_COLUMN,
        "excluded_columns": [TARGET_COLUMN, "failure_risk", "timestamp"],
        "feature_columns": list(X_train.columns),
        "candidate_models": sorted(candidates.keys()),
        "selected_model_params": model.named_steps["classifier"].get_params(),
        "artifact_format": (
            "telemetry_pipeline_v2_threshold"
            if policy["mode"] == "cost"
            else "telemetry_pipeline_v1"
        ),
        "engineered_feature_columns": model_feature_columns,
        "feature_set": (
            "raw" if model.named_steps["feature_engineering"] == "passthrough" else "engineered"
        ),
        "preprocessing": {
            "steps": list(model.named_steps),
            "imputation_strategy": "median",
            "empty_feature_fill_value": 0,
            "standard_scaling": model.named_steps["scaler"] != "passthrough",
        },
        "random_state": RANDOM_STATE,
        "test_size": config["test_size"],
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cross_validation": cross_validation,
        "risk_label": RISK_LABEL,
        "decision_policy": policy,
        "data_evidence": build_data_evidence(cleaned),
    }
    return model_metadata


def save_training_outputs(
    model,
    data: TrainingData,
    metrics: dict,
    test_predictions,
    config: dict,
    artifacts: RunArtifacts,
) -> None:
    """Write one consistent set of outputs from the already evaluated winner."""

    y_test = data.test_target
    protocol = data.protocol
    cross_validation = metrics["cross_validation"]
    model_filename = Path(metrics["model_artifact"]).name
    model_feature_columns = _save_featured_dataset(
        model, data.cleaned, artifacts.processed / "featured_dataset.csv"
    )
    save_json(metrics, artifacts.outputs / "metrics.json")
    # Candidate comparisons contain training CV scores, never candidate test scores.
    save_json(cross_validation, artifacts.outputs / "model_comparison.json")
    pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(test_predictions),
        }
    ).to_csv(artifacts.outputs / "predictions.csv", index=False)

    joblib.dump(model, artifacts.models / model_filename)
    model_metadata = _build_metadata(model, data, metrics, config, artifacts, model_feature_columns)
    save_json(model_metadata, artifacts.outputs / "model_metadata.json")
    classifier = model.named_steps["classifier"]
    plot_feature_importance(
        classifier, model_feature_columns, artifacts.figures / "feature_importance.png"
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        artifacts.figures / "confusion_matrix.png",
        labels=metrics["confusion_matrix_labels"],
    )
    write_text_file(
        format_training_summary(metrics, model_metadata, protocol),
        artifacts.outputs / "training_summary.md",
    )
