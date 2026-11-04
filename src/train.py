from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate

from src.artifacts import RunArtifacts, artifact_run
from src.config import (
    RANDOM_STATE,
    RAW_DATA_PATH,
    RISK_LABEL,
    RUNS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_csv
from src.evaluate import evaluate_predictions
from src.experiment import build_data_evidence, build_feature_ablation, format_training_summary
from src.modeling import build_model_candidates
from src.preprocessing import clean_dataset, split_features_target
from src.splitting import (
    build_evaluation_protocol,
    build_training_folds,
    validate_split_settings,
)
from src.utils import get_logger, save_json, utc_timestamp, write_text_file
from src.visualize import plot_confusion_matrix, plot_feature_importance

LOGGER = get_logger("src.train")


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


def _risk_recall(y_true, y_pred) -> float:
    """Missing risk observations make fold recall undefined, rather than measured zero."""

    if not np.any(np.asarray(y_true) == RISK_LABEL):
        return np.nan
    return recall_score(y_true, y_pred, labels=[RISK_LABEL], average="macro", zero_division=0)


def _cross_validate_candidates(
    candidates: dict, X: pd.DataFrame, y: pd.Series, *, cv_splits=None
) -> dict:
    """Evaluate identical precomputed folds, keeping the holdout outside model selection."""

    cv = build_training_folds(y) if cv_splits is None else cv_splits
    scoring = {
        "accuracy": "accuracy",
        # Small validation folds can leave a class without predicted observations.
        "precision_weighted": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    if RISK_LABEL in y.unique():
        scoring["recall_risk"] = make_scorer(_risk_recall)
    results = {}
    for model_name, model in candidates.items():
        LOGGER.info("Cross-validating candidate model: %s", model_name)
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",
        )
        results[model_name] = {"folds": len(cv)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            observed = values[np.isfinite(values)]
            results[model_name][f"{metric}_mean"] = (
                float(observed.mean()) if len(observed) else None
            )
            results[model_name][f"{metric}_std"] = float(observed.std()) if len(observed) else None
            # Persist support for aggregates so unavailable fold recall is visible to reviewers.
            results[model_name][f"{metric}_observed_folds"] = int(len(observed))
        # A different target vocabulary may have no class 1; its recall is undefined.
        results[model_name].setdefault("recall_risk_mean", None)
        results[model_name].setdefault("recall_risk_std", None)
        results[model_name].setdefault("recall_risk_observed_folds", 0)
    return results


def run_training(
    raw_data_path=RAW_DATA_PATH,
    *,
    output_dir: str | Path = RUNS_DIR,
    split_strategy: str = "stratified_random",
    test_size: float = TEST_SIZE,
    temporal_gap: int = 0,
) -> dict:
    """Run the end-to-end supervised training pipeline and save artifacts."""

    validate_split_settings(split_strategy, test_size, temporal_gap)
    candidates = build_model_candidates()
    config = {
        "raw_data_path": str(Path(raw_data_path).resolve()),
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "target_column": TARGET_COLUMN,
        "risk_label": RISK_LABEL,
        "selection_metric": "f1_weighted_mean",
        "split_strategy": split_strategy,
        "temporal_gap": temporal_gap,
        "max_cv_folds": 3,
        "candidate_models": {
            name: {
                "classifier": type(model.named_steps["classifier"]).__name__,
                "classifier_params": model.named_steps["classifier"].get_params(),
                "engineer_features": model.named_steps["feature_engineering"] != "passthrough",
                "scale_features": model.named_steps["scaler"] != "passthrough",
            }
            for name, model in candidates.items()
        },
    }
    with artifact_run(output_dir, "train", config) as artifacts:
        return _run_training(raw_data_path, artifacts, candidates, config)


def _run_training(raw_data_path, artifacts: RunArtifacts, candidates: dict, config: dict) -> dict:
    """Fit and evaluate within the directories assigned to this execution."""

    LOGGER.info("Loading dataset from %s", raw_data_path)
    df = load_csv(raw_data_path, purpose="training")
    # Split before any statistics are learned; CV clones and fits the entire pipeline.
    cleaned = clean_dataset(df, impute_numeric=False)
    LOGGER.info("Cleaned dataset with %s rows", len(cleaned))
    cleaned.to_csv(artifacts.processed / "cleaned_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = split_features_target(
        cleaned,
        TARGET_COLUMN,
        config["test_size"],
        RANDOM_STATE,
        split_strategy=config["split_strategy"],
        temporal_gap=config["temporal_gap"],
    )
    LOGGER.info(
        "Training split prepared with %s train rows and %s test rows",
        len(X_train),
        len(X_test),
    )

    cv_splits = build_training_folds(
        y_train,
        strategy=config["split_strategy"],
        timestamps=cleaned.loc[X_train.index, "timestamp"],
        temporal_gap=config["temporal_gap"],
    )
    protocol = build_evaluation_protocol(
        cleaned,
        X_train,
        X_test,
        cv_splits,
        config["split_strategy"],
        config["test_size"],
        config["temporal_gap"],
    )
    save_json(protocol, artifacts.outputs / "evaluation_protocol.json")
    # Keep test observations outside every fold and every model selection decision.
    cross_validation = _cross_validate_candidates(candidates, X_train, y_train, cv_splits=cv_splits)
    best_model_name = max(
        cross_validation,
        key=lambda name: cross_validation[name]["f1_weighted_mean"],
    )
    model = candidates[best_model_name]
    LOGGER.info("Fitting selected model on the full training split: %s", best_model_name)
    model.fit(X_train, y_train)
    model_feature_columns = _save_featured_dataset(
        model, cleaned, artifacts.processed / "featured_dataset.csv"
    )
    # Reuse these final predictions for metrics and CSV output to evaluate only once.
    test_predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, test_predictions)
    metrics["selected_model"] = best_model_name
    metrics["selection_metric"] = "f1_weighted_mean"
    metrics["selection_dataset"] = "training_cross_validation"
    metrics["evaluation_dataset"] = "held_out_test"
    metrics["split_strategy"] = config["split_strategy"]
    metrics["cv_strategy"] = protocol["cv_strategy"]
    metrics["temporal_gap"] = config["temporal_gap"]
    metrics["candidate_models"] = cross_validation
    metrics["cross_validation"] = cross_validation
    metrics["feature_ablation"] = build_feature_ablation(cross_validation)
    dummy_score = cross_validation.get("dummy_most_frequent", {}).get("f1_weighted_mean")
    metrics["dummy_f1_weighted_gap"] = (
        None
        if dummy_score is None
        else cross_validation[best_model_name]["f1_weighted_mean"] - dummy_score
    )
    metrics["run_timestamp_utc"] = utc_timestamp()
    metrics.update(artifacts.identity())
    model_filename = f"{best_model_name}_model.pkl"
    metrics["model_artifact"] = str(artifacts.models / model_filename)
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
        "artifact_format": "telemetry_pipeline_v1",
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
        "data_evidence": build_data_evidence(cleaned),
    }
    save_json(model_metadata, artifacts.outputs / "model_metadata.json")
    classifier = model.named_steps["classifier"]
    plot_feature_importance(
        classifier, model_feature_columns, artifacts.figures / "feature_importance.png"
    )
    plot_confusion_matrix(metrics["confusion_matrix"], artifacts.figures / "confusion_matrix.png")
    write_text_file(
        format_training_summary(metrics, model_metadata, protocol),
        artifacts.outputs / "training_summary.md",
    )
    LOGGER.info("Training completed. Selected model: %s", best_model_name)
    return metrics


if __name__ == "__main__":
    print(run_training())
