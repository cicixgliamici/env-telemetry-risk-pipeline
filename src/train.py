import joblib
import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RISK_LABEL,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_csv
from src.evaluate import evaluate_predictions
from src.experiment import build_data_evidence, build_feature_ablation, format_training_summary
from src.modeling import build_model_candidates
from src.preprocessing import clean_dataset, split_features_target
from src.utils import ensure_dir, get_logger, save_json, utc_timestamp, write_text_file
from src.visualize import plot_confusion_matrix, plot_feature_importance

LOGGER = get_logger("src.train")


def _save_featured_dataset(model, dataset: pd.DataFrame) -> list[str]:
    """Export interpretable features using only the selected model's fitted imputer."""

    raw_features = dataset[list(model.feature_names_in_)]
    # Pipeline slicing also handles raw-only candidates with a passthrough feature step.
    engineered = model[:2].transform(raw_features)
    # Keep labels and timestamps in this audit export, outside the model inputs.
    featured = dataset.copy()
    featured[engineered.columns] = engineered
    featured.to_csv(PROCESSED_DIR / "featured_dataset.csv", index=False)
    return list(engineered.columns)


def _cross_validate_candidates(candidates: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run compact stratified cross-validation when each class has enough rows."""

    min_class_count = int(y.value_counts().min())
    folds = min(3, min_class_count)
    if folds < 2:
        # Never fall back to choosing a model on the held-out test set.
        raise ValueError(
            "Model selection requires at least two training rows per class "
            "for stratified cross-validation. Provide more labeled observations."
        )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        # Small validation folds can leave a class without predicted observations.
        "precision_weighted": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    if RISK_LABEL in y.unique():
        scoring["recall_risk"] = make_scorer(
            recall_score, labels=[RISK_LABEL], average="macro", zero_division=0
        )
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
        results[model_name] = {"folds": folds}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            results[model_name][f"{metric}_mean"] = float(values.mean())
            results[model_name][f"{metric}_std"] = float(values.std())
        # A different target vocabulary may have no class 1; its recall is undefined.
        results[model_name].setdefault("recall_risk_mean", None)
        results[model_name].setdefault("recall_risk_std", None)
    return results


def run_training(raw_data_path=RAW_DATA_PATH) -> dict:
    """Run the end-to-end supervised training pipeline and save artifacts."""

    for path in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
        ensure_dir(path)

    LOGGER.info("Loading dataset from %s", raw_data_path)
    df = load_csv(raw_data_path, purpose="training")
    # Split before any statistics are learned; CV clones and fits the entire pipeline.
    cleaned = clean_dataset(df, impute_numeric=False)
    LOGGER.info("Cleaned dataset with %s rows", len(cleaned))
    cleaned.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = split_features_target(
        cleaned,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )
    LOGGER.info(
        "Training split prepared with %s train rows and %s test rows",
        len(X_train),
        len(X_test),
    )

    candidates = build_model_candidates()
    # Keep test observations outside every fold and every model selection decision.
    cross_validation = _cross_validate_candidates(candidates, X_train, y_train)
    best_model_name = max(
        cross_validation,
        key=lambda name: cross_validation[name]["f1_weighted_mean"],
    )
    model = candidates[best_model_name]
    LOGGER.info("Fitting selected model on the full training split: %s", best_model_name)
    model.fit(X_train, y_train)
    model_feature_columns = _save_featured_dataset(model, cleaned)
    # Reuse these final predictions for metrics and CSV output to evaluate only once.
    test_predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, test_predictions)
    metrics["selected_model"] = best_model_name
    metrics["selection_metric"] = "f1_weighted_mean"
    metrics["selection_dataset"] = "training_cross_validation"
    metrics["evaluation_dataset"] = "held_out_test"
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
    save_json(metrics, OUTPUT_DIR / "metrics.json")
    # Candidate comparisons contain training CV scores, never candidate test scores.
    save_json(cross_validation, OUTPUT_DIR / "model_comparison.json")
    pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(test_predictions),
        }
    ).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    model_filename = f"{best_model_name}_model.pkl"
    joblib.dump(model, MODELS_DIR / model_filename)
    model_metadata = {
        "run_timestamp_utc": metrics["run_timestamp_utc"],
        "selected_model": best_model_name,
        "selection_metric": metrics["selection_metric"],
        "selection_dataset": metrics["selection_dataset"],
        "evaluation_dataset": metrics["evaluation_dataset"],
        "model_artifact": str(MODELS_DIR / model_filename),
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
        "test_size": TEST_SIZE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cross_validation": cross_validation,
        "risk_label": RISK_LABEL,
        "data_evidence": build_data_evidence(cleaned),
    }
    save_json(model_metadata, OUTPUT_DIR / "model_metadata.json")
    classifier = model.named_steps["classifier"]
    plot_feature_importance(
        classifier, model_feature_columns, FIGURES_DIR / "feature_importance.png"
    )
    plot_confusion_matrix(metrics["confusion_matrix"], FIGURES_DIR / "confusion_matrix.png")
    write_text_file(
        format_training_summary(metrics, model_metadata),
        OUTPUT_DIR / "training_summary.md",
    )
    LOGGER.info("Training completed. Selected model: %s", best_model_name)
    return metrics


if __name__ == "__main__":
    print(run_training())
