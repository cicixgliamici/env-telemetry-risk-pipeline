import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_csv
from src.evaluate import evaluate_classifier, summarize_candidate_metrics
from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_dataset, split_features_target
from src.utils import ensure_dir, get_logger, save_json, utc_timestamp, write_text_file
from src.visualize import plot_confusion_matrix, plot_feature_importance


LOGGER = get_logger("src.train")


def run_training(raw_data_path=RAW_DATA_PATH) -> dict:
    """Run the end-to-end supervised training pipeline and save artifacts."""

    for path in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
        ensure_dir(path)

    LOGGER.info("Loading dataset from %s", raw_data_path)
    df = load_csv(raw_data_path)
    cleaned = clean_dataset(df)
    LOGGER.info("Cleaned dataset with %s rows", len(cleaned))
    cleaned.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)

    featured = add_engineered_features(cleaned)
    featured.to_csv(PROCESSED_DIR / "featured_dataset.csv", index=False)
    LOGGER.info("Feature-engineered dataset has %s columns", len(featured.columns))

    X_train, X_test, y_train, y_test = split_features_target(
        featured,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )
    LOGGER.info("Training split prepared with %s train rows and %s test rows", len(X_train), len(X_test))

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    trained_models = {}
    candidate_metrics = {}
    for model_name, model in candidates.items():
        LOGGER.info("Training candidate model: %s", model_name)
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        candidate_metrics[model_name] = evaluate_classifier(model, X_test, y_test)

    # Select the model with the strongest weighted F1 to balance class performance.
    best_model_name = max(
        candidate_metrics,
        key=lambda name: candidate_metrics[name]["f1_weighted"],
    )
    model = trained_models[best_model_name]
    metrics = candidate_metrics[best_model_name]
    metrics["selected_model"] = best_model_name
    metrics["candidate_models"] = summarize_candidate_metrics(candidate_metrics)
    metrics["run_timestamp_utc"] = utc_timestamp()
    save_json(metrics, OUTPUT_DIR / "metrics.json")
    save_json(candidate_metrics, OUTPUT_DIR / "model_comparison.json")
    pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(model.predict(X_test)),
        }
    ).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    model_filename = f"{best_model_name}_model.pkl"
    joblib.dump(model, MODELS_DIR / model_filename)
    plot_feature_importance(model, X_train.columns, FIGURES_DIR / "feature_importance.png")
    plot_confusion_matrix(metrics["confusion_matrix"], FIGURES_DIR / "confusion_matrix.png")
    write_text_file(
        "\n".join(
            [
                "# Training Run Summary",
                "",
                f"- Timestamp (UTC): {metrics['run_timestamp_utc']}",
                f"- Selected model: {best_model_name}",
                f"- Accuracy: {metrics['accuracy']:.4f}",
                f"- Weighted F1: {metrics['f1_weighted']:.4f}",
                "",
                "## Candidate Models",
                "",
            ]
            + [
                (
                    f"- {name}: accuracy={values['accuracy']:.4f}, "
                    f"precision={values['precision_weighted']:.4f}, "
                    f"recall={values['recall_weighted']:.4f}, "
                    f"f1={values['f1_weighted']:.4f}"
                )
                for name, values in metrics["candidate_models"].items()
            ]
        ),
        OUTPUT_DIR / "training_summary.md",
    )
    LOGGER.info("Training completed. Selected model: %s", best_model_name)
    return metrics


if __name__ == "__main__":
    print(run_training())
