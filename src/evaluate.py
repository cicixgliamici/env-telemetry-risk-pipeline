from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate a fitted classifier on the held-out test split."""

    preds = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }


def summarize_candidate_metrics(candidate_metrics: dict[str, dict]) -> dict[str, dict]:
    """Extract the comparable headline metrics for each candidate model."""

    summary = {}
    for model_name, metrics in candidate_metrics.items():
        summary[model_name] = {
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        }
    return summary
