import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.utils.multiclass import unique_labels

from src.config import RISK_LABEL


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate a fitted classifier on the held-out test split."""

    preds = model.predict(X_test)
    return evaluate_predictions(y_test, preds)


def evaluate_predictions(y_test, preds) -> dict:
    """Compute metrics from saved predictions without running inference again."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="weighted",
        zero_division=0,
    )
    labels = unique_labels(y_test, preds)
    risk_support = int(np.count_nonzero(np.asarray(y_test) == RISK_LABEL))
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "risk_label": RISK_LABEL,
        "risk_support": risk_support,
        # No observed risk cases means recall is unavailable, rather than a measured zero.
        "recall_risk": (
            float(
                recall_score(y_test, preds, labels=[RISK_LABEL], average="macro", zero_division=0)
            )
            if risk_support
            else None
        ),
        "confusion_matrix_labels": labels.tolist(),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=labels).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
        "classification_report_dict": classification_report(
            y_test,
            preds,
            zero_division=0,
            output_dict=True,
        ),
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
