"""Select a binary risk decision rule using training-only validation probabilities."""

from math import isfinite

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.config import RISK_LABEL


class RiskDecisionPipeline(Pipeline):
    """Persist an optional learned risk cutoff alongside the fitted preprocessing pipeline."""

    def predict(self, X, **params):
        policy = getattr(self, "risk_decision_policy_", None)
        if policy is None:
            return super().predict(X, **params)
        # Probabilities keep their original meaning; only the label decision changes.
        probabilities = self.predict_proba(X, **params)
        return threshold_predictions(
            probabilities, self.classes_, policy["threshold"], policy["risk_label"]
        )

    def score(self, X, y=None, sample_weight=None, **params):
        if getattr(self, "risk_decision_policy_", None) is None:
            return super().score(X, y, sample_weight=sample_weight, **params)
        # Pipeline.score otherwise delegates to the classifier and bypasses the selected cutoff.
        return accuracy_score(y, self.predict(X, **params), sample_weight=sample_weight)


def validate_decision_settings(mode: str, false_negative_cost: float, false_positive_cost: float):
    if mode not in {"classifier", "cost"}:
        raise ValueError(f"Unknown decision policy: {mode}")
    costs = [false_negative_cost, false_positive_cost]
    if any(not isfinite(cost) or cost < 0 for cost in costs) or not any(costs):
        raise ValueError("Error costs must be finite and non-negative, with at least one positive.")


def _binary_classes(classes, risk_label=RISK_LABEL):
    labels = list(classes)
    if len(labels) != 2 or risk_label not in labels:
        raise ValueError(
            "Cost-based decisions require exactly two classes including the risk label."
        )
    risk_position = labels.index(risk_label)
    return risk_position, labels[1 - risk_position]


def threshold_predictions(probabilities, classes, threshold, risk_label=RISK_LABEL):
    """Use inclusive cutoffs; None explicitly means never predict the risk class."""

    risk_position, other_label = _binary_classes(classes, risk_label)
    if threshold is None:
        return np.full(len(probabilities), other_label)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Risk threshold must be between 0 and 1, or None for never-risk.")
    return np.where(
        np.asarray(probabilities)[:, risk_position] >= threshold, risk_label, other_label
    )


def binary_error_cost(y_true, y_pred, false_negative_cost, false_positive_cost, other_label):
    """Report error counts and relative cost per observation without assuming calibrated scores."""

    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    if not len(actual) or len(actual) != len(predicted):
        raise ValueError("Cost evaluation requires non-empty, aligned labels and predictions.")
    if not set(np.unique(actual)).union(np.unique(predicted)) <= {RISK_LABEL, other_label}:
        raise ValueError("Cost evaluation requires the same two classes as the training policy.")
    positives, flagged = actual == RISK_LABEL, predicted == RISK_LABEL
    tp = int(np.count_nonzero(positives & flagged))
    fn = int(np.count_nonzero(positives & ~flagged))
    fp = int(np.count_nonzero(~positives & flagged))
    tn = int(np.count_nonzero(~positives & ~flagged))
    total = fn * false_negative_cost + fp * false_positive_cost
    if not isfinite(total):
        raise ValueError("Error costs are too large for finite evaluation.")
    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision_risk": tp / (tp + fp) if tp + fp else None,
        "recall_risk": tp / (tp + fn) if tp + fn else None,
        "total_cost": float(total),
        "cost_per_observation": float(total / len(actual)),
    }


def _validation_predictions(model, features, target, cv_splits) -> pd.DataFrame:
    """Clone and fit every preprocessing step on the historical training part of each fold."""

    frames = []
    for number, (train, validation) in enumerate(cv_splits, start=1):
        fitted = clone(model).fit(features.iloc[train], target.iloc[train])
        probabilities = fitted.predict_proba(features.iloc[validation])
        position, _ = _binary_classes(fitted.classes_)
        frames.append(
            pd.DataFrame(
                {
                    "row_index": features.index[validation],
                    "fold": number,
                    "y_true": target.iloc[validation].to_numpy(),
                    "risk_probability": probabilities[:, position],
                    "classifier_prediction": fitted.classes_[probabilities.argmax(axis=1)],
                }
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    if predictions["row_index"].duplicated().any():
        raise ValueError("Threshold selection requires at most one validation prediction per row.")
    return predictions


def _threshold_curve(predictions, classes, false_negative_cost, false_positive_cost):
    """Use a prespecified 0.05 grid plus the otherwise unreachable never-risk decision."""

    position, other_label = _binary_classes(classes)
    probabilities = np.zeros((len(predictions), 2))
    probabilities[:, position] = predictions["risk_probability"]
    probabilities[:, 1 - position] = 1 - predictions["risk_probability"]
    rows = []
    for threshold in [*[round(value, 2) for value in np.linspace(0, 1, 21).tolist()], None]:
        labels = threshold_predictions(probabilities, classes, threshold)
        rows.append(
            {
                "threshold": threshold,
                "rule": "never_risk" if threshold is None else "risk_probability_greater_or_equal",
                **binary_error_cost(
                    predictions["y_true"],
                    labels,
                    false_negative_cost,
                    false_positive_cost,
                    other_label,
                ),
            }
        )
    # Resolve equal costs by proximity to 0.5, then the lower threshold; never-risk is last on ties.
    selected = min(
        rows,
        key=lambda row: (
            row["total_cost"],
            round(abs(row["threshold"] - 0.5), 12) if row["threshold"] is not None else 2,
            row["threshold"] if row["threshold"] is not None else 2,
        ),
    )
    return selected, rows, probabilities


def select_cost_threshold(
    model, features, target, cv_splits, false_negative_cost, false_positive_cost
):
    """Tune only the already selected classifier's decision rule; the holdout is never an input."""

    validate_decision_settings("cost", false_negative_cost, false_positive_cost)
    classes = np.unique(target)
    _, other_label = _binary_classes(classes)
    predictions = _validation_predictions(model, features, target, cv_splits)
    if predictions["y_true"].nunique() != 2:
        raise ValueError(
            "Cost-based cutoff selection requires both classes in validation observations."
        )
    selected, rows, probabilities = _threshold_curve(
        predictions, classes, false_negative_cost, false_positive_cost
    )
    policy = {
        "mode": "cost",
        "risk_label": RISK_LABEL,
        "other_label": int(other_label),
        "threshold": selected["threshold"],
        "decision_rule": selected["rule"],
        "selection_dataset": "training_cross_validation_predictions",
        "selection_metric": "minimum_pooled_validation_error_cost",
        "false_negative_cost": false_negative_cost,
        "false_positive_cost": false_positive_cost,
        "validation_rows": len(predictions),
        "training_rows": len(features),
        "validation_coverage": len(predictions) / len(features),
        "selected_validation_cost": selected["cost_per_observation"],
        "classifier_validation_cost": binary_error_cost(
            predictions["y_true"],
            predictions["classifier_prediction"],
            false_negative_cost,
            false_positive_cost,
            other_label,
        )["cost_per_observation"],
        "threshold_grid_step": 0.05,
    }
    predictions["selected_prediction"] = threshold_predictions(
        probabilities, classes, policy["threshold"]
    )
    for row in rows:
        row["selected"] = row is selected
    return policy, pd.DataFrame(rows), predictions
