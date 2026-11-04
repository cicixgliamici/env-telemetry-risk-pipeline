"""Verify binary threshold semantics and empirical cost selection independently of holdout data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

from src.decision import (
    binary_error_cost,
    select_cost_threshold,
    threshold_predictions,
    validate_decision_settings,
)
from src.modeling import build_classifier_pipeline


class FixedProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Expose known scores so the cost objective can be checked without learning randomness."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        risk = X["risk_score"].to_numpy()
        return np.column_stack([risk if label == 1 else 1 - risk for label in self.classes_])


@pytest.fixture
def cost_problem():
    features = pd.DataFrame(
        {"risk_score": [0.05, 0.35, 0.4, 0.45, 0.65, 0.8, 0.15, 0.25, 0.55, 0.65, 0.85, 0.95]}
    )
    target = pd.Series([0, 1] * 6)
    splits = list(StratifiedKFold(n_splits=3).split(features, target))
    model = build_classifier_pipeline(FixedProbabilityClassifier(), engineer_features=False)
    return model, features, target, splits


@pytest.mark.parametrize("classes", [[0, 1], [1, 2]])
def test_cutoff_uses_risk_class_order_and_inclusive_boundary(classes):
    risk = np.array([0, 0.5, 1])
    probabilities = np.column_stack([risk if label == 1 else 1 - risk for label in classes])
    other = next(label for label in classes if label != 1)
    np.testing.assert_array_equal(threshold_predictions(probabilities, classes, 0.5), [other, 1, 1])
    np.testing.assert_array_equal(threshold_predictions(probabilities, classes, 0), [1, 1, 1])
    np.testing.assert_array_equal(threshold_predictions(probabilities, classes, None), [other] * 3)


def test_selected_rule_minimizes_validation_cost_and_tracks_each_row(cost_problem):
    model, features, target, splits = cost_problem
    policy, curve, predictions = select_cost_threshold(model, features, target, splits, 5, 1)

    assert curve["selected"].sum() == 1
    assert curve["threshold"].dropna().tolist() == [number / 20 for number in range(21)]
    chosen = curve[curve["selected"]].iloc[0]
    assert chosen["total_cost"] == curve["total_cost"].min()
    assert policy["selected_validation_cost"] == chosen["cost_per_observation"]
    assert set(predictions["row_index"]) == set(features.index)
    assert policy["validation_coverage"] == 1
    assert len(predictions) == len(features)
    assert not hasattr(model.named_steps["classifier"], "classes_")
    recomputed = binary_error_cost(
        target.loc[predictions["row_index"]], predictions["selected_prediction"], 5, 1, 0
    )
    assert recomputed["cost_per_observation"] == policy["selected_validation_cost"]


def test_larger_false_negative_cost_favors_fewer_missed_risks(cost_problem):
    model, features, target, splits = cost_problem
    cheap, cheap_curve, _ = select_cost_threshold(model, features, target, splits, 1, 1)
    costly, costly_curve, _ = select_cost_threshold(model, features, target, splits, 20, 1)
    assert costly["threshold"] <= cheap["threshold"]
    assert (
        costly_curve[costly_curve["selected"]].iloc[0]["false_negatives"]
        <= (cheap_curve[cheap_curve["selected"]].iloc[0]["false_negatives"])
    )


def test_never_risk_policy_handles_probability_one_without_invalid_threshold(cost_problem):
    model, features, target, splits = cost_problem
    features["risk_score"] = 1
    policy, _, _ = select_cost_threshold(model, features, target, splits, 0, 1)
    assert policy["threshold"] is None
    assert policy["decision_rule"] == "never_risk"
    model.risk_decision_policy_ = policy
    model.fit(features, target)
    assert model.predict(features).tolist() == [0] * len(features)
    assert model.predict_proba(features)[:, 1].tolist() == [1] * len(features)
    weights = np.arange(len(target)) + 1
    assert model.score(features, target, sample_weight=weights) == np.average(
        target == 0, weights=weights
    )


@pytest.mark.parametrize("target", [[0, 2] * 6, [0, 1, 2] * 4])
def test_cost_policy_rejects_unavailable_binary_risk_task(cost_problem, target):
    model, features, _, splits = cost_problem
    with pytest.raises(ValueError, match="exactly two classes"):
        select_cost_threshold(model, features, pd.Series(target), splits, 5, 1)


def test_cost_policy_requires_both_validation_classes(cost_problem):
    model, features, target, _ = cost_problem
    target[:] = 0
    target.iloc[0] = 1
    splits = [(np.array([0, 1]), np.array([2, 3])), (np.arange(4), np.array([4, 5]))]
    with pytest.raises(ValueError, match="both classes in validation"):
        select_cost_threshold(model, features, target, splits, 5, 1)


@pytest.mark.parametrize("fn_cost,fp_cost", [(0, 0), (-1, 1), (1, float("nan")), (float("inf"), 1)])
def test_error_cost_settings_reject_invalid_weights(fn_cost, fp_cost):
    with pytest.raises(ValueError, match="Error costs"):
        validate_decision_settings("cost", fn_cost, fp_cost)


def test_binary_cost_reports_precision_recall_and_counts():
    metrics = binary_error_cost([0, 0, 1, 1], [0, 1, 0, 1], 5, 2, 0)
    assert metrics["false_negatives"] == metrics["false_positives"] == 1
    assert metrics["true_positives"] == metrics["true_negatives"] == 1
    assert metrics["precision_risk"] == metrics["recall_risk"] == 0.5
    assert metrics["cost_per_observation"] == 7 / 4


def test_cost_metrics_do_not_fabricate_precision_or_recall_without_support():
    metrics = binary_error_cost([0, 0], [0, 0], 5, 1, 0)
    assert metrics["precision_risk"] is None
    assert metrics["recall_risk"] is None
