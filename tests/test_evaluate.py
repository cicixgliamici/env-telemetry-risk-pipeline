import json

import pytest

from src.evaluate import evaluate_predictions


def test_macro_f1_and_risk_recall_expose_majority_only_predictions():
    metrics = evaluate_predictions([0] * 9 + [1], [0] * 10)

    assert metrics["accuracy"] == 0.9
    assert metrics["f1_weighted"] == pytest.approx(81 / 95)
    assert metrics["f1_macro"] == pytest.approx(9 / 19)
    assert metrics["recall_risk"] == 0
    assert metrics["risk_support"] == 1
    assert metrics["confusion_matrix_labels"] == [0, 1]


def test_risk_recall_is_unavailable_without_observed_risk_cases():
    metrics = evaluate_predictions([0, 2], [0, 2])

    assert metrics["recall_risk"] is None
    assert metrics["risk_support"] == 0
    assert metrics["f1_macro"] == 1
    assert metrics["confusion_matrix_labels"] == [0, 2]
    json.dumps(metrics, allow_nan=False)


def test_risk_recall_uses_class_one_in_multiclass_predictions():
    metrics = evaluate_predictions([0, 1, 1, 2], [0, 1, 2, 2])

    assert metrics["recall_risk"] == 0.5
    assert metrics["risk_support"] == 2
    assert metrics["confusion_matrix_labels"] == [0, 1, 2]
