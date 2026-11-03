import json
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.train as training
from src.config import OUTPUT_DIR
from src.evaluate import evaluate_predictions
from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_dataset, split_features_target
from src.train import run_training


def test_training_produces_metrics():
    metrics = run_training()
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    assert metrics["selected_model"] in {"random_forest", "logistic_regression"}
    assert "candidate_models" in metrics
    assert "classification_report_dict" in metrics
    assert "cross_validation" in metrics


def test_training_writes_model_metadata():
    run_training()

    metadata_path = OUTPUT_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["selected_model"] in {"random_forest", "logistic_regression"}
    assert metadata["target_column"] == "risk_level"
    assert "failure_risk" in metadata["excluded_columns"]
    assert "timestamp" in metadata["excluded_columns"]
    assert "failure_risk" not in metadata["feature_columns"]
    assert "timestamp" not in metadata["feature_columns"]
    assert metadata["cross_validation"]


@pytest.mark.parametrize("flip_test_labels", [False, True])
def test_model_selection_uses_only_training_cv(
    valid_dataset_df, tmp_path, monkeypatch, flip_test_labels
):
    input_path = tmp_path / "dataset.csv"
    valid_dataset_df.to_csv(input_path, index=False)
    for directory_name in ["PROCESSED_DIR", "OUTPUT_DIR", "MODELS_DIR", "FIGURES_DIR"]:
        monkeypatch.setattr(training, directory_name, tmp_path / directory_name.lower())

    featured = add_engineered_features(clean_dataset(valid_dataset_df))
    X_train, X_test, y_train, y_test = split_features_target(
        featured, training.TARGET_COLUMN, training.TEST_SIZE, training.RANDOM_STATE
    )
    if flip_test_labels:
        # Changing held-out labels must never change which candidate is selected.
        y_test = 1 - y_test
    monkeypatch.setattr(
        training, "split_features_target", lambda *args: (X_train, X_test, y_train, y_test)
    )
    comparison = {
        name: {
            "folds": 2,
            "accuracy_mean": score,
            "accuracy_std": 0.0,
            "f1_weighted_mean": score,
            "f1_weighted_std": 0.0,
        }
        for name, score in [("random_forest", 0.1), ("logistic_regression", 0.9)]
    }
    training_cv = Mock(return_value=comparison)

    prediction_calls = []
    original_predict = LogisticRegression.predict

    def predict_once(model, features):
        pd.testing.assert_frame_equal(features, X_test)
        prediction_calls.append(True)
        return original_predict(model, features)

    evaluate_test = Mock(wraps=evaluate_predictions)
    monkeypatch.setattr(training, "_cross_validate_candidates", training_cv)
    monkeypatch.setattr(LogisticRegression, "predict", predict_once)
    monkeypatch.setattr(training, "evaluate_predictions", evaluate_test)

    metrics = run_training(input_path)

    training_cv.assert_called_once()
    candidates, features, target = training_cv.call_args.args
    pd.testing.assert_frame_equal(features, X_train)
    pd.testing.assert_series_equal(target, y_train)
    assert set(features.index).isdisjoint(X_test.index)
    assert set(candidates) == set(comparison)
    assert len(prediction_calls) == 1
    evaluate_test.assert_called_once()
    pd.testing.assert_series_equal(evaluate_test.call_args.args[0], y_test)
    _assert_selection_artifacts(metrics, comparison, y_test)


def _assert_selection_artifacts(metrics, comparison, y_test):
    """Check that saved comparisons and final test metrics stay distinguishable."""

    assert metrics["selected_model"] == "logistic_regression"
    assert metrics["selection_dataset"] == "training_cross_validation"
    assert metrics["selection_metric"] == "f1_weighted_mean"
    assert metrics["evaluation_dataset"] == "held_out_test"
    assert metrics["candidate_models"] == comparison
    saved_comparison = json.loads(
        (training.OUTPUT_DIR / "model_comparison.json").read_text(encoding="utf-8")
    )
    assert saved_comparison == comparison
    predictions = pd.read_csv(training.OUTPUT_DIR / "predictions.csv")
    assert predictions["y_true"].tolist() == y_test.tolist()
    expected_metrics = evaluate_predictions(y_test, predictions["y_pred"])
    for metric_name, expected_value in expected_metrics.items():
        assert metrics[metric_name] == expected_value
    assert isinstance(
        joblib.load(training.MODELS_DIR / "logistic_regression_model.pkl"), LogisticRegression
    )


def test_cross_validation_rejects_insufficient_training_rows(monkeypatch):
    def unexpected_cross_validation(*args, **kwargs):
        pytest.fail("Cross-validation cannot run with a single observation per class.")

    monkeypatch.setattr(training, "cross_validate", unexpected_cross_validation)
    with pytest.raises(ValueError, match="at least two training rows per class"):
        training._cross_validate_candidates(
            {"logistic_regression": LogisticRegression()},
            pd.DataFrame({"humidity_pct": [50, 90]}),
            pd.Series([0, 1]),
        )
