import json

from src.config import OUTPUT_DIR
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
