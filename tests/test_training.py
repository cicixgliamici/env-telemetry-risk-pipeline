from src.train import run_training


def test_training_produces_metrics():
    metrics = run_training()
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    assert metrics["selected_model"] in {"random_forest", "logistic_regression"}
    assert "candidate_models" in metrics
