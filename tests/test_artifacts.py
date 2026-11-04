"""Check run isolation, reproducibility records, and failure reporting."""

import json
from importlib.metadata import version
from pathlib import Path

import pytest

from src.artifacts import artifact_run
from src.cli import main
from src.modeling import build_model_candidates
from src.train import run_training


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_runs_preserve_previous_artifacts_and_record_environment(tmp_path):
    with artifact_run(tmp_path, "train", {"random_state": 42}) as first:
        marker = first.outputs / "metrics.json"
        marker.write_text('{"accuracy": 0.5}', encoding="utf-8")
        assert _read_json(first.run_dir / "run.json")["status"] == "running"

    with artifact_run(tmp_path, "train", {"random_state": 7}) as second:
        (second.outputs / "metrics.json").write_text('{"accuracy": 0.9}', encoding="utf-8")

    assert first.run_id != second.run_id
    assert _read_json(marker) == {"accuracy": 0.5}
    for run, seed in [(first, 42), (second, 7)]:
        manifest = _read_json(run.run_dir / "run.json")
        assert manifest["status"] == "completed"
        assert manifest["run_id"] == run.run_id
        assert Path(manifest["run_dir"]) == run.run_dir
        assert manifest["finished_at_utc"] >= manifest["started_at_utc"]
        assert _read_json(run.run_dir / "config.json") == {"random_state": seed}
        environment = _read_json(run.run_dir / "environment.json")
        assert environment["python_version"]
        assert environment["packages"]["scikit-learn"] == version("scikit-learn")
        pins = (run.run_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
        assert f"scikit-learn=={version('scikit-learn')}" in pins
        assert not any(line.startswith("env-telemetry-risk-pipeline==") for line in pins)


def test_failed_run_preserves_partial_artifacts_and_reraises(tmp_path):
    with pytest.raises(ValueError, match="Training failed"):
        with artifact_run(tmp_path, "train", {}) as run:
            marker = run.processed / "partial.csv"
            marker.write_text("humidity_pct\n50\n", encoding="utf-8")
            raise ValueError("Training failed")

    assert marker.exists()
    assert _read_json(run.run_dir / "run.json")["status"] == "failed"


def test_training_runs_are_self_contained(tmp_path, monkeypatch):
    # Use a real fast estimator so this exercises saving and CV without expensive forests.
    import src.train

    dummy = build_model_candidates()["dummy_most_frequent"]
    monkeypatch.setattr(src.train, "build_model_candidates", lambda: {"dummy_most_frequent": dummy})
    first = run_training(output_dir=tmp_path)
    first_dir = Path(first["run_dir"])
    original_metrics = (first_dir / "outputs" / "metrics.json").read_bytes()
    second = run_training(output_dir=tmp_path)

    assert first["run_id"] != second["run_id"]
    assert (first_dir / "outputs" / "metrics.json").read_bytes() == original_metrics
    for metrics in [first, second]:
        run_dir = Path(metrics["run_dir"])
        assert Path(metrics["model_artifact"]).is_relative_to(run_dir)
        assert Path(metrics["model_artifact"]).is_file()
        assert (run_dir / "processed" / "featured_dataset.csv").is_file()
        assert (run_dir / "figures" / "confusion_matrix.png").is_file()
        metadata = _read_json(run_dir / "outputs" / "model_metadata.json")
        assert metadata["run_id"] == metrics["run_id"]
        config = _read_json(run_dir / "config.json")
        assert config["random_state"] == metadata["random_state"]
        assert config["test_size"] == metadata["test_size"]
        assert config["candidate_models"][metrics["selected_model"]]["classifier_params"] == (
            metadata["selected_model_params"]
        )
        assert _read_json(run_dir / "run.json")["status"] == "completed"


@pytest.mark.parametrize("command", ["train", "anomaly"])
@pytest.mark.parametrize("custom_root", [False, True], ids=["default-root", "custom-root"])
def test_cli_output_root_reports_run(tmp_path, monkeypatch, capsys, command, custom_root):
    import src.train

    dummy = build_model_candidates()["dummy_most_frequent"]
    monkeypatch.setattr(src.train, "build_model_candidates", lambda: {"dummy_most_frequent": dummy})
    # The default root belongs to the caller's working directory, not the package location.
    monkeypatch.chdir(tmp_path)
    output_root = tmp_path / ("custom runs" if custom_root else "runs")
    arguments = [command]
    if custom_root:
        arguments.extend(["--output-dir", str(output_root)])

    assert main(arguments) == 0

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    assert run_dir.parent == output_root
    assert run_dir.name == payload["run_id"]
    assert _read_json(run_dir / "run.json")["status"] == "completed"
    if command == "anomaly":
        summary = _read_json(run_dir / "outputs" / "anomaly_summary.json")
        assert summary["run_id"] == payload["run_id"]
        assert (run_dir / "models" / "anomaly_detector.pkl").is_file()
