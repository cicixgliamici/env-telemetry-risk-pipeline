"""Check frozen-model stress evaluation, source identity, and scenario reproducibility."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import src.train as training
from src.cli import main
from src.evaluate import evaluate_predictions
from src.modeling import build_model_candidates
from src.robustness import run_robustness
from src.robustness_reference import file_sha256, load_evaluation_reference
from src.stress_scenarios import SCENARIOS, scenario_cases, training_stress_profile
from src.validation import validate_dataset_schema


@pytest.fixture
def source_run(tmp_path, monkeypatch, valid_dataset_df, request):
    # A longer alternating history supports real chronological CV with inexpensive candidates.
    dataset = pd.concat([valid_dataset_df] * 8, ignore_index=True)
    dataset["timestamp"] = pd.date_range("2025-01-01", periods=len(dataset), freq="h")
    dataset["risk_level"] = np.arange(len(dataset)) % 2
    if getattr(request, "param", None) == "no-risk-test":
        dataset.loc[51:, "risk_level"] = 0
    elif getattr(request, "param", None) == "missing-pollution":
        dataset.loc[:50, "pollution_proxy"] = np.nan
    input_path = tmp_path / "source.csv"
    dataset.to_csv(input_path, index=False)
    candidate = build_model_candidates()["logistic_regression"]
    monkeypatch.setattr(
        training, "build_model_candidates", lambda: {"logistic_regression": candidate}
    )
    metrics = training.run_training(
        input_path, output_dir=tmp_path / "runs", split_strategy="temporal", temporal_gap=1
    )
    return Path(metrics["run_dir"])


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_scenarios_are_reproducible_and_preserve_case_identity(valid_dataset_df, scenario):
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    original = features.copy()
    profile = training_stress_profile(features.iloc[:6])
    first, first_positions = scenario_cases(features, profile, scenario, 42, 0)
    second, second_positions = scenario_cases(features, profile, scenario, 42, 0)

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(first_positions, second_positions)
    pd.testing.assert_frame_equal(features, original)
    assert len(first) == len(features)
    assert first.columns.tolist() == features.columns.tolist()
    annotated = first.assign(
        timestamp=valid_dataset_df.iloc[first_positions]["timestamp"].to_numpy()
    )
    validate_dataset_schema(annotated, purpose="prediction")
    if scenario.kind == "case_mix":
        pd.testing.assert_frame_equal(first, features.iloc[first_positions].reset_index(drop=True))
    else:
        np.testing.assert_array_equal(first_positions, np.arange(len(features)))


def test_missing_masks_are_nested_and_random_repetitions_differ(valid_dataset_df):
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    profile = training_stress_profile(features)
    mild, _ = scenario_cases(features, profile, SCENARIOS[1], 42, 0)
    severe, _ = scenario_cases(features, profile, SCENARIOS[2], 42, 0)
    other_repeat, _ = scenario_cases(features, profile, SCENARIOS[2], 42, 1)

    assert (mild.isna() <= severe.isna()).to_numpy().all()
    assert not severe.equals(other_repeat)


def test_stress_profile_handles_missing_and_constant_training_features(valid_dataset_df):
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    features["humidity_pct"] = 70
    features["salinity_proxy"] = np.nan
    profile = training_stress_profile(features)

    assert profile["noise_std"]["humidity_pct"] == 0
    assert profile["noise_std"]["salinity_proxy"] == 0
    noisy, _ = scenario_cases(features, profile, SCENARIOS[4], 42, 0)
    assert noisy["humidity_pct"].eq(70).all()
    assert noisy["salinity_proxy"].isna().all()


def test_source_run_recovers_exact_holdout_and_supports_copied_folder(source_run, tmp_path):
    import shutil

    copied = tmp_path / "copied training run"
    shutil.copytree(source_run, copied)
    reference = load_evaluation_reference(copied)
    predictions = pd.read_csv(copied / "outputs" / "predictions.csv")

    assert reference.identity["source_run_dir"] == str(copied)
    np.testing.assert_array_equal(reference.test["risk_level"], predictions["y_true"])
    np.testing.assert_array_equal(reference.baseline_predictions, predictions["y_pred"])
    assert reference.training.index.intersection(reference.test.index).empty


def test_robustness_never_refits_or_changes_source_artifacts(source_run, tmp_path, monkeypatch):
    before = {path: file_sha256(path) for path in source_run.rglob("*") if path.is_file()}

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Robustness experiments must not fit or select a model.")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    monkeypatch.setattr(training, "run_training", unexpected_fit)
    result = run_robustness(source_run, output_dir=tmp_path / "stress runs", repeats=2)

    assert before == {path: file_sha256(path) for path in source_run.rglob("*") if path.is_file()}
    assert result["source_run_id"] == source_run.name
    assert result["scenarios"]["baseline"]["delta_from_baseline"]["f1_weighted"]["mean"] == 0
    assert result["scenarios"]["sensor_bias_0.5std"]["repetitions"] == 1
    assert result["scenarios"]["missing_30pct"]["repetitions"] == 2
    run_dir = Path(result["run_dir"])
    assert json.loads((run_dir / "run.json").read_text())["status"] == "completed"
    assert (run_dir / "figures" / "robustness_f1.png").is_file()
    audit = pd.read_csv(run_dir / "outputs" / "robustness_predictions.csv")
    reference = load_evaluation_reference(source_run)
    for _, cases in audit.groupby(["scenario", "repetition"]):
        expected = reference.test.loc[cases["source_row_index"], "risk_level"].to_numpy()
        np.testing.assert_array_equal(cases["y_true"], expected)
    profile = json.loads((run_dir / "outputs" / "stress_profile.json").read_text())
    assert profile == training_stress_profile(reference.training[reference.feature_columns])
    baseline = result["baseline_metrics"]
    saved_metrics = json.loads((source_run / "outputs" / "metrics.json").read_text())
    assert baseline["f1_weighted"] == saved_metrics["f1_weighted"]


def test_robustness_same_seed_reproduces_scores_and_changed_seed_changes_cases(
    source_run, tmp_path
):
    results = [
        run_robustness(source_run, output_dir=tmp_path / "stress", seed=seed, repeats=2)
        for seed in [42, 42, 43]
    ]
    assert results[0]["scenarios"] == results[1]["scenarios"]
    audits = [
        pd.read_csv(Path(result["run_dir"]) / "outputs" / "robustness_predictions.csv")
        for result in results
    ]
    pd.testing.assert_frame_equal(audits[0], audits[1])
    assert not audits[0].equals(audits[2])


@pytest.mark.parametrize("damage", ["incomplete", "changed-data", "overlap", "wrong-predictions"])
def test_robustness_rejects_inconsistent_source_run(source_run, damage):
    if damage == "incomplete":
        path = source_run / "run.json"
        record = json.loads(path.read_text())
        record["status"] = "running"
        path.write_text(json.dumps(record))
    elif damage == "changed-data":
        path = source_run / "processed" / "cleaned_dataset.csv"
        dataset = pd.read_csv(path)
        dataset.loc[0, "humidity_pct"] += 1
        dataset.to_csv(path, index=False)
    elif damage == "overlap":
        path = source_run / "outputs" / "evaluation_protocol.json"
        record = json.loads(path.read_text())
        record["test"]["row_indices"][0] = record["training"]["row_indices"][0]
        path.write_text(json.dumps(record))
    else:
        path = source_run / "outputs" / "predictions.csv"
        dataset = pd.read_csv(path)
        dataset["y_pred"] = 99
        dataset.to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_evaluation_reference(source_run)


@pytest.mark.parametrize("settings", [{"seed": -1}, {"repeats": 0}, {"repeats": 1.5}])
def test_robustness_rejects_invalid_settings(tmp_path, settings):
    with pytest.raises(ValueError):
        run_robustness(tmp_path / "missing run", output_dir=tmp_path, **settings)


def test_cli_robustness_reports_run_and_missing_source_errors(source_run, tmp_path, capsys):
    assert (
        main(
            [
                "robustness",
                "--training-run",
                str(source_run),
                "--output-dir",
                str(tmp_path / "stress"),
                "--seed",
                "7",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["seed"] == 7
    assert (Path(result["run_dir"]) / "outputs" / "robustness_summary.md").is_file()
    assert main(["robustness", "--training-run", str(tmp_path / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_all_missing_pollution_rejects_case_mix(valid_dataset_df):
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    features["pollution_proxy"] = np.nan
    profile = training_stress_profile(features)
    with pytest.raises(ValueError, match="observed training pollution"):
        scenario_cases(features, profile, SCENARIOS[-1], 42, 0)


@pytest.mark.parametrize("source_run", ["missing-pollution"], indirect=True)
def test_unavailable_case_mix_preserves_other_results(source_run, tmp_path):
    result = run_robustness(source_run, output_dir=tmp_path / "stress", repeats=1)
    assert result["scenarios"]["high_pollution_mix_3x"]["status"] == "unavailable"
    assert result["scenarios"]["missing_30pct"]["status"] == "completed"
    report = (Path(result["run_dir"]) / "outputs" / "robustness_summary.md").read_text()
    assert "No observed training pollution" in report


@pytest.mark.parametrize("source_run", ["no-risk-test"], indirect=True)
def test_risk_recall_is_unavailable_without_risk_cases(source_run, tmp_path):
    result = run_robustness(source_run, output_dir=tmp_path / "stress", repeats=2)
    for scenario in result["scenarios"].values():
        assert scenario["metrics"]["recall_risk"]["mean"] is None
        assert scenario["metrics"]["recall_risk"]["observed_repetitions"] == 0
    # Strict JSON serialization must not need NaN or infinity as a substitute for missing metrics.
    json.dumps(result, allow_nan=False)


def test_reported_metrics_are_recomputable_from_audit_rows(source_run, tmp_path):
    result = run_robustness(source_run, output_dir=tmp_path / "stress", repeats=1)
    run_dir = Path(result["run_dir"])
    audit = pd.read_csv(run_dir / "outputs" / "robustness_predictions.csv")
    model = joblib.load(next((source_run / "models").glob("*_model.pkl")))
    for name, cases in audit.groupby("scenario"):
        metrics = evaluate_predictions(cases["y_true"], cases["y_pred"])
        assert result["scenarios"][name]["metrics"]["f1_weighted"]["mean"] == metrics["f1_weighted"]
        predictions = model.predict(cases[list(model.feature_names_in_)])
        np.testing.assert_array_equal(cases["y_pred"], predictions)
