"""Check threshold selection isolation and consistency of saved, batch, and stress decisions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

import src.train as training
from src.cli import main
from src.decision import RiskDecisionPipeline
from src.modeling import build_model_candidates
from src.predict import run_batch_prediction
from src.robustness import run_robustness


@pytest.fixture
def cost_dataset(valid_dataset_df, monkeypatch):
    dataset = pd.concat([valid_dataset_df] * 8, ignore_index=True)
    dataset["timestamp"] = pd.date_range("2025-01-01", periods=len(dataset), freq="h")
    dataset["risk_level"] = np.arange(len(dataset)) % 2
    # Keep the real pipeline and classifier; isolate the decision feature from expensive forests.
    monkeypatch.setattr(
        training,
        "build_model_candidates",
        lambda: {"logistic_regression": build_model_candidates()["logistic_regression"]},
    )
    return dataset


@pytest.mark.parametrize("strategy", ["stratified_random", "temporal"])
def test_cost_training_fits_only_fold_history_and_persists_policy(
    cost_dataset, tmp_path, monkeypatch, strategy
):
    input_path = tmp_path / "input.csv"
    cost_dataset.to_csv(input_path, index=False)
    fits = []
    original_fit = SimpleImputer.fit

    def fit_and_check(imputer, features, y=None):
        fits.append(features.index.tolist())
        fitted = original_fit(imputer, features, y)
        np.testing.assert_allclose(fitted.statistics_, features.median().fillna(0))
        return fitted

    monkeypatch.setattr(SimpleImputer, "fit", fit_and_check)
    result = training.run_training(
        input_path,
        output_dir=tmp_path / "runs",
        split_strategy=strategy,
        temporal_gap=1 if strategy == "temporal" else 0,
        decision_policy="cost",
        false_negative_cost=5,
        false_positive_cost=1,
    )
    run_dir = Path(result["run_dir"])
    protocol = json.loads((run_dir / "outputs" / "evaluation_protocol.json").read_text())
    expected_folds = [fold["training"]["row_indices"] for fold in protocol["folds"]]
    assert fits == expected_folds * 2 + [protocol["training"]["row_indices"]]
    validation = pd.read_csv(run_dir / "outputs" / "threshold_validation_predictions.csv")
    assert set(validation["row_index"]) == {
        row for fold in protocol["folds"] for row in fold["validation"]["row_indices"]
    }
    assert set(validation["row_index"]).isdisjoint(protocol["test"]["row_indices"])
    assert set(validation["row_index"]).isdisjoint(protocol["excluded_row_indices"])
    model = joblib.load(result["model_artifact"])
    assert isinstance(model, RiskDecisionPipeline)
    assert model.risk_decision_policy_ == result["decision_policy"]
    metadata = json.loads((run_dir / "outputs" / "model_metadata.json").read_text())
    assert metadata["decision_policy"] == model.risk_decision_policy_
    assert metadata["artifact_format"] == "telemetry_pipeline_v2_threshold"
    cleaned = pd.read_csv(run_dir / "processed" / "cleaned_dataset.csv")
    test = cleaned.iloc[protocol["test"]["row_indices"]]
    expected = model.predict(test[list(model.feature_names_in_)])
    saved = pd.read_csv(run_dir / "outputs" / "predictions.csv")
    np.testing.assert_array_equal(saved["y_pred"], expected)
    assert (run_dir / "figures" / "threshold_selection.png").is_file()
    assert "Cost-Based Decision Policy" in (run_dir / "outputs" / "training_summary.md").read_text()

    # Scoring needs only the artifact; repeated and shuffled rows keep their policy decisions.
    incoming = test.iloc[[-1, 0, -1]].drop(columns=["risk_level", "failure_risk"])
    incoming_path = tmp_path / "incoming.csv"
    incoming.to_csv(incoming_path, index=False)
    batch = run_batch_prediction(
        result["model_artifact"], incoming_path, tmp_path / "batch.csv", metadata_path=None
    )
    np.testing.assert_array_equal(batch["predicted_risk_level"], expected[[-1, 0, -1]])
    stress = run_robustness(run_dir, output_dir=tmp_path / "stress", repeats=1)
    assert stress["baseline_metrics"]["f1_weighted"] == result["f1_weighted"]
    assert fits == expected_folds * 2 + [protocol["training"]["row_indices"]]


def test_future_labels_cannot_change_selected_threshold(cost_dataset, tmp_path):
    policies = []
    comparisons = []
    for flip in [False, True]:
        dataset = cost_dataset.copy()
        if flip:
            dataset.loc[51:, "risk_level"] = 1 - dataset.loc[51:, "risk_level"]
        input_path = tmp_path / f"input-{flip}.csv"
        dataset.to_csv(input_path, index=False)
        result = training.run_training(
            input_path,
            output_dir=tmp_path / "runs",
            split_strategy="temporal",
            temporal_gap=1,
            decision_policy="cost",
        )
        policies.append(result["decision_policy"])
        comparisons.append(
            pd.read_csv(Path(result["run_dir"]) / "outputs" / "threshold_comparison.csv")
        )
    assert policies[0] == policies[1]
    pd.testing.assert_frame_equal(comparisons[0], comparisons[1])


def test_cli_applies_cost_weights(cost_dataset, tmp_path, capsys):
    input_path = tmp_path / "input.csv"
    cost_dataset.to_csv(input_path, index=False)
    assert (
        main(
            [
                "train",
                "--input",
                str(input_path),
                "--output-dir",
                str(tmp_path / "runs"),
                "--decision-policy",
                "cost",
                "--false-negative-cost",
                "17",
                "--false-positive-cost",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["decision_policy"]["false_negative_cost"] == 17
    assert result["decision_policy"]["false_positive_cost"] == 2
    config = json.loads((Path(result["run_dir"]) / "config.json").read_text())
    assert config["decision_policy"] == "cost"


def test_saved_never_risk_rule_handles_majority_risk_dummy(cost_dataset, tmp_path, monkeypatch):
    cost_dataset.loc[:15, "risk_level"] = 1
    monkeypatch.setattr(
        training,
        "build_model_candidates",
        lambda: {"dummy_most_frequent": build_model_candidates()["dummy_most_frequent"]},
    )
    input_path = tmp_path / "input.csv"
    cost_dataset.to_csv(input_path, index=False)
    result = training.run_training(
        input_path,
        output_dir=tmp_path / "runs",
        decision_policy="cost",
        false_negative_cost=0,
        false_positive_cost=1,
    )
    assert result["decision_policy"]["threshold"] is None
    model = joblib.load(result["model_artifact"])
    features = cost_dataset[list(model.feature_names_in_)]
    assert model.predict(features).tolist() == [0] * len(features)
    assert model.predict_proba(features)[:, list(model.classes_).index(1)].min() == 1
