"""Verify chronological boundaries, preprocessing isolation, and protocol evidence."""

import json
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

import src.train as training
from src.cli import main
from src.modeling import build_model_candidates
from src.preprocessing import split_features_target
from src.splitting import build_training_folds, validate_split_settings


@pytest.fixture
def temporal_dataset(valid_dataset_df):
    dataset = pd.concat([valid_dataset_df] * 6, ignore_index=True)
    # Two observations per instant expose any row-based boundary that splits equal timestamps.
    dataset["timestamp"] = pd.date_range("2025-01-01", periods=24, freq="h").repeat(2)
    dataset["risk_level"] = np.repeat(np.arange(24) % 2, 2)
    dataset["failure_risk"] = dataset["risk_level"]
    return dataset


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_temporal_holdout_keeps_timestamp_groups_and_sorts(temporal_dataset, gap):
    shuffled = temporal_dataset.sample(frac=1, random_state=7)
    train, test, y_train, y_test = split_features_target(
        shuffled, "risk_level", 0.25, 42, split_strategy="temporal", temporal_gap=gap
    )

    train_times = temporal_dataset.loc[train.index, "timestamp"]
    test_times = temporal_dataset.loc[test.index, "timestamp"]
    assert train_times.is_monotonic_increasing
    assert test_times.is_monotonic_increasing
    assert train_times.max() < test_times.min()
    assert len(test) == ceil(24 * 0.25) * 2
    assert len(train) + len(test) == len(temporal_dataset) - gap * 2
    assert set(train_times).isdisjoint(test_times)
    pd.testing.assert_series_equal(y_train, temporal_dataset.loc[train.index, "risk_level"])
    pd.testing.assert_series_equal(y_test, temporal_dataset.loc[test.index, "risk_level"])
    assert set(train.columns).isdisjoint({"timestamp", "risk_level", "failure_risk"})


def test_temporal_cv_expands_history_and_honors_group_gap(temporal_dataset):
    target = temporal_dataset["risk_level"]
    times = temporal_dataset["timestamp"]
    folds = build_training_folds(target, strategy="temporal", timestamps=times, temporal_gap=2)
    previous_train = set()

    for train, validation in folds:
        train_times, validation_times = times.iloc[train], times.iloc[validation]
        assert train_times.max() < validation_times.min()
        assert set(train_times).isdisjoint(validation_times)
        assert previous_train < set(train)
        gap_times = times[(times > train_times.max()) & (times < validation_times.min())]
        assert gap_times.nunique() == 2
        assert target.iloc[train].nunique() == 2
        previous_train = set(train)
    assert len(folds) == 3


def test_temporal_test_fraction_counts_timestamps_not_rows(temporal_dataset):
    # Uneven group sizes must not silently change the documented timestamp-fraction rule.
    dataset = temporal_dataset.drop(index=range(0, 12, 2))
    train, test, _, _ = split_features_target(
        dataset, "risk_level", 0.25, 42, split_strategy="temporal"
    )
    assert dataset.loc[test.index, "timestamp"].nunique() == 6
    assert len(test) == 12
    assert len(train) + len(test) == len(dataset)


def test_temporal_training_imputes_only_past_data_and_records_membership(
    temporal_dataset, tmp_path, monkeypatch
):
    dataset = temporal_dataset.copy()
    dataset.loc[0, "humidity_pct"] = np.nan
    # Large late values would change medians if validation or test leaked into fitting.
    dataset.loc[30:, "humidity_pct"] = 100
    input_path = tmp_path / "input.csv"
    dataset.sample(frac=1, random_state=9).to_csv(input_path, index=False)
    candidates = build_model_candidates()
    monkeypatch.setattr(
        training,
        "build_model_candidates",
        lambda: {"logistic_regression": candidates["logistic_regression"]},
    )
    fitted_indices = []
    original_fit = SimpleImputer.fit

    def check_fit(imputer, features, y=None):
        fitted = original_fit(imputer, features, y)
        fitted_indices.append(features.index.tolist())
        np.testing.assert_allclose(fitted.statistics_, features.median().fillna(0))
        return fitted

    monkeypatch.setattr(SimpleImputer, "fit", check_fit)
    metrics = training.run_training(
        input_path, output_dir=tmp_path / "runs", split_strategy="temporal", temporal_gap=1
    )
    run_dir = Path(metrics["run_dir"])
    protocol = json.loads((run_dir / "outputs" / "evaluation_protocol.json").read_text())
    assert metrics["cv_strategy"] == "expanding_window"
    assert protocol["training"]["timestamp_max"] < protocol["test"]["timestamp_min"]
    assert len(protocol["excluded_row_indices"]) == 2
    assert fitted_indices == [fold["training"]["row_indices"] for fold in protocol["folds"]] + [
        protocol["training"]["row_indices"]
    ]
    held_out = set(protocol["test"]["row_indices"] + protocol["excluded_row_indices"])
    for fold in protocol["folds"]:
        assert fold["training"]["timestamp_max"] < fold["validation"]["timestamp_min"]
        assert held_out.isdisjoint(fold["training"]["row_indices"])
        assert held_out.isdisjoint(fold["validation"]["row_indices"])
    cleaned = pd.read_csv(run_dir / "processed" / "cleaned_dataset.csv")
    pipeline = joblib.load(metrics["model_artifact"])
    humidity_index = pipeline.feature_names_in_.tolist().index("humidity_pct")
    assert pipeline.named_steps["imputer"].statistics_[humidity_index] == (
        cleaned.loc[protocol["training"]["row_indices"], "humidity_pct"].median()
    )
    summary = (run_dir / "outputs" / "training_summary.md").read_text()
    assert "expanding_window" in summary
    assert protocol["test"]["timestamp_min"] in summary


def test_temporal_selection_is_unchanged_by_future_labels(temporal_dataset, tmp_path, monkeypatch):
    candidates = build_model_candidates()
    monkeypatch.setattr(
        training,
        "build_model_candidates",
        lambda: {name: candidates[name] for name in ["logistic_regression", "dummy_most_frequent"]},
    )
    results = []
    for flip in [False, True]:
        dataset = temporal_dataset.copy()
        if flip:
            dataset.loc[38:, "risk_level"] = 1 - dataset.loc[38:, "risk_level"]
        input_path = tmp_path / f"input-{flip}.csv"
        dataset.to_csv(input_path, index=False)
        results.append(
            training.run_training(
                input_path, output_dir=tmp_path / "runs", split_strategy="temporal"
            )
        )

    assert results[0]["selected_model"] == results[1]["selected_model"]
    assert results[0]["cross_validation"] == results[1]["cross_validation"]


def test_cli_temporal_settings_are_applied(temporal_dataset, tmp_path, monkeypatch, capsys):
    dummy = build_model_candidates()["dummy_most_frequent"]
    monkeypatch.setattr(training, "build_model_candidates", lambda: {"dummy_most_frequent": dummy})
    input_path = tmp_path / "input.csv"
    temporal_dataset.to_csv(input_path, index=False)

    assert (
        main(
            [
                "train",
                "--input",
                str(input_path),
                "--output-dir",
                str(tmp_path / "runs"),
                "--split-strategy",
                "temporal",
                "--test-size",
                "0.25",
                "--temporal-gap",
                "1",
            ]
        )
        == 0
    )

    metrics = json.loads(capsys.readouterr().out)
    config = json.loads((Path(metrics["run_dir"]) / "config.json").read_text())
    assert config["split_strategy"] == "temporal"
    assert config["test_size"] == 0.25
    assert config["temporal_gap"] == 1
    assert metrics["split_strategy"] == "temporal"


@pytest.mark.parametrize("reason", ["few-timestamps", "one-class-history", "large-gap"])
def test_temporal_cv_rejects_insufficient_history(temporal_dataset, reason):
    dataset = temporal_dataset.copy()
    gap = 0
    if reason == "few-timestamps":
        dataset = dataset.iloc[:4]
    elif reason == "one-class-history":
        dataset.loc[:35, "risk_level"] = 0
    else:
        gap = 24

    with pytest.raises(ValueError, match="Temporal CV"):
        build_training_folds(
            dataset["risk_level"],
            strategy="temporal",
            timestamps=dataset["timestamp"],
            temporal_gap=gap,
        )


@pytest.mark.parametrize("test_size", [0, 1, -0.1, float("nan"), float("inf")])
def test_split_settings_reject_invalid_test_fraction(test_size):
    with pytest.raises(ValueError, match="test_size"):
        validate_split_settings("temporal", test_size, 0)


@pytest.mark.parametrize(
    "strategy,gap", [("unknown", 0), ("temporal", -1), ("stratified_random", 1)]
)
def test_split_settings_reject_incompatible_options(strategy, gap):
    with pytest.raises(ValueError):
        validate_split_settings(strategy, 0.2, gap)


def test_temporal_holdout_rejects_empty_training_history(temporal_dataset):
    with pytest.raises(ValueError, match="no training timestamps"):
        split_features_target(
            temporal_dataset.iloc[:2], "risk_level", 0.2, 42, split_strategy="temporal"
        )


def test_cli_temporal_failure_records_failed_run(tmp_path, capsys):
    # The bundled sample's early history has one class; never fall back to random CV.
    output_root = tmp_path / "runs"
    assert main(["train", "--split-strategy", "temporal", "--output-dir", str(output_root)]) == 1
    error = json.loads(capsys.readouterr().out)
    assert "Temporal CV fold 1" in error["message"]
    runs = list(output_root.iterdir())
    assert len(runs) == 1
    assert json.loads((runs[0] / "run.json").read_text())["status"] == "failed"


def test_cv_risk_recall_uses_only_folds_with_observed_risk():
    features = pd.DataFrame({"humidity_pct": np.arange(8) + 50})
    target = pd.Series([0, 1, 0, 0, 0, 0, 0, 1])
    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([0, 1, 2, 3]), np.array([6, 7]))]
    candidates = {"dummy": build_model_candidates()["dummy_most_frequent"]}
    result = training._cross_validate_candidates(candidates, features, target, cv_splits=folds)[
        "dummy"
    ]

    assert result["recall_risk_mean"] == 0.0
    assert result["recall_risk_std"] == 0.0
    assert result["recall_risk_observed_folds"] == 1


def test_cv_risk_recall_is_unavailable_when_all_validation_windows_lack_risk():
    features = pd.DataFrame({"humidity_pct": np.arange(6) + 50})
    target = pd.Series([0, 1, 0, 0, 0, 0])
    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([0, 1, 2, 3]), np.array([4, 5]))]
    candidates = {"dummy": build_model_candidates()["dummy_most_frequent"]}
    result = training._cross_validate_candidates(candidates, features, target, cv_splits=folds)[
        "dummy"
    ]

    assert result["recall_risk_mean"] is None
    assert result["recall_risk_std"] is None
    assert result["recall_risk_observed_folds"] == 0
