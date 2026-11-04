from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.modeling import build_model_candidates
from src.predict import prepare_prediction_features, run_batch_prediction
from src.train import run_training


class ConstantModel:
    def predict(self, X):
        return [1] * len(X)


class HumidityModel:
    """Return row-dependent outputs to expose ordering errors in batch inference."""

    classes_ = [0, 1]

    def predict(self, X):
        return (X["humidity_pct"] >= 80).astype(int).to_numpy()

    def predict_proba(self, X):
        probability = X["humidity_pct"].to_numpy() / 100.0
        return pd.DataFrame({0: 1 - probability, 1: probability}).to_numpy()


class OrderedProbabilityModel:
    """Expose deliberately reordered classes to catch positional probability lookup."""

    def __init__(self, classes):
        if classes is not None:
            self.classes_ = classes

    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        # Distinct values make a wrong class-to-column association observable.
        return pd.DataFrame([[0.6, 0.3, 0.1]] * len(X)).to_numpy()


@pytest.mark.parametrize(
    "classes, expected_probability",
    [([1, 2, 0], 0.6), ([2, 0, 1], 0.1), ([0, 2, 3], None), (None, None)],
    ids=["risk-first", "risk-last", "risk-absent", "class-order-unavailable"],
)
def test_batch_prediction_respects_class_order(
    valid_dataset_df, tmp_path, classes, expected_probability
):
    input_path = tmp_path / "incoming.csv"
    output_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "model.pkl"
    valid_dataset_df.to_csv(input_path, index=False)
    joblib.dump(OrderedProbabilityModel(classes), model_path)

    result = run_batch_prediction(model_path, input_path, output_path, metadata_path=None)

    for predictions in [result, pd.read_csv(output_path)]:
        assert len(predictions) == len(valid_dataset_df)
        assert predictions["predicted_risk_level"].eq(1).all()
        if expected_probability is None:
            assert "predicted_risk_probability" not in predictions
        else:
            assert predictions["predicted_risk_probability"].tolist() == pytest.approx(
                [expected_probability] * len(valid_dataset_df)
            )


@pytest.mark.parametrize(
    "labels, expected_risk_index",
    [([0, 1], 1), ([1, 2], 0), ([0, 1, 2], 1), ([0, 2], None), ([1], 0), ([2], None)],
    ids=["binary", "risk-first", "multiclass", "risk-absent", "risk-only", "other-only"],
)
def test_saved_pipeline_probability_matches_risk_class(
    valid_dataset_df, tmp_path, labels, expected_risk_index
):
    # The dummy supports singleton classes too, exercising the full saved pipeline contract.
    pipeline = build_model_candidates()["dummy_most_frequent"]
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    target = [labels[index % len(labels)] for index in range(len(features))]
    pipeline.fit(features, target)
    model_path = tmp_path / "pipeline.pkl"
    input_path = tmp_path / "incoming.csv"
    output_path = tmp_path / "predictions.csv"
    joblib.dump(pipeline, model_path)
    valid_dataset_df.drop(columns=["risk_level", "failure_risk"]).to_csv(input_path, index=False)

    result = run_batch_prediction(model_path, input_path, output_path)

    for predictions in [result, pd.read_csv(output_path)]:
        assert predictions["predicted_risk_level"].tolist() == pipeline.predict(features).tolist()
        if expected_risk_index is None:
            assert "predicted_risk_probability" not in predictions
        else:
            expected = pipeline.predict_proba(features)[:, expected_risk_index]
            assert predictions["predicted_risk_probability"].tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    "row_positions",
    [
        [7, 0, 6, 1, 5, 2, 4, 3],
        [0, 1, 1, 2, 3, 4, 5, 6, 7],
        [7, 0, 7, 1, 6, 1, 5, 2, 4, 3],
    ],
    ids=["shuffled", "duplicates", "shuffled_duplicates"],
)
def test_prediction_preserves_input_rows(valid_dataset_df, tmp_path: Path, row_positions):
    dataset = valid_dataset_df.iloc[row_positions].reset_index(drop=True)
    input_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "model.pkl"
    dataset.to_csv(input_path, index=False)
    joblib.dump(HumidityModel(), model_path)

    features = prepare_prediction_features(input_path, feature_columns=["humidity_pct"])
    pd.testing.assert_series_equal(features["humidity_pct"], dataset["humidity_pct"])

    result = run_batch_prediction(
        model_path=model_path,
        raw_data_path=input_path,
        output_path=output_path,
        metadata_path=None,
    )

    # Check persisted results too: both labels and probabilities must follow input order.
    for predictions in [result, pd.read_csv(output_path)]:
        assert len(predictions) == len(dataset)
        assert (
            pd.to_datetime(predictions["timestamp"]).tolist()
            == pd.to_datetime(dataset["timestamp"]).tolist()
        )
        pd.testing.assert_frame_equal(
            predictions[dataset.columns].drop(columns="timestamp"),
            dataset.drop(columns="timestamp"),
        )
        assert (
            predictions["predicted_risk_level"].tolist()
            == (dataset["humidity_pct"] >= 80).astype(int).tolist()
        )
        assert predictions["predicted_risk_probability"].tolist() == pytest.approx(
            dataset["humidity_pct"] / 100.0
        )


def test_prepare_prediction_features_uses_metadata_feature_order(valid_dataset_df, tmp_path: Path):
    input_path = tmp_path / "dataset.csv"
    valid_dataset_df.to_csv(input_path, index=False)
    feature_columns = ["humidity_pct", "pressure_hpa"]

    X = prepare_prediction_features(input_path, feature_columns=feature_columns)

    assert list(X.columns) == feature_columns
    assert len(X) == len(valid_dataset_df)


def test_run_batch_prediction_writes_output(valid_dataset_df, tmp_path: Path):
    input_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "model.pkl"
    valid_dataset_df.to_csv(input_path, index=False)
    joblib.dump(ConstantModel(), model_path)

    result = run_batch_prediction(
        model_path=model_path,
        raw_data_path=input_path,
        output_path=output_path,
        metadata_path=None,
    )

    assert "predicted_risk_level" in result.columns
    assert result["predicted_risk_level"].eq(1).all()
    assert output_path.exists()
    assert "predicted_risk_level" in pd.read_csv(output_path).columns


def test_batch_prediction_can_use_training_metadata(tmp_path):
    metrics = run_training(output_dir=tmp_path / "runs")
    model_path = Path(metrics["model_artifact"])

    result = run_batch_prediction(
        model_path=model_path, output_path=tmp_path / "batch_predictions.csv"
    )

    assert "predicted_risk_level" in result.columns
    assert len(result) > 0


@pytest.mark.parametrize("include_labels", [False, True])
@pytest.mark.parametrize("row_positions", [[0], [0, 0]])
def test_pipeline_scores_unlabeled_and_single_class_batches(
    valid_dataset_df, tmp_path, include_labels, row_positions
):
    pipeline = build_model_candidates()["logistic_regression"]
    raw_columns = valid_dataset_df.columns.difference(["timestamp", "risk_level", "failure_risk"])
    pipeline.fit(valid_dataset_df[raw_columns], valid_dataset_df["risk_level"])
    model_path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, model_path)
    incoming = valid_dataset_df.iloc[row_positions].reset_index(drop=True)
    if not include_labels:
        incoming = incoming.drop(columns=["risk_level", "failure_risk"])
    input_path = tmp_path / "incoming.csv"
    incoming.to_csv(input_path, index=False)

    result = run_batch_prediction(model_path, input_path, tmp_path / "predictions.csv")

    assert len(result) == len(incoming)
    assert (
        result["predicted_risk_level"].tolist() == pipeline.predict(incoming[raw_columns]).tolist()
    )
    assert ("risk_level" in result.columns) == include_labels
    assert ("failure_risk" in result.columns) == include_labels
