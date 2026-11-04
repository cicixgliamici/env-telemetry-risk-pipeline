import pytest

from src.anomaly import detect_anomalies


def test_detect_anomalies_rejects_invalid_contamination():
    with pytest.raises(ValueError, match="contamination"):
        detect_anomalies(contamination=0)


def test_detect_anomalies_returns_expected_columns(tmp_path):
    out = detect_anomalies(contamination=0.2, output_dir=tmp_path)

    assert "anomaly_score" in out.columns
    assert "is_anomaly" in out.columns
    assert "zscore_flag" in out.columns


@pytest.mark.parametrize("include_labels", [False, True])
def test_anomaly_detection_accepts_unlabeled_and_single_class_batches(
    valid_dataset_df, tmp_path, include_labels
):
    dataset = valid_dataset_df.iloc[:4].copy()
    if not include_labels:
        dataset = dataset.drop(columns=["risk_level", "failure_risk"])
    input_path = tmp_path / "incoming.csv"
    dataset.to_csv(input_path, index=False)

    result = detect_anomalies(
        contamination=0.25, raw_data_path=input_path, output_dir=tmp_path / "runs"
    )

    assert len(result) == len(dataset)
    assert result["anomaly_score"].notna().all()
    assert result["is_anomaly"].isin([0, 1]).all()
    assert ("risk_level" in result.columns) == include_labels
