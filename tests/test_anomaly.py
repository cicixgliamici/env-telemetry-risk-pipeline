import pytest

from src.anomaly import detect_anomalies


def test_detect_anomalies_rejects_invalid_contamination():
    with pytest.raises(ValueError, match="contamination"):
        detect_anomalies(contamination=0)


def test_detect_anomalies_returns_expected_columns():
    out = detect_anomalies(contamination=0.2)

    assert "anomaly_score" in out.columns
    assert "is_anomaly" in out.columns
    assert "zscore_flag" in out.columns
