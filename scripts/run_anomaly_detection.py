from src.anomaly import detect_anomalies


if __name__ == "__main__":
    result = detect_anomalies()
    print(result["is_anomaly"].value_counts().to_dict())
