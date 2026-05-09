from src.cli import build_parser, main


def test_cli_train_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["train", "--input", "data/raw/raw_dataset.csv"])

    assert args.command == "train"
    assert args.input.endswith("raw_dataset.csv")


def test_cli_validate_returns_error_for_missing_file(capsys):
    exit_code = main(["validate", "--input", "data/raw/does_not_exist.csv"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert '"status": "error"' in captured.out


def test_cli_validate_returns_success_for_valid_dataset(capsys):
    exit_code = main(["validate", "--input", "data/raw/raw_dataset.csv"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "ok"' in captured.out


def test_cli_anomaly_returns_error_for_invalid_contamination(capsys):
    exit_code = main(
        ["anomaly", "--input", "data/raw/raw_dataset.csv", "--contamination", "0"]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert '"status": "error"' in captured.out


def test_cli_train_prints_metrics(monkeypatch, capsys):
    import src.train

    monkeypatch.setattr(
        src.train,
        "run_training",
        lambda raw_data_path: {"accuracy": 0.75, "input": raw_data_path},
    )

    exit_code = main(["train", "--input", "data/raw/raw_dataset.csv"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"accuracy": 0.75' in captured.out


def test_cli_anomaly_prints_summary(monkeypatch, capsys, valid_dataset_df):
    import src.anomaly

    out = valid_dataset_df.copy()
    out["is_anomaly"] = [0, 0, 0, 0, 1, 0, 0, 1]
    monkeypatch.setattr(
        src.anomaly,
        "detect_anomalies",
        lambda contamination, raw_data_path: out,
    )

    exit_code = main(
        ["anomaly", "--input", "data/raw/raw_dataset.csv", "--contamination", "0.2"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"rows": 8' in captured.out
    assert '"anomalies": 2' in captured.out


def test_cli_profile_prints_summary(monkeypatch, capsys):
    import src.profile

    monkeypatch.setattr(
        src.profile,
        "run_profile",
        lambda raw_data_path, output_dir: {"rows": 12, "warnings": []},
    )

    exit_code = main(["profile", "--input", "data/raw/raw_dataset.csv"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "ok"' in captured.out
    assert '"rows": 12' in captured.out


def test_cli_predict_prints_summary(monkeypatch, capsys, valid_dataset_df):
    import src.predict

    out = valid_dataset_df.copy()
    out["predicted_risk_level"] = 1
    monkeypatch.setattr(
        src.predict,
        "run_batch_prediction",
        lambda model_path, raw_data_path, output_path, metadata_path: out,
    )

    exit_code = main(
        [
            "predict",
            "--input",
            "data/raw/raw_dataset.csv",
            "--model",
            "models/model.pkl",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "ok"' in captured.out
    assert '"rows": 8' in captured.out
