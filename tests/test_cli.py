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
