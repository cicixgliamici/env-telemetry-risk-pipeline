import argparse
import json

from src.anomaly import detect_anomalies
from src.config import RAW_DATA_PATH
from src.data_loader import load_csv
from src.train import run_training
from src.validation import DataValidationError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the pipeline."""

    parser = argparse.ArgumentParser(description="Environmental telemetry risk pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate the input dataset")
    validate_parser.add_argument(
        "--input",
        default=str(RAW_DATA_PATH),
        help="Path to the raw CSV dataset",
    )

    train_parser = subparsers.add_parser("train", help="Run the training pipeline")
    train_parser.add_argument(
        "--input",
        default=str(RAW_DATA_PATH),
        help="Path to the raw CSV dataset",
    )

    anomaly_parser = subparsers.add_parser("anomaly", help="Run anomaly detection")
    anomaly_parser.add_argument(
        "--input",
        default=str(RAW_DATA_PATH),
        help="Path to the raw CSV dataset",
    )
    anomaly_parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Expected contamination ratio for IsolationForest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            df = load_csv(args.input)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "rows": int(len(df)),
                        "columns": list(df.columns),
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "train":
            metrics = run_training(raw_data_path=args.input)
            print(json.dumps(metrics, indent=2))
            return 0

        if args.command == "anomaly":
            df = detect_anomalies(contamination=args.contamination, raw_data_path=args.input)
            print(
                json.dumps(
                    {
                        "rows": int(len(df)),
                        "anomalies": int(df["is_anomaly"].sum()),
                    },
                    indent=2,
                )
            )
            return 0
    except (DataValidationError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1

    parser.error("Unknown command")
    return 2
