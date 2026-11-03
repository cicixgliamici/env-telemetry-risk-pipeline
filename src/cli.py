import argparse
import json

from src.config import RAW_DATA_PATH
from src.data_loader import load_csv
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

    profile_parser = subparsers.add_parser("profile", help="Create a data quality profile")
    profile_parser.add_argument(
        "--input",
        default=str(RAW_DATA_PATH),
        help="Path to the raw CSV dataset",
    )
    profile_parser.add_argument(
        "--output-dir",
        default="data/outputs",
        help="Directory for profile artifacts",
    )

    predict_parser = subparsers.add_parser("predict", help="Run batch prediction")
    predict_parser.add_argument(
        "--input",
        default=str(RAW_DATA_PATH),
        help="Path to the raw CSV dataset",
    )
    predict_parser.add_argument(
        "--model",
        required=True,
        help="Path to a serialized model artifact",
    )
    predict_parser.add_argument(
        "--metadata",
        default="data/outputs/model_metadata.json",
        help="Feature metadata for legacy classifiers; fitted pipelines use their own schema",
    )
    predict_parser.add_argument(
        "--output",
        default="data/outputs/batch_predictions.csv",
        help="Path for prediction CSV output",
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
            from src.train import run_training

            metrics = run_training(raw_data_path=args.input)
            print(json.dumps(metrics, indent=2))
            return 0

        if args.command == "profile":
            from src.profile import run_profile

            profile = run_profile(raw_data_path=args.input, output_dir=args.output_dir)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "rows": profile["rows"],
                        "warnings": profile["warnings"],
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "predict":
            from src.predict import run_batch_prediction

            predictions = run_batch_prediction(
                model_path=args.model,
                raw_data_path=args.input,
                output_path=args.output,
                metadata_path=args.metadata,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "rows": int(len(predictions)),
                        "output": args.output,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "anomaly":
            from src.anomaly import detect_anomalies

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
