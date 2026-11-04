import argparse
import json

from src.config import RANDOM_STATE, RAW_DATA_PATH, RUNS_DIR, TEST_SIZE
from src.data_loader import load_csv
from src.splitting import SPLIT_STRATEGIES
from src.validation import DataValidationError, ValidationPurpose


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the pipeline."""

    parser = argparse.ArgumentParser(description="Environmental telemetry risk pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate the input dataset")
    validate_parser.add_argument(
        "--purpose",
        choices=[purpose.value for purpose in ValidationPurpose],
        default=ValidationPurpose.TRAINING.value,
        help="Choose training label requirements or telemetry-only prediction/anomaly checks",
    )
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
    train_parser.add_argument(
        "--split-strategy",
        choices=SPLIT_STRATEGIES,
        default="stratified_random",
        help="Use stratified random evaluation or chronological holdout and expanding CV",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test fraction: rows for random splitting, distinct timestamps for temporal splitting",
    )
    train_parser.add_argument(
        "--temporal-gap",
        type=int,
        default=0,
        help="Timestamp groups excluded before temporal test and CV validation windows",
    )
    train_parser.add_argument(
        "--decision-policy",
        choices=["classifier", "cost"],
        default="classifier",
        help="Keep classifier decisions or choose a binary risk cutoff "
        "using training-only error cost",
    )
    train_parser.add_argument(
        "--false-negative-cost",
        type=float,
        default=5.0,
        help="Relative cost of a missed risk case (illustrative default: 5)",
    )
    train_parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=1.0,
        help="Relative cost of a false risk alert (illustrative default: 1)",
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
    robustness_parser = subparsers.add_parser(
        "robustness", help="Stress-test a frozen model on its recorded held-out observations"
    )
    robustness_parser.add_argument(
        "--training-run", required=True, help="Directory of a completed supervised training run"
    )
    robustness_parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE, help="Perturbation/resampling seed"
    )
    robustness_parser.add_argument(
        "--repeats", type=int, default=5, help="Repetitions per stochastic scenario (default: 5)"
    )
    for run_parser in [train_parser, anomaly_parser, robustness_parser]:
        run_parser.add_argument(
            "--output-dir",
            default=str(RUNS_DIR),
            help="Parent directory for a new, unique run folder (default: runs)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            df = load_csv(args.input, purpose=args.purpose)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "purpose": args.purpose,
                        "rows": int(len(df)),
                        "columns": list(df.columns),
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "train":
            from src.train import run_training

            metrics = run_training(
                raw_data_path=args.input,
                output_dir=args.output_dir,
                split_strategy=args.split_strategy,
                test_size=args.test_size,
                temporal_gap=args.temporal_gap,
                decision_policy=args.decision_policy,
                false_negative_cost=args.false_negative_cost,
                false_positive_cost=args.false_positive_cost,
            )
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

        if args.command == "robustness":
            from src.robustness import run_robustness

            result = run_robustness(
                args.training_run, output_dir=args.output_dir, seed=args.seed, repeats=args.repeats
            )
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "anomaly":
            from src.anomaly import detect_anomalies

            df = detect_anomalies(
                contamination=args.contamination,
                raw_data_path=args.input,
                output_dir=args.output_dir,
            )
            print(
                json.dumps(
                    {
                        **df.attrs,
                        "rows": int(len(df)),
                        "anomalies": int(df["is_anomaly"].sum()),
                    },
                    indent=2,
                )
            )
            return 0
    except (DataValidationError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1

    parser.error("Unknown command")
    return 2
