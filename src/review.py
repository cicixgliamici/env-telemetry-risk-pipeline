"""Reproduce the synthetic evidence and link it in one portable review directory."""

from pathlib import Path

from src.anomaly import detect_anomalies
from src.artifacts import artifact_run
from src.config import RANDOM_STATE, RUNS_DIR
from src.decision_evaluation import decision_summary_lines
from src.predict import run_batch_prediction
from src.profile import run_profile
from src.robustness import run_robustness
from src.synthetic_data import write_synthetic_dataset
from src.train import run_training
from src.utils import save_json, write_text_file


def _evidence_links(summary: dict, root: Path) -> list[str]:
    """Link reports, figures, and membership records without machine-specific paths."""

    links = []
    for name, result in summary["training"].items():
        directory = Path(result["run_dir"]).relative_to(root).as_posix()
        links.extend(
            [
                f"- [{name} training report]({directory}/outputs/training_summary.md)",
                f"- [{name} confusion matrix]({directory}/figures/confusion_matrix.png)",
                f"- [{name} evaluation membership]({directory}/outputs/evaluation_protocol.json)",
            ]
        )
        if result["decision_policy"]["mode"] == "cost":
            links.append(
                f"- [Training-only cutoff selection]({directory}/figures/threshold_selection.png)"
            )
    for name, filename, label in [
        ("robustness", "robustness_summary.md", "Frozen-model robustness"),
        ("anomaly", "anomaly_summary.json", "Anomaly summary"),
    ]:
        report = Path(summary[name]["run_dir"]) / "outputs" / filename
        links.append(f"- [{label}]({report.relative_to(root).as_posix()})")
    return links


def _report(summary: dict, root: Path) -> str:
    """Use relative links so the entire directory can be shared with a reviewer."""

    lines = [
        "# Synthetic Review Evidence",
        "",
        "Generated runtime evidence; tests and package checks run separately.",
        "",
        f"Dataset: {summary['rows']} synthetic rows; seed {summary['seed']}; "
        f"stress repetitions {summary['repeats']}. Model/split seed: {RANDOM_STATE}.",
        "",
        "| Protocol | Selected model | Weighted F1 | Risk recall |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, metrics in summary["training"].items():
        recall = metrics["recall_risk"]
        recall_text = "unavailable" if recall is None else f"{recall:.4f}"
        lines.append(
            f"| {name} | {metrics['selected_model']} | "
            f"{metrics['f1_weighted']:.4f} | {recall_text} |"
        )
    lines.extend(
        [
            "",
            *decision_summary_lines(summary["training"]["temporal_cost"]),
            "",
            "## Evidence",
            "",
            *_evidence_links(summary, root),
        ]
    )
    lines.extend(
        [
            "- [Dataset profile](outputs/profile/data_profile.md)",
            "- [Batch predictions](outputs/batch_predictions.csv)",
            "- [Machine-readable summary](outputs/review_summary.json)",
            "",
            "## Interpretation limits",
            "",
            "Synthetic labels demonstrate implementation behavior, not observed failures. "
            "Random and temporal protocols use different holdouts and cannot isolate "
            "a causal effect.",
            "The cost policy uses illustrative 5:1 missed-risk/false-alert weights. "
            "Its threshold comes from training validation only; probabilities are uncalibrated. "
            "The temporal classifier and cost runs reuse the same holdout, so their comparison "
            "is exploratory rather than fresh independent confirmation.",
            "Stress repetitions describe one frozen model and holdout, not confidence intervals. "
            "Anomaly flags are unsupervised and do not establish risk labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_experiments(dataset: Path, directory: Path) -> dict:
    """Keep protocol differences explicit rather than choosing the best test result."""

    settings = {
        "stratified_random": {},
        "temporal": {"split_strategy": "temporal", "temporal_gap": 1},
        "temporal_cost": {
            "split_strategy": "temporal",
            "temporal_gap": 1,
            "decision_policy": "cost",
            "false_negative_cost": 5,
            "false_positive_cost": 1,
        },
    }
    return {
        name: run_training(dataset, output_dir=directory, **options)
        for name, options in settings.items()
    }


def run_review(
    *, rows: int = 1000, seed: int = 42, repeats: int = 5, output_dir: str | Path = RUNS_DIR
) -> dict:
    """Generate self-contained runtime evidence without modifying tracked input data."""

    if rows < 20 or seed < 0 or repeats < 1:
        raise ValueError("Review requires rows >= 20, seed >= 0, and repeats >= 1.")
    with artifact_run(
        output_dir, "review", {"rows": rows, "seed": seed, "repeats": repeats}, compact=True
    ) as artifacts:
        dataset = write_synthetic_dataset(artifacts.processed / "synthetic_dataset.csv", rows, seed)
        run_profile(dataset, output_dir=artifacts.outputs / "profile")
        directory = artifacts.run_dir / "experiments"
        training = _run_experiments(dataset, directory)
        stress = run_robustness(
            training["temporal"]["run_dir"], output_dir=directory, seed=seed, repeats=repeats
        )
        run_batch_prediction(
            model_path=training["temporal_cost"]["model_artifact"],
            raw_data_path=dataset,
            output_path=artifacts.outputs / "batch_predictions.csv",
        )
        anomalies = detect_anomalies(
            raw_data_path=dataset, output_dir=directory, contamination=0.05
        )
        summary = {
            **artifacts.identity(),
            "training": training,
            "robustness": stress,
            "anomaly": dict(anomalies.attrs),
            "rows": rows,
            "seed": seed,
            "model_and_split_seed": RANDOM_STATE,
            "repeats": repeats,
        }
        save_json(summary, artifacts.outputs / "review_summary.json")
        report = artifacts.run_dir / "REVIEW_SUMMARY.md"
        write_text_file(_report(summary, artifacts.run_dir), report)
        return {**artifacts.identity(), "report": str(report)}
