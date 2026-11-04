"""Verify the review handoff contains completed, reproducible, linked evidence."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

import src.train as training
from src.cli import main
from src.modeling import build_model_candidates
from src.review import run_review
from src.visualize import plot_confusion_matrix


def test_review_bundle_has_completed_runs_and_resolvable_links(tmp_path, monkeypatch):
    # Exercise real training and serialization with one fast candidate per protocol.
    candidate = build_model_candidates()["logistic_regression"]
    monkeypatch.setattr(
        training, "build_model_candidates", lambda: {"logistic_regression": candidate}
    )
    result = run_review(rows=250, repeats=1, output_dir=tmp_path)
    root = Path(result["run_dir"])
    summary = json.loads((root / "outputs/review_summary.json").read_text())
    manifests = list(root.rglob("run.json"))
    assert len(manifests) == 6
    assert all(json.loads(path.read_text())["status"] == "completed" for path in manifests)
    assert set(summary["training"]) == {"stratified_random", "temporal", "temporal_cost"}
    assert summary["training"]["temporal_cost"]["decision_policy"]["mode"] == "cost"
    assert len(pd.read_csv(root / "outputs/batch_predictions.csv")) == 250
    for target in re.findall(r"\]\(([^)]+)\)", Path(result["report"]).read_text()):
        assert (root / target).is_file(), target


@pytest.mark.parametrize("labels,counts", [([2, 5], [[3, 1], [2, 4]]), ([1], [[7]])])
def test_confusion_plot_preserves_class_names_and_counts(tmp_path, monkeypatch, labels, counts):
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    plot_confusion_matrix(counts, tmp_path / "confusion.png", labels=labels)
    axis = figures[0].axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == list(map(str, labels))
    assert [tick.get_text() for tick in axis.get_yticklabels()] == list(map(str, labels))
    assert [text.get_text() for text in axis.texts] == [str(n) for row in counts for n in row]


def test_review_cli_rejects_invalid_settings(capsys):
    assert main(["review", "--repeats", "0"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
