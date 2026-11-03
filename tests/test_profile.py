import json
from pathlib import Path

from src.profile import build_data_profile, format_profile_markdown, run_profile


def test_build_data_profile_summarizes_dataset(valid_dataset_df):
    profile = build_data_profile(valid_dataset_df)

    assert profile["rows"] == len(valid_dataset_df)
    assert profile["target_counts"] == {"0": 4, "1": 4}
    assert profile["missing_values"]["humidity_pct"] == 0
    assert "humidity_pct" in profile["numeric_summary"]
    assert profile["warnings"]


def test_format_profile_markdown_includes_core_sections(valid_dataset_df):
    profile = build_data_profile(valid_dataset_df)

    markdown = format_profile_markdown(profile)

    assert "# Data Profile" in markdown
    assert "## Target Balance" in markdown
    assert "## Numeric Summary" in markdown


def test_run_profile_writes_json_and_markdown(valid_dataset_df, tmp_path: Path):
    input_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "outputs"
    valid_dataset_df.to_csv(input_path, index=False)

    profile = run_profile(raw_data_path=input_path, output_dir=output_dir)

    assert profile["rows"] == len(valid_dataset_df)
    assert (output_dir / "data_profile.json").exists()
    assert (output_dir / "data_profile.md").exists()
    saved = json.loads((output_dir / "data_profile.json").read_text(encoding="utf-8"))
    assert saved["rows"] == len(valid_dataset_df)


def test_profile_accepts_unlabeled_telemetry(valid_dataset_df, tmp_path):
    input_path = tmp_path / "incoming.csv"
    valid_dataset_df.drop(columns=["risk_level", "failure_risk"]).to_csv(input_path, index=False)

    profile = run_profile(input_path, output_dir=tmp_path / "profile")

    assert profile["target_counts"] == {}
    assert "Target column is absent; target balance is unavailable." in profile["warnings"]
    assert "No target labels available." in format_profile_markdown(profile)
