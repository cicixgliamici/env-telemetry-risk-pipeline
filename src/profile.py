from pathlib import Path

import pandas as pd

from src.config import NUMERIC_COLUMNS, OUTPUT_DIR, RAW_DATA_PATH, TARGET_COLUMN
from src.data_loader import load_csv
from src.utils import ensure_dir, save_json, utc_timestamp, write_text_file


def build_data_profile(df: pd.DataFrame) -> dict:
    """Build a compact data quality profile for a validated dataset."""

    numeric_columns = [column for column in df.columns if column in NUMERIC_COLUMNS]
    missing_values = {column: int(count) for column, count in df.isna().sum().items()}
    numeric_summary = {}
    for column in numeric_columns:
        series = df[column]
        numeric_summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
        }

    # Profiling incoming telemetry must not depend on labels becoming available.
    target_counts = (
        {
            str(label): int(count)
            for label, count in df[TARGET_COLUMN].value_counts().sort_index().items()
        }
        if TARGET_COLUMN in df.columns
        else {}
    )
    timestamps = pd.to_datetime(df["timestamp"], errors="raise", format="mixed")
    timestamp_min = timestamps.min()
    timestamp_max = timestamps.max()

    warnings = []
    if any(count > 0 for count in missing_values.values()):
        warnings.append("Dataset contains missing values.")
    if TARGET_COLUMN not in df.columns:
        warnings.append("Target column is absent; target balance is unavailable.")
    elif len(target_counts) < 2:
        warnings.append("Target column has fewer than two classes.")
    if len(df) < 50:
        warnings.append("Dataset has fewer than 50 rows; model metrics may be unstable.")

    return {
        "run_timestamp_utc": utc_timestamp(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
        "target_counts": target_counts,
        "timestamp_range": {
            "min": timestamp_min.isoformat(),
            "max": timestamp_max.isoformat(),
        },
        "warnings": warnings,
    }


def format_profile_markdown(profile: dict) -> str:
    """Render a data profile as a small Markdown report."""

    lines = [
        "# Data Profile",
        "",
        f"- Timestamp (UTC): {profile['run_timestamp_utc']}",
        f"- Rows: {profile['rows']}",
        f"- Columns: {len(profile['columns'])}",
        (
            f"- Timestamp range: {profile['timestamp_range']['min']} "
            f"to {profile['timestamp_range']['max']}"
        ),
        "",
        "## Target Balance",
        "",
    ]
    lines.extend(f"- Class {label}: {count}" for label, count in profile["target_counts"].items())
    if not profile["target_counts"]:
        lines.append("- No target labels available.")
    lines.extend(["", "## Missing Values", ""])
    lines.extend(f"- {column}: {count}" for column, count in profile["missing_values"].items())
    lines.extend(["", "## Numeric Summary", ""])
    lines.extend(
        (
            f"- {column}: min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
        for column, stats in profile["numeric_summary"].items()
    )
    lines.extend(["", "## Warnings", ""])
    if profile["warnings"]:
        lines.extend(f"- {warning}" for warning in profile["warnings"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def run_profile(
    raw_data_path: str | Path = RAW_DATA_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict:
    """Validate and profile a dataset, saving JSON and Markdown reports."""

    output_path = Path(output_dir)
    ensure_dir(output_path)
    df = load_csv(raw_data_path, purpose="prediction")
    profile = build_data_profile(df)
    save_json(profile, output_path / "data_profile.json")
    write_text_file(format_profile_markdown(profile), output_path / "data_profile.md")
    return profile
