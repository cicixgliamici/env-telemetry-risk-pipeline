import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_dataset(rows: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible environmental telemetry dataset for review and tests."""

    if rows < 20:
        raise ValueError("rows must be at least 20 to produce a useful class mix")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="h")

    humidity = np.clip(rng.normal(72, 14, rows), 25, 100)
    temperature = np.clip(rng.normal(24, 8, rows), -5, 45)
    rainfall = np.clip(rng.gamma(shape=1.4, scale=2.2, size=rows), 0, 35)
    pollution = np.clip(rng.normal(35, 18, rows), 0, 100)
    salinity = np.clip(rng.normal(30, 16, rows), 0, 100)
    pressure = np.clip(rng.normal(1012, 9, rows), 970, 1045)
    vibration = np.clip(rng.lognormal(mean=-1.0, sigma=0.65, size=rows), 0.02, 4.5)
    exposure = np.clip(rng.gamma(shape=2.2, scale=18, size=rows), 1, 180)

    risk_score = (
        0.020 * humidity
        + 0.030 * np.maximum(temperature - 28, 0)
        + 0.025 * rainfall
        + 0.018 * pollution
        + 0.018 * salinity
        + 0.070 * vibration
        + 0.004 * exposure
        + 0.015 * np.abs(pressure - 1013)
        + rng.normal(0, 0.35, rows)
    )
    threshold = np.quantile(risk_score, 0.62)
    risk_level = (risk_score >= threshold).astype(int)
    failure_risk = (risk_score >= np.quantile(risk_score, 0.72)).astype(int)

    return pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "humidity_pct": humidity.round(2),
            "temperature_c": temperature.round(2),
            "rainfall_mm": rainfall.round(2),
            "pollution_proxy": pollution.round(2),
            "salinity_proxy": salinity.round(2),
            "pressure_hpa": pressure.round(2),
            "vibration_mm_s": vibration.round(3),
            "exposure_hours": exposure.round(2),
            "risk_level": risk_level,
            "failure_risk": failure_risk,
        }
    )


def write_synthetic_dataset(output_path: str | Path, rows: int = 250, seed: int = 42) -> Path:
    """Write a synthetic dataset CSV and return the resolved output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_dataset(rows=rows, seed=seed).to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic environmental telemetry data")
    parser.add_argument("--output", default="data/raw/synthetic_dataset.csv")
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = write_synthetic_dataset(args.output, rows=args.rows, seed=args.seed)
    print(f"Wrote synthetic dataset to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
