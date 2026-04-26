# Architecture

## Overview

The project is organized as a lightweight Python package with a simple CLI entrypoint. The goal is to keep the runtime path easy to follow while still separating responsibilities clearly enough for maintenance and extension.

## Execution Flow

1. `src.cli` parses the user command.
2. `src.data_loader` loads the raw CSV and delegates schema checks to `src.validation`.
3. `src.preprocessing` normalizes and cleans the dataset.
4. `src.feature_engineering` adds derived model features.
5. `src.train` or `src.anomaly` runs the selected pipeline.
6. `src.evaluate`, `src.visualize`, and `src.utils` write metrics and artifacts.

## Module Responsibilities

- `src/config.py`: project paths and core constants
- `src/validation.py`: file-level and schema-level input validation
- `src/data_loader.py`: CSV loading after validation
- `src/preprocessing.py`: cleaning and train/test split preparation
- `src/feature_engineering.py`: derived telemetry features
- `src/train.py`: supervised model comparison and artifact generation
- `src/anomaly.py`: unsupervised anomaly detection pipeline
- `src/evaluate.py`: model metrics and comparable summaries
- `src/visualize.py`: chart generation
- `src/utils.py`: filesystem, logging, and timestamp helpers
- `src/cli.py`: operator-facing commands

## Production Notes

- Validation fails early before model execution.
- Outputs are written to predictable project folders.
- Logging is intentionally simple and console-first.
- Current configuration is code-based; environment-driven config would be the next step for deployment flexibility.
