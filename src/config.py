from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "raw_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COLUMN = "risk_level"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TELEMETRY_COLUMNS = {
    "humidity_pct",
    "temperature_c",
    "rainfall_mm",
    "pollution_proxy",
    "salinity_proxy",
    "pressure_hpa",
    "vibration_mm_s",
    "exposure_hours",
}

LABEL_COLUMNS = {TARGET_COLUMN, "failure_risk"}
REQUIRED_COLUMNS = TELEMETRY_COLUMNS | {"timestamp", TARGET_COLUMN}
NUMERIC_COLUMNS = TELEMETRY_COLUMNS | LABEL_COLUMNS

# Bounds reject impossible measurements without clipping valid extreme observations.
# Proxy units are unspecified, so only non-negativity is enforced for those inputs.
NUMERIC_RANGES = {
    "humidity_pct": (0, 100),
    "temperature_c": (-273.15, None),
    "rainfall_mm": (0, None),
    "pollution_proxy": (0, None),
    "salinity_proxy": (0, None),
    "vibration_mm_s": (0, None),
    "exposure_hours": (0, None),
}
