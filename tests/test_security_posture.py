from src.config import (
    BASE_DIR,
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DATA_PATH,
)
from src.validation import validate_file_exists


def test_configured_artifact_paths_stay_inside_project():
    project_root = BASE_DIR.resolve()

    for path in [DATA_DIR, RAW_DATA_PATH, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
        assert path.resolve().is_relative_to(project_root)


def test_validate_file_exists_rejects_directories(tmp_path):
    directory = tmp_path / "dataset.csv"
    directory.mkdir()

    try:
        validate_file_exists(directory)
    except ValueError as exc:
        assert "Expected a file" in str(exc)
    else:
        raise AssertionError("validate_file_exists should reject directory inputs")
