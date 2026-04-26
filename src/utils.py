from datetime import datetime, UTC
import json
import logging
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory tree if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    """Persist a JSON payload with stable UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def write_text_file(contents: str, path: Path) -> None:
    """Write a UTF-8 text artifact, creating parent folders when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def get_logger(name: str) -> logging.Logger:
    """Return a process-local console logger configured for this project."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def utc_timestamp() -> str:
    """Return a filesystem- and JSON-friendly UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()
