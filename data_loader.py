import json
from pathlib import Path
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
