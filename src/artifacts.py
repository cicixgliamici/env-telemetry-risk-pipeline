"""Keep each execution's configuration, environment, and artifacts together."""

import platform
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

from src.utils import get_logger, save_json, utc_timestamp, write_text_file

LOGGER = get_logger("src.artifacts")


@dataclass(frozen=True)
class RunArtifacts:
    """Describe the directories owned by one execution."""

    run_id: str
    run_dir: Path

    @property
    def outputs(self) -> Path:
        return self.run_dir / "outputs"

    @property
    def models(self) -> Path:
        return self.run_dir / "models"

    @property
    def figures(self) -> Path:
        return self.run_dir / "figures"

    @property
    def processed(self) -> Path:
        return self.run_dir / "processed"

    def identity(self) -> dict:
        """Use absolute paths so CLI callers can locate the saved run directly."""

        return {"run_id": self.run_id, "run_dir": str(self.run_dir)}


def _save_environment(run_dir: Path) -> None:
    """Record installed versions without copying local paths or package credentials."""

    packages = dict(
        sorted(
            (package.metadata["Name"], package.version)
            for package in distributions()
            if package.metadata["Name"]
        )
    )
    save_json(
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        run_dir / "environment.json",
    )
    # Snapshot the environment, including transitive packages, rather than resolving new versions.
    # Install this repository separately: its local version may not exist on a package index.
    pins = [
        f"{name}=={version}"
        for name, version in packages.items()
        if name.lower().replace("_", "-") != "env-telemetry-risk-pipeline"
    ]
    write_text_file(
        "# Installed environment snapshot; use the Python/platform in environment.json.\n"
        + "\n".join(pins)
        + "\n",
        run_dir / "requirements.txt",
    )


@contextmanager
def artifact_run(output_dir: str | Path, operation: str, config: dict, *, compact: bool = False):
    """Create an exclusive run directory and record success or failure explicitly."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    # UUIDs avoid collisions between concurrent executions, even with identical clocks.
    # Nested review runs omit the redundant timestamp to stay within Windows path limits.
    identifier = uuid4().hex if compact else f"{timestamp}-{uuid4().hex}"
    run_id = f"{operation}-{identifier}"
    run_dir = Path(output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = RunArtifacts(run_id, run_dir)
    manifest = {
        **artifacts.identity(),
        "operation": operation,
        "started_at_utc": utc_timestamp(),
        "status": "running",
    }
    save_json(manifest, run_dir / "run.json")
    LOGGER.info("Starting %s run in %s", operation, run_dir)
    try:
        for directory in [
            artifacts.outputs,
            artifacts.models,
            artifacts.figures,
            artifacts.processed,
        ]:
            directory.mkdir()
        save_json(config, run_dir / "config.json")
        _save_environment(run_dir)
        yield artifacts
    except Exception:
        # Partial artifacts remain reviewable and must not look like a successful run.
        manifest.update(status="failed", finished_at_utc=utc_timestamp())
        save_json(manifest, run_dir / "run.json")
        raise
    else:
        manifest.update(status="completed", finished_at_utc=utc_timestamp())
        save_json(manifest, run_dir / "run.json")
