"""Version pipeline outputs with DVC and push them to the GCS remote.

`dvc add` writes/updates a small .dvc pointer file (e.g. data/raw.dvc) that
Git tracks; the data itself goes to gs://<bucket>/dvc-store via `dvc push`.
Committing the updated pointer file to Git is left to a team member, so every
data version in Git history was reviewed by a person.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

from scripts.common.config import PIPELINE_ROOT
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("versioning")


class VersioningError(Exception):
    """Raised when a DVC command fails."""


def _run(cmd: List[str], cwd: Path) -> str:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VersioningError(
            f"{' '.join(cmd)} failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def version_data(target: str, push: bool = True) -> Dict[str, str]:
    """`dvc add` a data folder (relative to Data-Pipeline/) and optionally push it."""
    path = PIPELINE_ROOT / target
    if not path.exists():
        raise VersioningError(f"Nothing to version: {path} does not exist")
    _run(["dvc", "add", target], cwd=PIPELINE_ROOT)
    if push:
        _run(["dvc", "push", f"{target}.dvc"], cwd=PIPELINE_ROOT)
    pointer = PIPELINE_ROOT / f"{target}.dvc"
    logger.info("Versioned %s -> %s. Commit it with: git add %s && git commit",
                target, pointer.name, pointer.relative_to(PIPELINE_ROOT.parent))
    return {"target": target, "pointer": str(pointer), "pushed": str(push)}


def commit_pipeline_outputs(push: bool = True) -> Dict[str, str]:
    """Record the outputs of the dvc.yaml stages (run by Airflow) in dvc.lock and push.

    Airflow runs the same code as the dvc.yaml stages; `dvc commit` stores the
    resulting hashes without re-running anything, so `dvc repro` and Airflow
    share one lock file. Commit dvc.lock to Git to record the data version.
    """
    _run(["dvc", "commit", "--force"], cwd=PIPELINE_ROOT)
    if push:
        _run(["dvc", "push"], cwd=PIPELINE_ROOT)
    lock = PIPELINE_ROOT / "dvc.lock"
    logger.info("Pipeline outputs versioned in %s. Commit it with: git add %s && git commit",
                lock.name, lock.relative_to(PIPELINE_ROOT.parent))
    return {"lock_file": str(lock), "pushed": str(push)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Version a data folder with DVC")
    parser.add_argument("target", help="Folder relative to Data-Pipeline/, e.g. data/raw")
    parser.add_argument("--no-push", action="store_true", help="Only dvc add, do not push")
    args = parser.parse_args()
    configure_console_logging()
    version_data(args.target, push=not args.no_push)


if __name__ == "__main__":
    main()
