"""Load pipeline parameters and resolve project paths."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Data-Pipeline/ directory. Can be overridden (e.g. inside the Airflow container).
PIPELINE_ROOT = Path(
    os.environ.get("SKINSORT_PIPELINE_ROOT", Path(__file__).resolve().parents[2])
)
DEFAULT_PARAMS_PATH = PIPELINE_ROOT / "params.yaml"


class ConfigError(Exception):
    """Raised when params.yaml is missing or malformed."""


def load_params(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read params.yaml and return it as a dict."""
    path = Path(path) if path else DEFAULT_PARAMS_PATH
    if not path.is_file():
        raise ConfigError(f"Parameter file not found: {path}")
    with path.open() as fh:
        params = yaml.safe_load(fh)
    if not isinstance(params, dict):
        raise ConfigError(f"Parameter file is empty or not a mapping: {path}")
    return params


def load_yaml(path: Path) -> Dict[str, Any]:
    """Read any YAML config file (e.g. the urgency mapping) as a dict."""
    return load_params(path)


def resolve_path(params: Dict[str, Any], key: str) -> Path:
    """Return an absolute path for an entry under params['paths']."""
    try:
        relative = params["paths"][key]
    except KeyError as exc:
        raise ConfigError(f"Missing paths.{key} in params.yaml") from exc
    path = Path(relative)
    return path if path.is_absolute() else PIPELINE_ROOT / path


def raw_dataset_dir(params: Dict[str, Any], dataset: str) -> Path:
    """Directory where a dataset's raw download is stored."""
    return resolve_path(params, "raw") / dataset
