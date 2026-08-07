"""Load the single LocateAnything compiler configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - compiler environment owns PyYAML
    raise SystemExit(
        "PyYAML is required; install the host requirements with "
        "`python -m pip install -r compiler/requirements-host.txt`"
    ) from exc


class ConfigurationFileError(ValueError):
    """Raised when a compiler configuration file cannot be resolved."""


def load_config_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationFileError(f"config file not found: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationFileError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationFileError(f"config root must be a mapping: {path}")

    if "extends" in value:
        raise ConfigurationFileError(
            "config inheritance is not supported; keep the complete build in one file"
        )
    return value
