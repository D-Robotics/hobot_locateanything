"""Load LocateAnything compiler configuration files with inheritance."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - compiler environment owns PyYAML
    raise SystemExit(
        "PyYAML is required; install the compiler package with "
        "`python -m pip install -e compiler`"
    ) from exc


class ConfigurationFileError(ValueError):
    """Raised when a compiler configuration file cannot be resolved."""


def merge_mappings(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(item) for item in (*chain, path))
        raise ConfigurationFileError(f"config inheritance cycle: {cycle}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationFileError(f"config file not found: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationFileError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationFileError(f"config root must be a mapping: {path}")

    parent = value.get("extends")
    override = {key: item for key, item in value.items() if key != "extends"}
    if parent is None:
        return override
    parent_path = Path(str(parent)).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return merge_mappings(
        load_config_file(parent_path, (*chain, path)),
        override,
    )
