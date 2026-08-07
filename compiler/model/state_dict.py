"""Strict checkpoint loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def load_state_dict_strict(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    component: str,
) -> None:
    """Load a remapped checkpoint and reject every parameter mismatch."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise RuntimeError(
            f"{component} checkpoint contract mismatch: " + "; ".join(details)
        )
