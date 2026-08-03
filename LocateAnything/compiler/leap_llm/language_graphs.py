"""LocateAnything Language graph sets.

One graph-set selection controls calibration, BC export, HBM linking, and
runtime validation. Exact graph names are defined once and shared by every
stage that produces or consumes a Language model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


STANDARD_GRAPH_SET = "standard"
FUSED_DECODE_GRAPH_SET = "fused_decode"
_LEGACY_METADATA_ALIASES = {
    "legacy": STANDARD_GRAPH_SET,
    "fused": FUSED_DECODE_GRAPH_SET,
}

BASE_LANGUAGE_GRAPHS = ("prefill", "decode", "decode_ar")
FUSED_PBD_GRAPHS = tuple(f"decode_pbd_q{q_len}" for q_len in range(7, 13))
FUSED_AR_GRAPHS = tuple(f"decode_ar_q{q_len}" for q_len in range(2, 6))

BASE_CALIBRATION_STAGES = ("prefill", "pbd_q6", "ar_q1")
FUSED_PBD_CALIBRATION_STAGES = tuple(f"pbd_q{q_len}" for q_len in range(7, 13))
FUSED_AR_CALIBRATION_STAGES = tuple(f"ar_q{q_len}" for q_len in range(2, 6))


@dataclass(frozen=True)
class LanguageGraphSet:
    name: str
    graphs: tuple[str, ...]
    calibration_stages: tuple[str, ...]
    sequential_ar_q1_tokens: int

    @property
    def uses_fused_decode(self) -> bool:
        return self.name == FUSED_DECODE_GRAPH_SET

    def calibration_execution_counts(self, context_count: int) -> dict[str, int]:
        """Return the exact replay count required for each Language stage."""

        if context_count < 0:
            raise ValueError("context_count must be non-negative")
        counts = {stage: context_count for stage in self.calibration_stages}
        counts["ar_q1"] *= self.sequential_ar_q1_tokens
        return counts


LANGUAGE_GRAPH_SETS = {
    STANDARD_GRAPH_SET: LanguageGraphSet(
        name=STANDARD_GRAPH_SET,
        graphs=BASE_LANGUAGE_GRAPHS,
        calibration_stages=BASE_CALIBRATION_STAGES,
        # The standard path commits a six-token PBD window through repeated q1
        # calls. Replaying all six positions covers every partial-prefix depth.
        sequential_ar_q1_tokens=6,
    ),
    FUSED_DECODE_GRAPH_SET: LanguageGraphSet(
        name=FUSED_DECODE_GRAPH_SET,
        graphs=BASE_LANGUAGE_GRAPHS + FUSED_PBD_GRAPHS + FUSED_AR_GRAPHS,
        calibration_stages=(
            BASE_CALIBRATION_STAGES
            + FUSED_PBD_CALIBRATION_STAGES
            + FUSED_AR_CALIBRATION_STAGES
        ),
        sequential_ar_q1_tokens=1,
    ),
}

LANGUAGE_GRAPH_SET_NAMES = tuple(LANGUAGE_GRAPH_SETS)


def language_graph_set(name: str) -> LanguageGraphSet:
    try:
        return LANGUAGE_GRAPH_SETS[str(name)]
    except KeyError as exc:
        choices = ", ".join(LANGUAGE_GRAPH_SET_NAMES)
        raise ValueError(f"unknown Language graph set {name!r}; choose {choices}") from exc


def normalize_graph_set_metadata(name: object) -> str:
    """Normalize graph-set names stored by older calibration runs."""

    value = str(name)
    return _LEGACY_METADATA_ALIASES.get(value, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a LocateAnything Language graph set")
    parser.add_argument("graph_set", choices=LANGUAGE_GRAPH_SET_NAMES)
    parser.add_argument(
        "--field", choices=("graphs", "calibration-stages"), default="graphs"
    )
    args = parser.parse_args(argv)
    graph_set = language_graph_set(args.graph_set)
    values = (
        graph_set.graphs
        if args.field == "graphs"
        else graph_set.calibration_stages
    )
    print("\n".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
