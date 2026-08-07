"""The default LocateAnything Language graph catalog.

The compiler defaults to fused decode. Keeping the catalog in one source file
makes calibration, BC conversion, HBM linking, and runtime validation agree,
while users can extend or replace the catalog without editing every stage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


FUSED_DECODE_GRAPH_SET = "fused_decode"
BASE_LANGUAGE_GRAPHS = ("prefill", "decode", "decode_ar")
FUSED_PBD_GRAPHS = tuple(f"decode_pbd_q{q_len}" for q_len in range(7, 13))
FUSED_AR_GRAPHS = tuple(f"decode_ar_q{q_len}" for q_len in range(2, 6))
FUSED_LANGUAGE_GRAPHS = BASE_LANGUAGE_GRAPHS + FUSED_PBD_GRAPHS + FUSED_AR_GRAPHS

BASE_CALIBRATION_STAGES = ("prefill", "pbd_q6", "ar_q1")
FUSED_CALIBRATION_STAGES = (
    BASE_CALIBRATION_STAGES
    + tuple(f"pbd_q{q_len}" for q_len in range(7, 13))
    + tuple(f"ar_q{q_len}" for q_len in range(2, 6))
)


@dataclass(frozen=True)
class LanguageGraphSet:
    name: str
    graphs: tuple[str, ...]
    calibration_stages: tuple[str, ...]
    sequential_ar_q1_tokens: int = 1

    @property
    def uses_fused_decode(self) -> bool:
        return True

    def calibration_execution_counts(self, context_count: int) -> dict[str, int]:
        if context_count < 0:
            raise ValueError("context_count must be non-negative")
        counts = {stage: context_count for stage in self.calibration_stages}
        counts["ar_q1"] *= self.sequential_ar_q1_tokens
        return counts


DEFAULT_LANGUAGE_GRAPH_SET = LanguageGraphSet(
    name=FUSED_DECODE_GRAPH_SET,
    graphs=FUSED_LANGUAGE_GRAPHS,
    calibration_stages=FUSED_CALIBRATION_STAGES,
)


def language_graph_set() -> LanguageGraphSet:
    """Return the source-defined default Language graph catalog."""

    return DEFAULT_LANGUAGE_GRAPH_SET


def normalize_graph_set_metadata(name: object) -> str:
    """Normalize a stored catalog name before comparing calibration metadata."""

    return str(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the LocateAnything Language graph catalog")
    parser.add_argument("--field", choices=("graphs", "calibration-stages"), default="graphs")
    args = parser.parse_args(argv)
    values = (
        DEFAULT_LANGUAGE_GRAPH_SET.graphs
        if args.field == "graphs"
        else DEFAULT_LANGUAGE_GRAPH_SET.calibration_stages
    )
    print("\n".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
