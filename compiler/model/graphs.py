"""The fixed LocateAnything Language graph catalog."""

from __future__ import annotations

import argparse


LANGUAGE_GRAPHS = (
    "prefill",
    "decode",
    "decode_ar",
    *tuple(f"decode_pbd_q{q_len}" for q_len in range(7, 13)),
    *tuple(f"decode_ar_q{q_len}" for q_len in range(2, 6)),
)
CALIBRATION_STAGES = (
    "prefill",
    "pbd_q6",
    "ar_q1",
    *tuple(f"pbd_q{q_len}" for q_len in range(7, 13)),
    *tuple(f"ar_q{q_len}" for q_len in range(2, 6)),
)


def calibration_execution_counts(context_count: int) -> dict[str, int]:
    if context_count < 0:
        raise ValueError("context_count must be non-negative")
    return {stage: context_count for stage in CALIBRATION_STAGES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the LocateAnything Language graph catalog")
    parser.add_argument("--field", choices=("graphs", "calibration-stages"), default="graphs")
    args = parser.parse_args(argv)
    values = (
        LANGUAGE_GRAPHS
        if args.field == "graphs"
        else CALIBRATION_STAGES
    )
    print("\n".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
