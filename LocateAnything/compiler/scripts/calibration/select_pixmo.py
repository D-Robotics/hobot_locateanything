#!/usr/bin/env python3
"""Expand the deterministic PixMo candidate pool without downloading images."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=240)
    args = parser.parse_args()
    if args.count < 80 or args.count % 80:
        raise ValueError("--count must be a positive multiple of 80")

    spec = importlib.util.spec_from_file_location("pixmo_selector", args.selector)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load selector: {args.selector}")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)

    scale = math.ceil(args.count / 80)
    selector.QUOTAS = {key: value * scale for key, value in selector.QUOTAS.items()}
    dataset = selector.load_from_disk(selector.DATASET_PATH)["train"]
    table = dataset.data.table
    mask, debug = selector.vectorized_valid_mask(table)
    sha_index = selector.build_sha_index(table, mask)
    selected = selector.select(sha_index, table)
    records = selector.build_records(selected, table, sha_index)
    if len(records) < args.count:
        raise RuntimeError(f"selector produced only {len(records)}/{args.count} candidates")
    records = records[: args.count]
    for rank, row in enumerate(records, start=1):
        row["candidate_id"] = f"pixmo-cand-{rank:03d}"
        row["selection_rank"] = rank

    shas = [row["expected_sha256"] for row in records]
    if len(set(shas)) != args.count:
        raise RuntimeError("expanded selection contains duplicate SHA256 values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"[pixmo] validity: {debug}")
    print(f"[pixmo] unique valid images: {len(sha_index)}")
    print(f"[pixmo] wrote {len(records)} candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
