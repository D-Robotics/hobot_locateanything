#!/usr/bin/env python3
"""Create a small offline DatasetDict from verified PixMo candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid", type=int, default=50)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    candidates = {row["candidate_id"]: row for row in read_jsonl(args.manifest)}
    verified = {
        row["candidate_id"]: row
        for row in read_jsonl(args.results)
        if row["status"] in {"downloaded", "verified_existing"}
    }
    rows = []
    for candidate_id, result in verified.items():
        source = candidates[candidate_id]
        image_path = Path(result["local_path"])
        data = image_path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != source["expected_sha256"]:
            raise ValueError(f"SHA256 changed after download: {candidate_id}")
        rows.append({
            "candidate_id": candidate_id,
            "image_path": str(image_path.resolve()),
            "image_sha256": actual,
            "image_url": source["image_url"],
            "label": source["label"],
            "points": source["points"],
            "count": source["count"],
            "collection_method": source["collection_method"],
            "selection_rank": int(source["selection_rank"]),
        })
    rows.sort(key=lambda row: row["selection_rank"])
    if len(rows) < args.minimum_valid:
        raise RuntimeError(
            f"only {len(rows)} verified images; need at least {args.minimum_valid}"
        )
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(args.output_dir))
    print(f"[pixmo] materialized {len(rows)} verified candidates -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
