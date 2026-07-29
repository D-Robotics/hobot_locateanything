#!/usr/bin/env python3
"""Run a frozen LocateAnything manifest through the S600 demo one sample at a time."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


END_RE = re.compile(r"^\[callback\] END:\s*(.*)$", re.MULTILINE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def final_response(log_text: str) -> str | None:
    matches = END_RE.findall(log_text)
    return matches[-1].strip() if matches else None


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", default="s600_hybrid")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    if not args.demo.is_file() or not os.access(args.demo, os.X_OK):
        raise SystemExit(f"demo is missing or not executable: {args.demo}")
    if not args.config.is_file():
        raise SystemExit(f"config is missing: {args.config}")
    if args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    rows = read_jsonl(args.manifest_jsonl)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "logs"
    logs.mkdir()
    predictions = args.output_dir / "predictions.jsonl"
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        image = Path(str(row["image"]))
        if not image.is_absolute():
            image = args.bundle_dir / image
        command = [str(args.demo), "-c", str(args.config), "-i", str(image), "-p", str(row["prompt"])]
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
            exit_code = completed.returncode
            output = completed.stdout.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            output = (exc.stdout or b"").decode("utf-8", errors="replace")
        answer = final_response(output)
        log_path = logs / f"{index:04d}-{row['bundle_id']}.log"
        log_path.write_text(output, encoding="utf-8")
        success = not timed_out and exit_code == 0 and answer is not None
        counts["success" if success else "failed"] += 1
        append_jsonl(predictions, {
            "schema_version": 1,
            "bundle_id": row["bundle_id"],
            "task": row.get("task"),
            "mode": args.mode,
            "answer": answer,
            "success": success,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "image_sha256": row.get("image_sha256"),
            "log_path": str(log_path),
        })
        print(f"[heldout] {index + 1}/{len(rows)} {row['bundle_id']} success={success}", flush=True)
    summary = {
        "schema_version": 1,
        "records": len(rows),
        "success_count": counts["success"],
        "failed_count": counts["failed"],
        "prediction_coverage": counts["success"] / len(rows) if rows else None,
        "mode": args.mode,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
