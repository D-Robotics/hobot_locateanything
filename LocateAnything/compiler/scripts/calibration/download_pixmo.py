#!/usr/bin/env python3
"""Download and verify only the deterministic PixMo calibration candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def download_one(record: dict[str, Any], output_dir: Path, timeout: int) -> dict[str, Any]:
    expected = str(record["expected_sha256"])
    target = output_dir / str(record["local_filename"])
    result = {
        "candidate_id": record["candidate_id"],
        "expected_sha256": expected,
        "image_url": record["image_url"],
        "local_path": str(target.resolve()),
    }
    if target.is_file():
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() == expected:
            result.update(status="verified_existing", bytes=len(data))
            return result

    request = urllib.request.Request(
        str(record["image_url"]),
        headers={"User-Agent": "LocateAnything-calibration/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            result.update(status="sha256_mismatch", actual_sha256=actual, bytes=len(data))
            return result
        with Image.open(BytesIO(data)) as image:
            image.verify()
        output_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=output_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_name).replace(target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        result.update(status="downloaded", bytes=len(data))
        return result
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--minimum-valid", type=int, default=50)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if len(records) != len({row["expected_sha256"] for row in records}):
        raise ValueError("candidate manifest contains duplicate SHA256 values")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, args.output_dir, args.timeout): row
            for row in records
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[pixmo] {result['candidate_id']} {result['status']}", flush=True)

    rank = {row["candidate_id"]: row["selection_rank"] for row in records}
    results.sort(key=lambda row: rank[row["candidate_id"]])
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    valid = sum(row["status"] in {"downloaded", "verified_existing"} for row in results)
    print(f"[pixmo] verified valid images: {valid}/{len(records)}")
    return 0 if valid >= args.minimum_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
