#!/usr/bin/env python3
"""Monitor a durable LocateAnything JSONL job with progress, rate, and ETA."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path


def count_records(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def process_alive(pid_path: Path | None) -> bool | None:
    if pid_path is None or not pid_path.is_file():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-jsonl", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--exit-file", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--bar-width", type=int, default=40)
    args = parser.parse_args()
    if args.total <= 0 or args.interval <= 0 or args.bar_width <= 0:
        parser.error("total, interval, and bar-width must be positive")

    started = time.monotonic()
    initial = count_records(args.progress_jsonl)
    while True:
        current = count_records(args.progress_jsonl)
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = max(current - initial, 0) / elapsed
        remaining = max(args.total - current, 0)
        eta = remaining / rate if rate > 0 else None
        filled = min(args.bar_width, int(args.bar_width * current / args.total))
        bar = "#" * filled + "-" * (args.bar_width - filled)
        state = process_alive(args.pid_file)
        state_text = "alive" if state is True else "stopped" if state is False else "unknown"
        eta_text = f"{eta:.0f}s" if eta is not None else "unknown"
        print(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
            f"[{bar}] {current}/{args.total} ({100 * current / args.total:.1f}%) "
            f"rate={rate:.2f}/s eta={eta_text} process={state_text}",
            flush=True,
        )
        if current >= args.total:
            return 0
        if state is False:
            if args.exit_file and args.exit_file.is_file():
                print(args.exit_file.read_text(encoding="utf-8").strip(), flush=True)
            return 2
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
