"""Consistent progress reporting for long-running compiler commands."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Generic, TextIO, TypeVar


T = TypeVar("T")
PROGRESS_MODES = ("auto", "bar", "log", "off")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def resolve_progress_mode(mode: str | None = None, *, stream: TextIO | None = None) -> str:
    selected = (mode or os.environ.get("LA_PROGRESS", "auto")).strip().lower()
    if selected not in PROGRESS_MODES:
        choices = ", ".join(PROGRESS_MODES)
        raise ValueError(f"invalid progress mode {selected!r}; choose one of: {choices}")
    if selected == "auto":
        output = stream or sys.stderr
        return "bar" if output.isatty() else "log"
    return selected


class ProgressReporter(Generic[T]):
    """Iterable progress reporter with terminal-bar, log, and quiet modes."""

    def __init__(
        self,
        items: Sequence[T],
        description: str,
        *,
        unit: str = "item",
        mode: str | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.items = items
        self.description = description
        self.unit = unit
        self.stream = stream or sys.stderr
        self.mode = resolve_progress_mode(mode, stream=self.stream)
        self.total = len(items)
        self.postfix: dict[str, Any] = {}
        self._bar: Any = None

        if self.mode == "bar":
            try:
                from tqdm import tqdm

                self._bar = tqdm(
                    items,
                    desc=description,
                    unit=unit,
                    dynamic_ncols=True,
                    file=self.stream,
                )
            except ImportError:
                self.mode = "log"

    def __iter__(self) -> Iterator[T]:
        if self._bar is not None:
            yield from self._bar
            return
        if self.mode == "off":
            yield from self.items
            return

        started = time.monotonic()
        interval = max(1, self.total // 20)
        for index, item in enumerate(self.items, start=1):
            yield item
            if index != self.total and index % interval:
                continue
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = index / elapsed
            remaining = (self.total - index) / rate if rate else 0.0
            percent = index / self.total * 100.0 if self.total else 100.0
            details = " ".join(f"{key}={value}" for key, value in self.postfix.items())
            suffix = f" {details}" if details else ""
            print(
                f"[progress] {self.description}: {index}/{self.total} {self.unit} "
                f"({percent:.1f}%) elapsed={_format_duration(elapsed)} "
                f"rate={rate:.2f}/{self.unit}/s eta={_format_duration(remaining)}{suffix}",
                file=self.stream,
                flush=True,
            )

    def set_postfix(self, values: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.postfix = dict(values or {})
        self.postfix.update(kwargs)
        if self._bar is not None:
            self._bar.set_postfix(self.postfix)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def track(
    items: Sequence[T],
    description: str,
    *,
    unit: str = "item",
    mode: str | None = None,
    stream: TextIO | None = None,
) -> ProgressReporter[T]:
    return ProgressReporter(
        items,
        description,
        unit=unit,
        mode=mode,
        stream=stream,
    )


class StageProgress:
    """Report overall progress across a small number of long build stages."""

    def __init__(
        self,
        total: int,
        description: str,
        *,
        mode: str | None = None,
        stream: TextIO | None = None,
    ) -> None:
        if total <= 0:
            raise ValueError("stage total must be positive")
        self.total = total
        self.description = description
        self.stream = stream or sys.stderr
        self.mode = resolve_progress_mode(mode, stream=self.stream)
        self.current = 0
        self.started = time.monotonic()

    def _prefix(self) -> str:
        percent = self.current / self.total * 100.0
        filled = int(self.current / self.total * 20)
        bar = "=" * filled + "-" * (20 - filled)
        return (
            f"[{bar}] {self.description} "
            f"{self.current}/{self.total} {percent:5.1f}%"
        )

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        self.current += 1
        if self.current > self.total:
            raise RuntimeError(
                f"{self.description} reported more than {self.total} stages"
            )
        stage_started = time.monotonic()
        if self.mode != "off":
            print(f"{self._prefix()} START {label}", file=self.stream, flush=True)
        try:
            yield
        except BaseException:
            if self.mode != "off":
                elapsed = _format_duration(time.monotonic() - stage_started)
                print(
                    f"{self._prefix()} FAILED {label} elapsed={elapsed}",
                    file=self.stream,
                    flush=True,
                )
            raise
        if self.mode != "off":
            elapsed = _format_duration(time.monotonic() - stage_started)
            total_elapsed = _format_duration(time.monotonic() - self.started)
            print(
                f"{self._prefix()} DONE {label} elapsed={elapsed} "
                f"total_elapsed={total_elapsed}",
                file=self.stream,
                flush=True,
            )
