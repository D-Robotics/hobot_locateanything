"""Consistent progress reporting for long-running compiler commands."""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Generic, TextIO, TypeVar


T = TypeVar("T")
PROGRESS_MODES = ("auto", "bar", "log", "off")
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_BLUE = "\033[34m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"
_SEVERITY_PATTERN = re.compile(r"^(\[(?:INFO|WARN|ERROR)\])")
_STAGE_EVENT_PATTERN = re.compile(
    r"^\[(?:INFO|WARN|ERROR)\]\s+"
    r"\[(?P<scope>build\.[^]]+)\]\s+"
    r"\[(?P<current>\d+)/(?P<total>\d+)\]\s+"
    r"(?P<state>START|COMPLETE|FAILED)\s+"
    r"(?P<label>.*?)(?:\s+\|\s+.*)?$"
)
_PERCENT_PATTERN = re.compile(r"(?P<percent>\d{1,3})%\s*$")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_status_line(
    scope: str,
    current: int | None,
    total: int | None,
    state: str,
    label: str,
    *,
    details: str = "",
) -> str:
    """Return one ROS-style event line shared by the build entry points."""
    normalized_state = state.upper()
    if normalized_state in {"ERROR", "FAILED"}:
        severity = "ERROR"
    elif normalized_state == "WARN":
        severity = "WARN"
    else:
        severity = "INFO"
    if current is None or total is None:
        stage = "[-/-]"
    else:
        stage = f"[{current}/{total}]"
    severity_column = f"[{severity}]".ljust(8)
    scope_column = f"[{scope.lower()}]".ljust(20)
    stage_column = stage.ljust(8)
    state_column = normalized_state.ljust(10)
    suffix = f" | {details}" if details else ""
    return (
        f"{severity_column}{scope_column}{stage_column}"
        f"{state_column}{label}{suffix}"
    )


def _supports_color(stream: TextIO) -> bool:
    return (
        stream.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM", "") != "dumb"
    )


def colorize_console_line(line: str, *, stream: TextIO) -> str:
    """Color only the fixed prefix so copied messages remain readable."""
    if not _supports_color(stream):
        return line
    upper = line.upper()
    if " ERROR " in upper or " FAILED " in upper:
        color = _ANSI_BOLD + _ANSI_RED
    elif " COMPLETE " in upper:
        color = _ANSI_BOLD + _ANSI_GREEN
    elif " REUSE " in upper:
        color = _ANSI_BOLD + _ANSI_CYAN
    elif " WARN " in upper:
        color = _ANSI_BOLD + _ANSI_YELLOW
    elif " START " in upper or " PROGRESS " in upper:
        color = _ANSI_BOLD + _ANSI_BLUE
    elif " CONFIG " in upper or " LOG " in upper:
        color = _ANSI_DIM
    else:
        color = _ANSI_CYAN
    return _SEVERITY_PATTERN.sub(
        lambda match: f"{color}{match.group(1)}{_ANSI_RESET}", line, count=1
    )


def print_console_line(line: str, *, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    print(colorize_console_line(line, stream=output), file=output, flush=True)


def compact_build_line(component: str, raw_line: str) -> str | None:
    """Keep actionable build output while the complete stream stays in the log."""
    line = raw_line.strip()
    if not line:
        return None
    component_name = component.upper()
    scope = f"build.{component}"
    if re.match(
        rf"^\[(?:INFO|WARN|ERROR)\]\s+\[{re.escape(scope)}\]", line
    ):
        return line
    component_prefix = f"[{component_name}]"
    if line.startswith(component_prefix):
        remainder = line[len(component_prefix):].strip()
        state, _, label = remainder.partition(" ")
        return format_status_line(
            scope, None, None, state, label.strip()
        )
    translations = {
        "[PASS]": "COMPLETE",
        "[RESUME]": "REUSE",
        "[REUSE]": "REUSE",
        "[STALE]": "WARN",
    }
    for prefix, state in translations.items():
        if line.startswith(prefix):
            return format_status_line(
                scope, None, None, state, line[len(prefix):].strip()
            )
    legacy_prefix = f"[build:{component}]"
    if line.startswith(legacy_prefix):
        message = line[len(legacy_prefix):].strip()
        lowered = message.lower()
        if "failed" in lowered:
            state = "ERROR"
        elif "completed" in lowered:
            state = "COMPLETE"
        elif "resume" in lowered or "reuse" in lowered:
            state = "REUSE"
        else:
            state = "INFO"
        return format_status_line(scope, None, None, state, message)
    return None


def filter_build_output(component: str, *, source: TextIO, stream: TextIO) -> None:
    active_stage: tuple[str, int, int, str, float] | None = None
    last_progress = -1
    progress_open = False
    buffer = ""

    def close_progress() -> None:
        nonlocal progress_open
        if progress_open:
            stream.write("\n")
            stream.flush()
            progress_open = False

    def handle_fragment(fragment: str) -> None:
        nonlocal active_stage, last_progress, progress_open
        if not fragment:
            return
        percent_match = _PERCENT_PATTERN.search(fragment)
        if percent_match is not None and active_stage is not None:
            percent = min(100, int(percent_match.group("percent")))
            if percent == last_progress:
                return
            last_progress = percent
            scope, current, total, label, started = active_stage
            progress_line = format_status_line(
                scope,
                current,
                total,
                "PROGRESS",
                label,
                details=(
                    f"progress={percent}% elapsed="
                    f"{_format_duration(time.monotonic() - started)}"
                ),
            )
            rendered = colorize_console_line(progress_line, stream=stream)
            if stream.isatty():
                stream.write("\r" + rendered + " " * 8)
                stream.flush()
                progress_open = percent < 100
                if percent == 100:
                    stream.write("\n")
                    stream.flush()
            elif percent in {0, 100} or percent % 10 == 0:
                print(rendered, file=stream, flush=True)
            return

        line = compact_build_line(component, fragment)
        if line is None:
            return
        close_progress()
        event = _STAGE_EVENT_PATTERN.match(line)
        if event is not None:
            state = event.group("state")
            if state == "START":
                active_stage = (
                    event.group("scope"),
                    int(event.group("current")),
                    int(event.group("total")),
                    event.group("label"),
                    time.monotonic(),
                )
                last_progress = -1
            elif active_stage is not None:
                active_stage = None
                last_progress = -1
        print_console_line(line, stream=stream)

    while True:
        chunk = source.read(1)
        if not chunk:
            break
        buffer += chunk
        while True:
            positions = [
                position
                for position in (buffer.find("\n"), buffer.find("\r"))
                if position >= 0
            ]
            if not positions:
                break
            boundary = min(positions)
            handle_fragment(buffer[:boundary])
            buffer = buffer[boundary + 1:]
    handle_fragment(buffer)
    close_progress()


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
        self.scope = f"build.{description.removesuffix(' build').lower()}"
        self.stream = stream or sys.stderr
        self.mode = resolve_progress_mode(mode, stream=self.stream)
        self.current = 0
        self.started = time.monotonic()

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        self.current += 1
        if self.current > self.total:
            raise RuntimeError(
                f"{self.description} reported more than {self.total} stages"
            )
        stage_started = time.monotonic()
        if self.mode != "off":
            print_console_line(
                format_status_line(
                    self.scope, self.current, self.total, "START", label
                ),
                stream=self.stream,
            )
        try:
            yield
        except BaseException:
            if self.mode != "off":
                elapsed = _format_duration(time.monotonic() - stage_started)
                print_console_line(
                    format_status_line(
                        self.scope,
                        self.current,
                        self.total,
                        "FAILED",
                        label,
                        details=f"elapsed={elapsed}",
                    ),
                    stream=self.stream,
                )
            raise
        if self.mode != "off":
            elapsed = _format_duration(time.monotonic() - stage_started)
            total_elapsed = _format_duration(time.monotonic() - self.started)
            print_console_line(
                format_status_line(
                    self.scope,
                    self.current,
                    self.total,
                    "COMPLETE",
                    label,
                    details=f"elapsed={elapsed} total={total_elapsed}",
                ),
                stream=self.stream,
            )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Filter detailed compiler output")
    parser.add_argument("--component", choices=("vision", "language"), required=True)
    args = parser.parse_args()
    filter_build_output(args.component, source=sys.stdin, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
