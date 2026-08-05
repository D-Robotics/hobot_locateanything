"""Low-overhead S600 resource sampling and fixed terminal header."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from console import (
    BOLD,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    fit_terminal_line,
)


S600_BPU_RATIO_RE = re.compile(
    r"\bbpu(?P<core>[0-3])(?:\s+ratio)?\s*[:=]\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
S600_DIRECT_BPU_TEMPERATURE_RE = re.compile(
    r"\bpvt_bpu_[^:]+:\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?:\(C\)|C\b)",
    re.IGNORECASE,
)
S600_SECTION_VALUE_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9_]+)\s*:\s*(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>m?C)\b",
    re.IGNORECASE,
)


def parse_s600_resource_status(text: str) -> tuple[list[float | None], float | None]:
    """Parse four BPU ratios and the hottest BPU temperature."""
    bpu: list[float | None] = [None] * 4
    for match in S600_BPU_RATIO_RE.finditer(text):
        bpu[int(match.group("core"))] = float(match.group("value"))

    temperatures = [
        float(match.group("value"))
        for match in S600_DIRECT_BPU_TEMPERATURE_RE.finditer(text)
    ]
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.endswith("-->"):
            section = line[:-3].strip().lower()
            continue
        if section != "temperature":
            continue
        match = S600_SECTION_VALUE_RE.match(line)
        if match is None or "bpu" not in match.group("label").lower():
            continue
        value = float(match.group("value"))
        if match.group("unit").lower() == "mc":
            value /= 1000.0
        temperatures.append(value)
    return bpu, max(temperatures) if temperatures else None

@dataclass(frozen=True)
class ResourceSummary:
    sample_count: int
    elapsed_seconds: float
    bpu_average_percent: tuple[float | None, ...]
    bpu_peak_percent: tuple[float | None, ...]
    cpu_average_percent: float | None
    cpu_peak_percent: float | None
    memory_average_percent: float | None
    memory_peak_percent: float | None
    bpu_temperature_peak_celsius: float | None

    def as_dict(self) -> dict[str, object]:
        def rounded(value: float | None) -> float | None:
            return round(value, 3) if value is not None else None

        return {
            "sample_count": self.sample_count,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "bpu_average_percent": [rounded(value) for value in self.bpu_average_percent],
            "bpu_peak_percent": [rounded(value) for value in self.bpu_peak_percent],
            "cpu_average_percent": rounded(self.cpu_average_percent),
            "cpu_peak_percent": rounded(self.cpu_peak_percent),
            "memory_average_percent": rounded(self.memory_average_percent),
            "memory_peak_percent": rounded(self.memory_peak_percent),
            "bpu_temperature_peak_celsius": rounded(self.bpu_temperature_peak_celsius),
        }


class ResourceMonitor:
    """Collect request telemetry and render a low-overhead fixed terminal layout."""

    def __init__(
        self,
        *,
        visible: bool = True,
        interval_seconds: float = 0.25,
    ) -> None:
        if interval_seconds < 0.1:
            raise ValueError("telemetry interval must be at least 0.1 seconds")
        size = shutil.get_terminal_size((120, 32))
        self.visible = (
            visible
            and sys.stdout.isatty()
            and os.environ.get("TERM") != "dumb"
        )
        self.interval_seconds = interval_seconds
        self._rows = size.lines
        self._columns = size.columns
        self._header_lines: tuple[str, ...] = ()
        self._layout_active = False
        self._stop = threading.Event()
        self._active = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._previous_cpu: tuple[int, int] | None = None
        self._bpu_paths = tuple(
            Path(f"/sys/devices/system/bpu/bpu{core}/ratio") for core in range(4)
        )
        self._temperature_paths = self._discover_temperature_paths()
        self._request_started = 0.0
        self._request_elapsed = 0.0
        self._reset_aggregates()

    @staticmethod
    def _discover_temperature_paths() -> tuple[Path, ...]:
        result: list[Path] = []
        for type_path in Path("/sys/devices/virtual/thermal").glob("thermal_zone*/type"):
            try:
                if "bpu" in type_path.read_text(encoding="ascii").strip().lower():
                    result.append(type_path.with_name("temp"))
            except OSError:
                continue
        return tuple(result)

    def _reset_aggregates(self) -> None:
        self._sample_count = 0
        self._bpu_sum = [0.0] * 4
        self._bpu_count = [0] * 4
        self._bpu_peak: list[float | None] = [None] * 4
        self._cpu_sum = 0.0
        self._cpu_count = 0
        self._cpu_peak: float | None = None
        self._memory_sum = 0.0
        self._memory_count = 0
        self._memory_peak: float | None = None
        self._temperature_peak: float | None = None

    @staticmethod
    def _read_system_cpu() -> tuple[int, int] | None:
        try:
            fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
            values = [int(value) for value in fields[1:]]
        except (OSError, ValueError, IndexError):
            return None
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return total, idle

    @staticmethod
    def _read_memory_percent() -> float | None:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                key, separator, tail = line.partition(":")
                if separator and key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(tail.split()[0])
            return 100.0 * (1.0 - values["MemAvailable"] / values["MemTotal"])
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

    def _read_bpu_direct(self) -> tuple[list[float | None], float | None]:
        ratios: list[float | None] = []
        for path in self._bpu_paths:
            try:
                ratios.append(float(path.read_text(encoding="ascii").strip()))
            except (OSError, ValueError):
                ratios.append(None)
        temperatures: list[float] = []
        for path in self._temperature_paths:
            try:
                temperatures.append(float(path.read_text(encoding="ascii").strip()) / 1000.0)
            except (OSError, ValueError):
                continue
        return ratios, max(temperatures) if temperatures else None

    @staticmethod
    def _read_vendor_fallback() -> tuple[list[float | None], float | None]:
        command = Path("/usr/hobot/bin/hrut_somstatus")
        if not command.is_file():
            return [None] * 4, None
        try:
            completed = subprocess.run(
                [str(command)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=1.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return [None] * 4, None
        if completed.returncode != 0:
            return [None] * 4, None
        return parse_s600_resource_status(completed.stdout)

    @staticmethod
    def _average(total: float, count: int) -> float | None:
        return total / count if count else None

    def _sample(self) -> None:
        with self._sample_lock:
            self._sample_once()

    def _sample_once(self) -> None:
        cpu_now = self._read_system_cpu()
        cpu_percent = None
        if cpu_now is not None and self._previous_cpu is not None:
            total_delta = cpu_now[0] - self._previous_cpu[0]
            idle_delta = cpu_now[1] - self._previous_cpu[1]
            if total_delta > 0:
                cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)
        self._previous_cpu = cpu_now

        bpu, temperature = self._read_bpu_direct()
        if all(value is None for value in bpu) and temperature is None:
            bpu, temperature = self._read_vendor_fallback()
        memory = self._read_memory_percent()
        with self._lock:
            if not self._active.is_set():
                return
            self._sample_count += 1
            for core, value in enumerate(bpu):
                if value is None:
                    continue
                self._bpu_sum[core] += value
                self._bpu_count[core] += 1
                previous = self._bpu_peak[core]
                self._bpu_peak[core] = value if previous is None else max(previous, value)
            if cpu_percent is not None:
                self._cpu_sum += cpu_percent
                self._cpu_count += 1
                self._cpu_peak = (
                    cpu_percent if self._cpu_peak is None else max(self._cpu_peak, cpu_percent)
                )
            if memory is not None:
                self._memory_sum += memory
                self._memory_count += 1
                self._memory_peak = (
                    memory if self._memory_peak is None else max(self._memory_peak, memory)
                )
            if temperature is not None:
                self._temperature_peak = (
                    temperature
                    if self._temperature_peak is None
                    else max(self._temperature_peak, temperature)
                )

    def begin_request(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._reset_aggregates()
            self._request_started = now
            self._request_elapsed = 0.0
        self._previous_cpu = self._read_system_cpu()
        self._active.set()
        self._wake.set()

    def end_request(self) -> None:
        if not self._active.is_set():
            return
        self._sample()
        self._active.clear()
        with self._lock:
            self._request_elapsed = time.monotonic() - self._request_started
        self._wake.set()

    @contextmanager
    def request_scope(self) -> Iterator["ResourceMonitor"]:
        self.begin_request()
        try:
            yield self
        finally:
            self.end_request()

    def summary(self, *, sample_now: bool = False) -> ResourceSummary:
        if sample_now and self._active.is_set():
            self._sample()
        with self._lock:
            elapsed = (
                time.monotonic() - self._request_started
                if self._active.is_set() and self._request_started
                else self._request_elapsed
            )
            return ResourceSummary(
                sample_count=self._sample_count,
                elapsed_seconds=elapsed,
                bpu_average_percent=tuple(
                    self._average(self._bpu_sum[index], self._bpu_count[index])
                    for index in range(4)
                ),
                bpu_peak_percent=tuple(self._bpu_peak),
                cpu_average_percent=self._average(self._cpu_sum, self._cpu_count),
                cpu_peak_percent=self._cpu_peak,
                memory_average_percent=self._average(
                    self._memory_sum, self._memory_count
                ),
                memory_peak_percent=self._memory_peak,
                bpu_temperature_peak_celsius=self._temperature_peak,
            )

    def _run(self) -> None:
        next_sample = 0.0
        while not self._stop.is_set():
            active = self._active.is_set()
            now = time.monotonic()
            if active and now >= next_sample:
                self._sample()
                next_sample = now + self.interval_seconds
            elif not active:
                next_sample = 0.0
            timeout = (
                max(0.0, next_sample - time.monotonic())
                if active
                else None
            )
            self._wake.wait(timeout)
            self._wake.clear()

    def start(self, header_lines: list[str] | tuple[str, ...] = ()) -> None:
        if self._thread is not None:
            return
        self._header_lines = tuple(header_lines)
        size = shutil.get_terminal_size((120, 32))
        self._rows = size.lines
        self._columns = size.columns
        content_top = len(self._header_lines) + 1
        content_bottom = self._rows
        self._layout_active = (
            self.visible and content_bottom - content_top + 1 >= 4
        )
        if self._layout_active:
            sys.stdout.flush()
            header = "\r\n".join(
                fit_terminal_line(line, self._columns) for line in self._header_lines
            )
            sys.stdout.write(
                f"\033[2J\033[H{header}\033[{content_top};{content_bottom}r"
                f"\033[{content_top};1H"
            )
            sys.stdout.flush()
        else:
            self.visible = False
            if self._header_lines:
                print("\n".join(self._header_lines), flush=True)
        self._thread = threading.Thread(
            target=self._run, name="s600-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._layout_active:
            sys.stdout.write("\033[r")
            sys.stdout.flush()
            self._layout_active = False


def _summary_value(number: float | None, suffix: str = "%") -> str:
    if number is None:
        return f"{DIM}--{RESET}"
    threshold = 70.0 if suffix == " C" else 55.0
    warning = 85.0
    color = GREEN if number < threshold else YELLOW if number < warning else RED
    return f"{color}{number:.1f}{suffix}{RESET}"


def format_resource_values(values: tuple[float | None, ...]) -> str:
    return "/".join(_summary_value(value) for value in values)


def resource_summary_lines(summary: ResourceSummary) -> list[str]:
    return [
        f"  {BOLD}BPU{RESET}     avg "
        f"{format_resource_values(summary.bpu_average_percent)}  "
        f"peak {format_resource_values(summary.bpu_peak_percent)}",
        f"  {BOLD}System{RESET}  CPU avg {_summary_value(summary.cpu_average_percent)} "
        f"peak {_summary_value(summary.cpu_peak_percent)}  |  "
        f"Memory avg {_summary_value(summary.memory_average_percent)} "
        f"peak {_summary_value(summary.memory_peak_percent)}  |  "
        f"Temp {_summary_value(summary.bpu_temperature_peak_celsius, ' C')}",
    ]
