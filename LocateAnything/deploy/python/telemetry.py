"""Low-overhead S600 resource sampling and live terminal status."""

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

from console import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, format_duration


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
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


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


def _fit_terminal_line(value: str, columns: int) -> str:
    plain = ANSI_ESCAPE_RE.sub("", value)
    return value if len(plain) < columns else plain[: max(1, columns - 1)]


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
    """Collect request telemetry and optionally render a fixed two-line footer."""

    def __init__(
        self,
        *,
        visible: bool = True,
        interval_seconds: float = 0.25,
    ) -> None:
        if interval_seconds < 0.1:
            raise ValueError("telemetry interval must be at least 0.1 seconds")
        rows = shutil.get_terminal_size((120, 32)).lines
        self.visible = (
            visible
            and rows >= 6
            and sys.stdout.isatty()
            and os.environ.get("TERM") != "dumb"
        )
        self.interval_seconds = interval_seconds
        self._rows = rows
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
        self._current_bpu: list[float | None] = [None] * 4
        self._current_cpu: float | None = None
        self._current_memory: float | None = None
        self._current_temperature: float | None = None
        self._stage_index = 0
        self._stage_total = 0
        self._stage_name = "Idle"
        self._stage_started = 0.0
        self._request_started = 0.0
        self._request_elapsed = 0.0
        self._token_count = 0
        self._completed_request = False
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
            self._current_bpu = bpu
            self._current_cpu = cpu_percent
            self._current_memory = memory
            self._current_temperature = temperature
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
            self._stage_index = 0
            self._stage_total = 0
            self._stage_name = "Starting"
            self._stage_started = now
            self._request_started = now
            self._request_elapsed = 0.0
            self._token_count = 0
            self._completed_request = False
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
            self._completed_request = True
            self._stage_index = self._stage_total
            self._stage_name = "Complete"
        self._wake.set()

    @contextmanager
    def request_scope(self) -> Iterator["ResourceMonitor"]:
        self.begin_request()
        try:
            yield self
        finally:
            self.end_request()

    def set_stage(self, index: int, total: int, name: str) -> None:
        with self._lock:
            self._stage_index = index
            self._stage_total = total
            self._stage_name = name
            self._stage_started = time.monotonic()
            self._token_count = 0
        self._wake.set()

    def observe_token(self) -> None:
        with self._lock:
            self._token_count += 1
        self._wake.set()

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

    @staticmethod
    def _plain_percent(value: float | None) -> str:
        return "--" if value is None else f"{value:.0f}%"

    @staticmethod
    def _colored_percent(value: float | None) -> str:
        if value is None:
            return f"{DIM}--{RESET}"
        color = GREEN if value < 55.0 else YELLOW if value < 85.0 else RED
        return f"{color}{value:3.0f}%{RESET}"

    @staticmethod
    def _progress_bar(index: int, total: int, complete: bool, width: int = 14) -> str:
        if total <= 0:
            return "[" + "-" * width + "]"
        fraction = 1.0 if complete else max(0.0, min(1.0, (index - 1) / total))
        filled = min(width, int(fraction * width))
        if complete:
            body = "=" * width
        elif filled < width:
            body = "=" * filled + ">" + "-" * (width - filled - 1)
        else:
            body = "=" * width
        return f"[{body}]"

    def _render(self) -> tuple[str, str]:
        with self._lock:
            now = time.monotonic()
            active = self._active.is_set()
            elapsed = (
                now - self._request_started
                if active and self._request_started
                else self._request_elapsed
            )
            stage_elapsed = now - self._stage_started if self._stage_started else 0.0
            rate = self._token_count / stage_elapsed if stage_elapsed > 0 else 0.0
            bar = self._progress_bar(
                self._stage_index, self._stage_total, self._completed_request
            )
            stage = (
                f"{BOLD}{CYAN}{bar}{RESET} "
                f"{self._stage_index}/{self._stage_total} {self._stage_name}"
            )
            if (
                active
                and self._stage_name.lower().startswith("language")
                and self._token_count
            ):
                stage += f"  {self._token_count} tokens  {rate:.1f} tokens/s"
            stage += f"  {DIM}{format_duration(elapsed)}{RESET}"

            bpu = " ".join(
                f"{core}:{self._colored_percent(value)}"
                for core, value in enumerate(self._current_bpu)
            )
            temperature = (
                "--" if self._current_temperature is None else f"{self._current_temperature:.1f} C"
            )
            resources = (
                f"BPU {bpu}  |  CPU {self._colored_percent(self._current_cpu)}  |  "
                f"Memory {self._colored_percent(self._current_memory)}  |  Temp {temperature}"
            )
            return stage, resources

    def _draw(self) -> None:
        if not self.visible:
            return
        stage, resources = self._render()
        columns = shutil.get_terminal_size((120, 32)).columns
        if columns < 88:
            with self._lock:
                bpu = "/".join(self._plain_percent(value) for value in self._current_bpu)
                resources = (
                    f"BPU {bpu} | CPU {self._plain_percent(self._current_cpu)} | "
                    f"MEM {self._plain_percent(self._current_memory)}"
                )
        stage = _fit_terminal_line(stage, columns)
        resources = _fit_terminal_line(resources, columns)
        sys.stdout.write(
            f"\0337\033[{self._rows - 1};1H\033[2K{stage}"
            f"\033[{self._rows};1H\033[2K{resources}\0338"
        )
        sys.stdout.flush()

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
            self._draw()
            timeout = max(0.0, next_sample - time.monotonic()) if active else None
            self._wake.wait(timeout)
            self._wake.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        if self.visible:
            sys.stdout.flush()
            sys.stdout.write(
                f"\033[1;{self._rows - 2}r\033[{self._rows - 2};1H"
            )
            sys.stdout.flush()
        self._thread = threading.Thread(
            target=self._run, name="s600-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self.visible:
            sys.stdout.write(
                f"\0337\033[r\033[{self._rows - 1};1H\033[2K"
                f"\033[{self._rows};1H\033[2K\0338"
            )
            sys.stdout.flush()


def format_resource_values(values: tuple[float | None, ...]) -> str:
    return "/".join("--" if value is None else f"{value:.1f}%" for value in values)


def resource_summary_lines(summary: ResourceSummary) -> list[str]:
    def value(number: float | None, suffix: str = "%") -> str:
        return "--" if number is None else f"{number:.1f}{suffix}"

    return [
        f"  BPU cores   avg {format_resource_values(summary.bpu_average_percent):<27} "
        f"peak {format_resource_values(summary.bpu_peak_percent)}",
        f"  CPU         avg {value(summary.cpu_average_percent):<8} "
        f"peak {value(summary.cpu_peak_percent)}",
        f"  Memory      avg {value(summary.memory_average_percent):<8} "
        f"peak {value(summary.memory_peak_percent)}",
        f"  BPU temp    peak {value(summary.bpu_temperature_peak_celsius, ' C')}  "
        f"samples {summary.sample_count}",
    ]
