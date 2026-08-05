"""Small terminal presentation helpers shared by S600 frontends."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import unicodedata
from contextlib import AbstractContextManager
from typing import TextIO


def _supports_color(stream: TextIO) -> bool:
    return (
        stream.isatty()
        and os.environ.get("TERM") != "dumb"
        and "NO_COLOR" not in os.environ
    )


COLOR = _supports_color(sys.stdout)
RESET = "\033[0m" if COLOR else ""
DIM = "\033[2m" if COLOR else ""
BOLD = "\033[1m" if COLOR else ""
CYAN = "\033[38;2;0;156;196m" if COLOR else ""
GREEN = "\033[32m" if COLOR else ""
YELLOW = "\033[33m" if COLOR else ""
BLUE = "\033[34m" if COLOR else ""
MAGENTA = "\033[35m" if COLOR else ""
RED = "\033[31m" if COLOR else ""

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

LOCATE_BANNER = (
    "  ██╗      ██████╗  ██████╗ █████╗ ████████╗███████╗",
    "  ██║     ██╔═══██╗██╔════╝██╔══██╗╚══██╔══╝██╔════╝",
    "  ██║     ██║   ██║██║     ███████║   ██║   █████╗  ",
    "  ██║     ██║   ██║██║     ██╔══██║   ██║   ██╔══╝  ",
    "  ███████╗╚██████╔╝╚██████╗██║  ██║   ██║   ███████╗",
    "  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝",
)


def strip_ansi(value: str) -> str:
    return ANSI_ESCAPE_RE.sub("", value)


def character_width(value: str) -> int:
    if unicodedata.combining(value):
        return 0
    return 2 if unicodedata.east_asian_width(value) in {"W", "F"} else 1


def visible_width(value: str) -> int:
    return sum(character_width(character) for character in strip_ansi(value))


def truncate_visible(value: str, width: int) -> str:
    """Truncate colored text without splitting ANSI escapes or wide characters."""
    if width <= 0:
        return ""
    result: list[str] = []
    position = 0
    used_color = False
    while position < len(value):
        match = ANSI_ESCAPE_RE.match(value, position)
        if match is not None:
            result.append(match.group(0))
            used_color = True
            position = match.end()
            continue
        character = value[position]
        next_width = character_width(character)
        if next_width and width < next_width:
            break
        result.append(character)
        width -= next_width
        position += 1
    if used_color and position < len(value):
        result.append(RESET or "\033[0m")
    return "".join(result)


def fit_terminal_line(value: str, columns: int) -> str:
    return truncate_visible(value, max(1, columns - 1))


def pad_visible(value: str, width: int) -> str:
    return value + " " * max(0, width - visible_width(value))


def banner_lines() -> list[str]:
    return [f"{BOLD}{CYAN}{line}{RESET}" for line in LOCATE_BANNER]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def heading(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{title}{RESET}")


class WaitIndicator(AbstractContextManager["WaitIndicator"]):
    """Show elapsed time while a blocking startup operation is running."""

    _FRAMES = ("-", "\\", "|", "/")

    def __init__(
        self,
        index: int,
        total: int,
        label: str,
        *,
        stream: TextIO = sys.stdout,
    ) -> None:
        self.index = index
        self.total = total
        self.label = label
        self.stream = stream
        self.interactive = stream.isatty() and os.environ.get("TERM") != "dumb"
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _line(self, frame: str) -> str:
        elapsed = format_duration(time.monotonic() - self.started)
        return (
            f"{CYAN}[{self.index}/{self.total}]{RESET} "
            f"{YELLOW}{frame}{RESET} {BOLD}{self.label}{RESET}  "
            f"{DIM}{elapsed}{RESET}"
        )

    def _animate(self) -> None:
        frame = 0
        while not self._stop.wait(0.12):
            self.stream.write(f"\r\033[2K{self._line(self._FRAMES[frame % 4])}")
            self.stream.flush()
            frame += 1

    def __enter__(self) -> "WaitIndicator":
        self.started = time.monotonic()
        if self.interactive:
            self.stream.write(f"\r\033[2K{self._line(self._FRAMES[0])}")
            self.stream.flush()
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        else:
            print(f"[{self.index}/{self.total}] START {self.label}", file=self.stream, flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        elapsed = format_duration(time.monotonic() - self.started)
        state = f"{GREEN}DONE{RESET}" if exc_type is None else f"{RED}FAILED{RESET}"
        prefix = "\r\033[2K" if self.interactive else ""
        print(
            f"{prefix}[{self.index}/{self.total}] {state} {self.label}  {elapsed}",
            file=self.stream,
            flush=True,
        )
        return False
