#!/usr/bin/env python3
"""Interactive LocateAnything S600 frontend.

The HBM runners stay resident for the whole session.  This provides the same
interaction model as the Qwen ``vlm`` demo without routing LocateAnything
through libxlm's Qwen-specific preprocessing and decode loop.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from run_locateanything import (
    RUNTIME_VERSION,
    RuntimeConfig,
    TASK_COMMANDS,
    build_runtime_environment,
    create_prediction_paths,
    decode_tokens,
    language_runner_command,
    load_tokenizer,
    load_runtime_config_from_args,
    normalize_prompt,
    parse_detections,
    parse_points,
    parse_s600_resource_status,
    postprocess_detections,
    prepare_image,
    print_stage,
    read_generation,
    require_runtime_paths,
    save_annotated_image,
    tokenize_prompt,
    unwrap_box_command,
)


VERSION = RUNTIME_VERSION
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[38;2;0;128;255m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
SUCCESS = "\033[42;37;1m SUCCESS \033[0m"
PROFILE_VALUE_RE = re.compile(r"([a-z_]+)=(-?\d+(?:\.\d+)?)")
LOGO = r"""
  ██╗      ██████╗  ██████╗ █████╗ ████████╗███████╗
  ██║     ██╔═══██╗██╔════╝██╔══██╗╚══██╔══╝██╔════╝
  ██║     ██║   ██║██║     ███████║   ██║   █████╗
  ██║     ██║   ██║██║     ██╔══██║   ██║   ██╔══╝
  ███████╗╚██████╔╝╚██████╗██║  ██║   ██║   ███████╗
  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
                 L O C A T E   A N Y T H I N G
"""


class HbmServer:
    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        name: str,
        startup_timeout_seconds: float,
    ) -> None:
        self.name = name
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self.recent_output: list[str] = []
        self.startup_output: list[str] = []
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_output,
            name=f"{name}-output",
            daemon=True,
        )
        try:
            self._reader_thread.start()
            self._wait_ready(startup_timeout_seconds)
        except BaseException:
            self._terminate()
            raise

    def _read_output(self) -> None:
        try:
            if self.process.stdout is None:
                return
            for line in self.process.stdout:
                self._output_queue.put(line.rstrip("\r\n"))
        finally:
            self._output_queue.put(None)

    def _write(self, value: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError(f"{self.name} server stdin is closed")
        self.process.stdin.write(value + "\n")
        self.process.stdin.flush()

    def _readline(self, timeout: float | None = None) -> str:
        try:
            value = self._output_queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(f"{self.name} server did not become ready in time") from error
        if value is None:
            details = " | ".join(self.recent_output[-8:])
            suffix = f"; output: {details}" if details else ""
            raise RuntimeError(
                f"{self.name} server exited with code {self.process.poll()}{suffix}"
            )
        self.recent_output.append(value)
        del self.recent_output[:-32]
        return value

    def _wait_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.name} server did not become ready within {timeout_seconds:.1f}s"
                )
            line = self._readline(remaining)
            if line.startswith("LAHBM/1\tREADY\t"):
                return
            self.startup_output.append(line)

    def request(
        self,
        fields: list[str],
        on_token: Callable[[int], None] | None = None,
    ) -> tuple[str, list[str]]:
        self._write("\t".join(fields))
        log: list[str] = []
        request_id = fields[2]
        callback_error: Exception | None = None
        terminal_seen = False
        try:
            while True:
                line = self._readline()
                if line.startswith("LAHBM/1\tTOKEN\t"):
                    token_fields = line.split("\t")
                    if len(token_fields) != 4 or token_fields[2] != request_id:
                        raise RuntimeError(f"invalid token frame: {line}")
                    if on_token is not None and callback_error is None:
                        try:
                            on_token(int(token_fields[3]))
                        except Exception as error:
                            # The runner keeps producing this request. Drain it
                            # to a terminal frame before returning the callback
                            # failure, otherwise the next request is misaligned.
                            callback_error = error
                    continue
                if line.startswith(("LAHBM/1\tRESULT\t", "LAHBM/1\tERROR\t")):
                    terminal_fields = line.split("\t")
                    if len(terminal_fields) < 3 or terminal_fields[2] != request_id:
                        raise RuntimeError(f"invalid terminal frame: {line}")
                    terminal_seen = True
                    if callback_error is not None:
                        raise callback_error
                    if terminal_fields[1] == "ERROR":
                        raise RuntimeError(line)
                    return line, log
                log.append(line)
        except BaseException:
            if not terminal_seen:
                self._terminate()
            raise

    def _terminate(self) -> None:
        if self.process.poll() is not None:
            if self._reader_thread.ident is not None:
                try:
                    self._reader_thread.join(timeout=1)
                except KeyboardInterrupt:
                    pass
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            self.process.kill()
            try:
                self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                pass
        if self._reader_thread.ident is not None:
            try:
                self._reader_thread.join(timeout=1)
            except KeyboardInterrupt:
                pass

    def close(self) -> None:
        if self.process.poll() is not None:
            if self._reader_thread.ident is not None:
                self._reader_thread.join(timeout=1)
            return
        try:
            self._write("LAHBM/1\tQUIT")
            self.process.wait(timeout=5)
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt):
            self._terminate()
        else:
            if self._reader_thread.ident is not None:
                try:
                    self._reader_thread.join(timeout=1)
                except KeyboardInterrupt:
                    pass


class ResourceDashboard:
    """Fixed terminal footer backed only by measured board/runtime counters."""

    def __init__(self, enabled: bool = True, interval_seconds: float = 1.0) -> None:
        if interval_seconds < 0.25:
            raise ValueError("dashboard interval must be at least 0.25 seconds")
        self.enabled = enabled and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._active = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._bpu: list[float | None] = [None] * 4
        self._cpu: float | None = None
        self._memory: float | None = None
        self._temperature: float | None = None
        self._ucp_gib_s: float | None = None
        self._previous_cpu: tuple[int, int] | None = None
        self._profile_bytes_mib = 0.0
        self._profile_ms = 0.0
        self._completed_request = False
        self._rows = max(2, shutil.get_terminal_size((140, 32)).lines)

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

    @staticmethod
    def _read_vendor_status() -> tuple[list[float | None], float | None]:
        command = Path("/usr/hobot/bin/hrut_somstatus")
        if not command.is_file():
            return [None] * 4, None
        try:
            completed = subprocess.run(
                [str(command)], capture_output=True, text=True, errors="replace",
                timeout=0.8, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return [None] * 4, None
        if completed.returncode != 0:
            return [None] * 4, None
        return parse_s600_resource_status(completed.stdout)

    @staticmethod
    def _color_percent(value: float | None) -> str:
        if value is None:
            return f"{DIM}--{RESET}"
        color = GREEN if value < 55 else YELLOW if value < 85 else RED
        return f"{color}{value:3.0f}%{RESET}"

    def begin_request(self) -> None:
        with self._lock:
            self._profile_bytes_mib = 0.0
            self._profile_ms = 0.0
            self._ucp_gib_s = None
            self._cpu = None
            self._memory = None
            self._bpu = [None] * 4
            self._temperature = None
            self._completed_request = False
        self._previous_cpu = self._read_system_cpu()
        self._active.set()
        self._wake.set()

    def end_request(self) -> None:
        self._active.clear()
        with self._lock:
            self._completed_request = True
        self._wake.set()

    @contextmanager
    def request_scope(self):
        self.begin_request()
        try:
            yield
        finally:
            self.end_request()

    def observe_profile(self, line: str) -> None:
        values = {name: float(value) for name, value in PROFILE_VALUE_RE.findall(line)}
        host_mib = values.get("input_mib", 0.0) + values.get("output_mib", 0.0)
        host_ms = sum(values.get(name, 0.0) for name in (
            "pack", "input_flush", "output_flush", "unpack"
        ))
        if host_mib <= 0 or host_ms <= 0:
            return
        with self._lock:
            self._profile_bytes_mib += host_mib
            self._profile_ms += host_ms
            self._ucp_gib_s = self._profile_bytes_mib * 1000.0 / self._profile_ms / 1024.0
        self._wake.set()

    def _sample(self) -> None:
        cpu_now = self._read_system_cpu()
        cpu_percent = None
        if cpu_now is not None and self._previous_cpu is not None:
            total_delta = cpu_now[0] - self._previous_cpu[0]
            idle_delta = cpu_now[1] - self._previous_cpu[1]
            if total_delta > 0:
                cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)
        self._previous_cpu = cpu_now
        bpu, temperature = self._read_vendor_status()
        with self._lock:
            self._bpu = bpu
            self._cpu = cpu_percent
            self._memory = self._read_memory_percent()
            self._temperature = temperature

    def _render(self) -> str:
        with self._lock:
            bpu = " ".join(
                f"{index}:{self._color_percent(value)}"
                for index, value in enumerate(self._bpu)
            )
            ucp = (
                f"{GREEN}{self._ucp_gib_s:.2f} GiB/s{RESET}"
                if self._ucp_gib_s is not None else f"{DIM}--{RESET}"
            )
            temperature = (
                f"{self._temperature:.1f} C" if self._temperature is not None else "--"
            )
            state = (
                "ACTIVE" if self._active.is_set()
                else "LAST REQUEST" if self._completed_request
                else "IDLE"
            )
            return (
                f"{BOLD}{CYAN}◆ S600 {state}{RESET}  "
                f"BPU[{bpu}]  │  "
                f"SYS CPU {self._color_percent(self._cpu)}  │  "
                f"MEM {self._color_percent(self._memory)}  │  "
                f"Host I/O(est.) {ucp}  │  BPU {temperature}"
            )

    def _draw(self) -> None:
        if not self.enabled:
            return
        columns = shutil.get_terminal_size((140, 32)).columns
        line = self._render()
        # ANSI color sequences do not consume columns. A small margin avoids
        # wrapping even on terminals narrower than the full dashboard.
        if columns < 112:
            with self._lock:
                line = (
                    f"{BOLD}{CYAN}◆ S600{RESET} BPU "
                    + "/".join(
                        "--" if value is None else f"{value:.0f}%" for value in self._bpu
                    )
                    + f" │ CPU {('--' if self._cpu is None else f'{self._cpu:.0f}%')}"
                    + f" │ MEM {('--' if self._memory is None else f'{self._memory:.0f}%')}"
                )
        sys.stdout.write(f"\0337\033[{self._rows};1H\033[2K{line}\0338")
        sys.stdout.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._active.is_set():
                self._sample()
            self._draw()
            if self._stop.is_set():
                break
            timeout = self.interval_seconds if self._active.is_set() else None
            self._wake.wait(timeout)
            self._wake.clear()

    def start(self) -> None:
        if not self.enabled:
            return
        # DECSTBM resets the cursor to the home position. Move it back to the
        # last scrollable row before input() writes the interactive prompt.
        sys.stdout.flush()
        sys.stdout.write(
            f"\033[1;{self._rows - 1}r\033[{self._rows - 1};1H"
        )
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._run, name="s600-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        sys.stdout.write(f"\0337\033[r\033[{self._rows};1H\033[2K\0338")
        sys.stdout.flush()


def close_runtime_resources(
    dashboard: ResourceDashboard | None,
    language: HbmServer | None,
    vision: HbmServer | None,
) -> None:
    """Attempt every shutdown step so one broken stream cannot leak a runner."""
    failures: list[tuple[str, BaseException]] = []
    resources = (
        ("dashboard", dashboard, "stop"),
        ("language", language, "close"),
        ("vision", vision, "close"),
    )
    previous_sigint = None
    can_mask_sigint = threading.current_thread() is threading.main_thread()
    if can_mask_sigint:
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for label, resource, method_name in resources:
            if resource is None:
                continue
            try:
                getattr(resource, method_name)()
            except (Exception, KeyboardInterrupt) as error:
                failures.append((label, error))
    finally:
        if can_mask_sigint and previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
    for label, error in failures:
        try:
            print(
                f"[WARN] failed to close {label}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        except Exception:
            # Output streams may be the reason cleanup started failing. All
            # resource shutdown attempts have already completed at this point.
            break


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    runtime = load_runtime_config_from_args(argv)
    parser = argparse.ArgumentParser(
        prog="LocateAnything",
        description="Resident interactive LocateAnything Vision + Language HBM runtime.",
        epilog=(
            "Examples:\n"
            "  LocateAnything\n"
            "  LocateAnything -i image.jpg\n"
            "  LocateAnything -i image.jpg -p '/detect cat'\n\n"
            "Task commands:\n  " + "\n  ".join(TASK_COMMANDS)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=runtime.source,
        help=f"runtime config (default: {runtime.source})",
    )
    parser.add_argument("-i", "--image", type=Path, help="initial image")
    parser.add_argument("-p", "--prompt", help="run one initial prompt after startup")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="session directory; each request gets prediction, image, timings, and logs",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=runtime.default_max_new_tokens,
        help=("maximum generated tokens per request "
              f"(default: {runtime.default_max_new_tokens})"),
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="disable the live S600 resource footer",
    )
    parser.add_argument(
        "--show-runner-details",
        action="store_true",
        help="print detailed runner timing lines in addition to the live footer",
    )
    parser.add_argument(
        "--show-profiles",
        dest="show_runner_details",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--generation-mode", choices=("hybrid", "slow"),
        default=runtime.default_generation_mode,
        help=("hybrid=q6 PBD with q1 fallback; slow=q1 AR "
              f"(default: {runtime.default_generation_mode})"),
    )
    parser.add_argument(
        "--nms-iou", type=float, default=runtime.default_nms_iou,
        help=f"same-label Detection NMS threshold (default: {runtime.default_nms_iou:.2f})",
    )
    parser.add_argument(
        "--no-nms", action="store_true",
        help="disable Detection NMS while retaining raw model boxes",
    )
    parser.add_argument("--version", action="version", version=f"LocateAnything {VERSION}")
    args = parser.parse_args(argv)
    args.runtime = runtime
    return args


def print_result(
    detections: list[dict[str, object]],
    points: list[dict[str, object]],
    vit_ms: float,
    vit_infer_ms: float,
    vision_cached: bool,
    prefill_tokens: int,
    prefill_ms: float,
    decode_tokens: int,
    decode_ms: float,
    total_ms: float,
) -> None:
    labels = list(dict.fromkeys(
        str(item["label"])
        for item in [*detections, *points]
        if item.get("label")
    ))
    prefill_tps = prefill_tokens * 1000.0 / prefill_ms if prefill_ms > 0 else 0.0
    decode_tps = decode_tokens * 1000.0 / decode_ms if decode_ms > 0 else 0.0
    decode_tpot = decode_ms / decode_tokens if decode_tokens > 0 else 0.0
    print()
    print(
        f"{CYAN}===== vit cost: {vit_ms:.3f} ms, "
        f"vit infer cost: {vit_infer_ms:.3f} ms, "
        f"cached: {'yes' if vision_cached else 'no'} ====={RESET}"
    )
    print(
        f"{CYAN}===== prefill token num: {prefill_tokens} "
        f"prefill cost: {prefill_ms:.3f} ms, "
        f"prefill speed: {prefill_tps:.3f} tokens/s ====={RESET}"
    )
    print(
        f"{CYAN}===== decode token num: {decode_tokens} "
        f"cost per token: {decode_tpot:.3f} ms, "
        f"decode speed: {decode_tps:.3f} tokens/s ====={RESET}"
    )
    print(f"{CYAN}===== end to end cost: {total_ms:.3f} ms ====={RESET}")
    print()
    print(f"[LocateAnything] labels={', '.join(labels) if labels else '(none)'}")
    print(f"[LocateAnything] boxes={len(detections)}")
    for index, item in enumerate(detections, 1):
        print(
            f"  box[{index}] label={item['label']!r} "
            f"normalized_1000={item['bbox_profile_1000']} "
            f"original_px={item['bbox_xyxy']}"
        )
    print(f"[LocateAnything] points={len(points)}")
    for index, item in enumerate(points, 1):
        print(
            f"  point[{index}] label={item['label']!r} "
            f"normalized_1000={item['point_profile_1000']} "
            f"original_px={item['point_xy']}"
        )


def print_runtime_info(
    vision: HbmServer,
    language: HbmServer,
    runtime: RuntimeConfig,
    image: Path | None,
    generation_mode: str,
    max_new_tokens: int,
) -> None:
    printed: set[str] = set()
    for line in vision.startup_output + language.startup_output:
        if line not in printed and any(
            marker in line for marker in ("[UCP]", "[DNN]", "[BPU]")
        ):
            print(line)
            printed.add(line)
    core_text = ",".join(str(core) for core in runtime.bpu_cores)
    print(f"[INFO] runtime_version={RUNTIME_VERSION} config={runtime.source}")
    print(f"[INFO] model_type={runtime.model_type}")
    print(f"[INFO] backend=HBRT target=S600/Nash-P bpu_cores={core_text}")
    print(f"[INFO] load visual graph success: {runtime.vision_model}")
    print(f"[INFO] load language graphs success: {runtime.language_model}")
    print(f"[INFO] load token embeddings success: {runtime.embeddings}")
    print(f"{SUCCESS} {GREEN}LocateAnything S600 Runtime is ready.{RESET}")
    print(LOGO)
    print(f"{CYAN}================== RUNTIME CONFIG =================={RESET}")
    print(f"{BOLD}模型{RESET}  {runtime.model_type}  |  图像、文本定位")
    print(
        f"{BOLD}Vision{RESET} {runtime.vision_model.name}  |  "
        f"{runtime.image_width}x{runtime.image_height}, {runtime.visual_tokens} visual tokens"
    )
    print(
        f"{BOLD}Language{RESET} {runtime.language_model.name}  |  "
        f"prefill={runtime.prefill_chunk}, PBD={runtime.pbd_query_len}, "
        f"AR={runtime.ar_query_len}, cache={runtime.cache_len}"
    )
    print(f"{BOLD}KV cache{RESET} UCP/DDR resident ring  |  Host only commits new rows")
    print(f"{BOLD}Embedding{RESET} {runtime.embeddings.name}")
    print(f"{BOLD}Generation{RESET} {generation_mode}  |  max_new_tokens={max_new_tokens}")
    print(f"{BOLD}BPU{RESET} Nash-P core {core_text}  |  L2M={runtime.l2m_sizes}")
    print(
        f"{BOLD}Telemetry{RESET} {runtime.telemetry_interval_seconds:.2f}s refresh  |  "
        "disable with --no-dashboard"
    )
    print(f"{DIM}  SYS CPU=整机占用；Host I/O(est.)=由 profile 搬运字节与 Host 时间估算{RESET}")
    if image is not None:
        print(f"{BOLD}Image{RESET} {image.name}")
    print()
    print(f"{CYAN}================== TASK COMMANDS ==================={RESET}")
    print(f"  {GREEN}/detect{RESET} cat,dog                 {DIM}目标检测{RESET}")
    print(f"  {MAGENTA}/ground{RESET} <phrase>             {DIM}指代表达，多目标{RESET}")
    print(f"  {MAGENTA}/ground_single{RESET} <phrase>      {DIM}指代表达，单目标{RESET}")
    print(f"  {BLUE}/gui{RESET} <element>                 {DIM}GUI 点定位{RESET}")
    print(f"  {BLUE}/gui_box{RESET} <element>             {DIM}GUI 框定位{RESET}")
    print(f"  {YELLOW}/text{RESET}                         {DIM}文本 OCR{RESET}")
    print(f"  {YELLOW}/ground_text{RESET} <text>          {DIM}指定文本定位{RESET}")
    print(f"  {CYAN}/layout{RESET} title,table,figure      {DIM}文档版面分析{RESET}")
    print(f"  {RED}/point{RESET} <target>                 {DIM}通用点定位{RESET}")
    print(f"  {GREEN}/box{RESET} /detect cat              {DIM}保存预测框图片{RESET}")
    print()
    print(f"{CYAN}================== SESSION ========================={RESET}")
    print(f"  {BOLD}/image{RESET} <image_path>             {DIM}加载图片{RESET}")
    print(f"  {BOLD}regen{RESET}                           {DIM}重跑上次请求{RESET}")
    print(f"  {BOLD}reset{RESET}                           {DIM}清除当前图片与缓存{RESET}")
    print(f"  {BOLD}exit{RESET}                            {DIM}退出程序{RESET}")
    print()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be between 0 and 1")
    if args.prompt and args.image is None:
        raise ValueError("--prompt requires --image")

    current_image = args.image.expanduser().resolve() if args.image else None
    if current_image is not None and not current_image.is_file():
        raise FileNotFoundError(current_image)

    runtime: RuntimeConfig = args.runtime
    require_runtime_paths(runtime)
    vision_runner = runtime.vision_runner
    tokenizer_dir = runtime.tokenizer_dir
    vision_model = runtime.vision_model
    stream_tokenizer = load_tokenizer(tokenizer_dir)
    output_dir = args.output_dir.resolve() if args.output_dir else None

    env = build_runtime_environment(runtime)
    env.setdefault("LA_PROFILE_EXECUTION", "1")
    vision = HbmServer(
        [str(vision_runner), "--model", str(vision_model), "--server"],
        env,
        "vision",
        runtime.runner_startup_timeout_seconds,
    )
    try:
        language = HbmServer(
            language_runner_command(runtime) + ["--server"],
            env,
            "language",
            runtime.runner_startup_timeout_seconds,
        )
    except BaseException:
        close_runtime_resources(None, None, vision)
        raise

    last_request: tuple[Path, str] | None = None
    vision_cache: tuple[tuple[str, int, int], bytes, dict[str, object]] | None = None
    request_index = 0
    dashboard: ResourceDashboard | None = None
    try:
        dashboard = ResourceDashboard(
            enabled=not args.no_dashboard,
            interval_seconds=runtime.telemetry_interval_seconds,
        )
        print_runtime_info(
            vision,
            language,
            runtime,
            current_image,
            args.generation_mode,
            args.max_new_tokens,
        )
        dashboard.start()
    except BaseException:
        close_runtime_resources(dashboard, language, vision)
        raise

    # The initialization guard above guarantees this before infer() is defined.
    assert dashboard is not None

    def infer(image: Path, prompt: str) -> None:
        nonlocal request_index, last_request, vision_cache
        if not image.is_file():
            raise FileNotFoundError(image)
        task_prompt, annotate = unwrap_box_command(prompt)
        normalized_prompt, task = normalize_prompt(task_prompt)
        request_index += 1
        last_request = (image, prompt)
        started = time.monotonic()
        request_output_dir = (
            output_dir / f"request_{request_index:04d}" if output_dir else None
        )
        paths = create_prediction_paths(
            runtime.layout_root,
            image,
            output_dir=request_output_dir,
        )
        with dashboard.request_scope(), tempfile.TemporaryDirectory(
            prefix="locateanything-interactive-"
        ) as raw_dir:
            work_dir = Path(raw_dir)
            print_stage(1, "Vision")
            vit_started = time.monotonic()
            vision_input_path = work_dir / "vision_input.f16.bin"
            visual_output_path = work_dir / "visual_features.f16.bin"
            token_path = work_dir / "prompt_tokens.i32.bin"
            generation_path = work_dir / "generation.txt"
            image_stat = image.stat()
            cache_key = (str(image.resolve()), image_stat.st_size, image_stat.st_mtime_ns)
            vision_cached = vision_cache is not None and vision_cache[0] == cache_key
            if vision_cached:
                transform = vision_cache[2]
                visual_output_path.write_bytes(vision_cache[1])
                vit_infer_ms = 0.0
                vision_result = "cached visual features"
            else:
                vision_input, transform = prepare_image(image)
                vision_input.tofile(vision_input_path)
                vision_result, _ = vision.request([
                    "LAHBM/1", "RUN", str(request_index), str(vision_input_path), str(visual_output_path)
                ])
                vision_fields = vision_result.split("\t")
                if len(vision_fields) < 5:
                    raise RuntimeError(f"invalid Vision response: {vision_result}")
                vit_infer_ms = float(vision_fields[3])
                vision_cache = (cache_key, visual_output_path.read_bytes(), transform)
            vit_ms = (time.monotonic() - vit_started) * 1000.0
            print_stage(1, "Vision", vit_ms / 1000.0)

            print_stage(2, "Language")
            language_started = time.monotonic()
            tokens = tokenize_prompt(tokenizer_dir, task_prompt, stream_tokenizer)
            tokens.tofile(token_path)

            streamed_ids: list[int] = []
            streamed_text = ""

            def stream_token(token: int) -> None:
                nonlocal streamed_text
                streamed_ids.append(token)
                decoded = stream_tokenizer.decode(
                    streamed_ids, skip_special_tokens=False
                ).rstrip("\ufffd")
                if decoded.startswith(streamed_text):
                    print(decoded[len(streamed_text):], end="", flush=True)
                    streamed_text = decoded

            if normalized_prompt != prompt:
                print(f"[LocateAnything] task={task}")
                print(f"[LocateAnything] prompt={normalized_prompt}")
            print("[Assistant] >>> ", end="", flush=True)
            try:
                language_result, language_log = language.request([
                    "LAHBM/1", "RUN", str(request_index), str(token_path),
                    str(visual_output_path), str(generation_path),
                    str(args.max_new_tokens), args.generation_mode,
                ], on_token=stream_token)
            except Exception:
                print()
                raise
            language_fields = language_result.split("\t")
            if len(language_fields) < 8:
                raise RuntimeError(f"invalid Language response: {language_result}")
            decode_token_count = int(language_fields[4])
            prefill_tokens = int(language_fields[5])
            prefill_ms = float(language_fields[6])
            decode_ms = float(language_fields[7])

            stop_reason, token_ids = read_generation(generation_path)
            text = decode_tokens(tokenizer_dir, token_ids, stream_tokenizer)
            if text.startswith(streamed_text) and len(text) > len(streamed_text):
                print(text[len(streamed_text):], end="", flush=True)
            elif text != streamed_text:
                print(f"\n[Assistant final] >>> {text}", end="", flush=True)
            print()
            for line in language_log:
                if line.startswith("[profile]"):
                    dashboard.observe_profile(line)
                    if args.show_runner_details:
                        print(line)
                elif line.startswith(("[pbd]", "[hybrid:")):
                    print(line)
            language_seconds = time.monotonic() - language_started
            print_stage(2, "Language", language_seconds)

            print_stage(3, "Postprocess")
            postprocess_started = time.monotonic()
            raw_detections = parse_detections(text, transform)
            detections, suppressed_detections = postprocess_detections(
                raw_detections,
                task,
                iou_threshold=args.nms_iou,
                enabled=not args.no_nms,
            )
            points = parse_points(text, transform)
            annotated_image = None
            if annotate or detections or points:
                annotated_image = paths.annotated_image
                save_annotated_image(image, detections, points, annotated_image)
            postprocess_seconds = time.monotonic() - postprocess_started
            total_ms = (time.monotonic() - started) * 1000.0
            print_stage(3, "Postprocess", postprocess_seconds)
            print_result(
                detections,
                points,
                vit_ms,
                vit_infer_ms,
                vision_cached,
                prefill_tokens,
                prefill_ms,
                decode_token_count,
                decode_ms,
                total_ms,
            )
            if suppressed_detections:
                print(
                    f"[LocateAnything] NMS suppressed={len(suppressed_detections)} "
                    f"same-label boxes at IoU>={args.nms_iou:.2f}"
                )
            if annotated_image:
                print(f"[Output] annotated image: {annotated_image}")
            timing_data = {
                "schema_version": 1,
                "stages_seconds": {
                    "vision": round(vit_ms / 1000.0, 6),
                    "language": round(language_seconds, 6),
                    "postprocess": round(postprocess_seconds, 6),
                },
                "runner_seconds": {
                    "vision_hbm": round(vit_infer_ms / 1000.0, 6),
                    "language_prefill_hbm": round(prefill_ms / 1000.0, 6),
                    "language_decode_hbm": round(decode_ms / 1000.0, 6),
                },
                "total_seconds": round(total_ms / 1000.0, 6),
            }
            paths.timings.write_text(
                json.dumps(timing_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            paths.runtime_log.write_text(
                "VISION\n"
                + vision_result
                + "\n\nLANGUAGE\n"
                + "\n".join(language_log)
                + "\n",
                encoding="utf-8",
            )
            paths.prediction.write_text(json.dumps({
                    "schema_version": 1,
                    "runtime": {
                        "version": RUNTIME_VERSION,
                        "config": str(runtime.source),
                        "model_type": runtime.model_type,
                        "runtime_specification": runtime.specification(),
                    },
                    "image": str(image),
                    "prompt": prompt,
                    "normalized_prompt": normalized_prompt,
                    "task": task,
                    "text": text,
                    "raw_detections": raw_detections,
                    "detections": detections,
                    "suppressed_detections": suppressed_detections,
                    "points": points,
                    "annotated_image": str(annotated_image) if annotated_image else None,
                    "generation": {
                        "mode": args.generation_mode,
                        "stop_reason": stop_reason,
                        "token_count": len(token_ids),
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "elapsed_seconds": round(total_ms / 1000.0, 3),
                    "postprocess": {
                        "method": "class_aware_nms"
                        if task == "object_detection" and not args.no_nms
                        else "none",
                        "iou_threshold": args.nms_iou,
                        "raw_detection_count": len(raw_detections),
                        "kept_detection_count": len(detections),
                        "suppressed_detection_count": len(suppressed_detections),
                    },
                    "performance": {
                        "vit_ms": round(vit_ms, 3),
                        "vit_infer_ms": round(vit_infer_ms, 3),
                        "vision_cached": vision_cached,
                        "prefill_tokens": prefill_tokens,
                        "prefill_ms": round(prefill_ms, 3),
                        "decode_tokens": decode_token_count,
                        "decode_ms": round(decode_ms, 3),
                    },
                    "timings": timing_data,
                    "runtime_log": str(paths.runtime_log),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[Output] prediction: {paths.prediction}")
            print(f"[Output] timings: {paths.timings}")
            print(f"[Output] run directory: {paths.root}")

    try:
        if current_image is not None and args.prompt:
            print(f"[User] <<< {args.prompt}")
            infer(current_image, args.prompt)
        while True:
            try:
                line = input("[User] <<< ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "exit":
                break
            if line == "reset":
                current_image = None
                last_request = None
                vision_cache = None
                print("[LocateAnything] session reset")
                continue
            if line == "regen":
                if last_request is None:
                    print("[LocateAnything] no previous request")
                else:
                    print(f"[User] <<< {last_request[1]}")
                    infer(*last_request)
                continue
            if line.startswith("/image"):
                parts = shlex.split(line)
                if len(parts) != 2:
                    print("usage: /image IMAGE_PATH")
                    continue
                candidate = Path(parts[1]).expanduser().resolve()
                if not candidate.is_file():
                    print(f"[LocateAnything] image not found: {candidate}")
                    continue
                current_image = candidate
                print(f"[LocateAnything] image loaded: {current_image}")
                continue
            if current_image is None:
                print("[LocateAnything] load an image first with /image IMAGE_PATH")
                continue
            try:
                infer(current_image, line)
            except Exception as error:
                print(f"[LocateAnything] request failed: {error}", file=sys.stderr)
    finally:
        close_runtime_resources(dashboard, language, vision)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
