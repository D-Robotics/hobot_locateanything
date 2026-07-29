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
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from run_locateanything import (
    DEFAULT_GENERATION_MODE,
    DEFAULT_NMS_IOU,
    EMBEDDINGS,
    LANGUAGE_MODEL,
    TASK_COMMANDS,
    VISION_MODEL,
    decode_tokens,
    load_tokenizer,
    normalize_prompt,
    annotated_output_path,
    parse_detections,
    parse_points,
    postprocess_detections,
    prepare_image,
    resolve_repo_path,
    save_annotated_image,
    tokenize_prompt,
    unwrap_box_command,
)


VERSION = "0.4.0"
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
BPU_STATUS_RE = re.compile(
    r"^\s*bpu(?P<core>[0-3])\s*:\s*(?P<value>\d+(?:\.\d+)?)\s*%?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BPU_TEMPERATURE_RE = re.compile(
    r"pvt_bpu_[^:]+:\s*(?P<value>-?\d+(?:\.\d+)?)\s*\(C\)",
    re.IGNORECASE,
)
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
    def __init__(self, command: list[str], env: dict[str, str], name: str) -> None:
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
        self._wait_ready()

    def _write(self, value: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError(f"{self.name} server stdin is closed")
        self.process.stdin.write(value + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        if self.process.stdout is None:
            raise RuntimeError(f"{self.name} server stdout is closed")
        line = self.process.stdout.readline()
        if not line:
            details = " | ".join(self.recent_output[-8:])
            suffix = f"; output: {details}" if details else ""
            raise RuntimeError(
                f"{self.name} server exited with code {self.process.poll()}{suffix}"
            )
        value = line.rstrip("\r\n")
        self.recent_output.append(value)
        del self.recent_output[:-32]
        return value

    def _wait_ready(self) -> None:
        while True:
            line = self._readline()
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
        while True:
            line = self._readline()
            if line.startswith("LAHBM/1\tTOKEN\t"):
                token_fields = line.split("\t")
                if len(token_fields) != 4 or token_fields[2] != request_id:
                    raise RuntimeError(f"invalid token frame: {line}")
                if on_token is not None:
                    on_token(int(token_fields[3]))
                continue
            if line.startswith("LAHBM/1\tRESULT\t"):
                return line, log
            if line.startswith("LAHBM/1\tERROR\t"):
                raise RuntimeError(line)
            log.append(line)

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self._write("LAHBM/1\tQUIT")
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ResourceDashboard:
    """Fixed terminal footer backed only by measured board/runtime counters."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self._stop = threading.Event()
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
        bpu: list[float | None] = [None] * 4
        for match in BPU_STATUS_RE.finditer(completed.stdout):
            bpu[int(match.group("core"))] = float(match.group("value"))
        temperatures = [
            float(match.group("value"))
            for match in BPU_TEMPERATURE_RE.finditer(completed.stdout)
        ]
        return bpu, max(temperatures) if temperatures else None

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
            return (
                f"{BOLD}{CYAN}◆ S600 LIVE{RESET}  BPU[{bpu}]  │  "
                f"CPU {self._color_percent(self._cpu)}  │  "
                f"MEM {self._color_percent(self._memory)}  │  "
                f"Host↔UCP {ucp}  │  BPU {temperature}"
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
            self._sample()
            self._draw()
            self._stop.wait(0.5)

    def start(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write(f"\033[1;{self._rows - 1}r")
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._run, name="s600-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        sys.stdout.write(f"\0337\033[r\033[{self._rows};1H\033[2K\0338")
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("-i", "--image", type=Path, help="initial image")
    parser.add_argument("-p", "--prompt", help="run one initial prompt after startup")
    parser.add_argument("-o", "--output-dir", type=Path, help="optional per-request JSON directory")
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="maximum generated tokens per request (default: 2048)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="disable the live S600 resource footer",
    )
    parser.add_argument(
        "--show-profiles", action="store_true",
        help="print raw graph profile lines in addition to the live footer",
    )
    parser.add_argument(
        "--generation-mode", choices=("hybrid", "slow"),
        default=DEFAULT_GENERATION_MODE,
        help="hybrid=q6 PBD with q1 fallback; slow=q1 AR (default: hybrid)",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=DEFAULT_NMS_IOU,
        help="same-label Detection NMS threshold (default: 0.90)",
    )
    parser.add_argument(
        "--no-nms", action="store_true",
        help="disable Detection NMS while retaining raw model boxes",
    )
    parser.add_argument("--version", action="version", version=f"LocateAnything {VERSION}")
    return parser.parse_args()


def parse_generation(path: Path) -> tuple[str, list[int]]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    if "stop_reason" not in values or "token_ids" not in values:
        raise ValueError(f"invalid Language output: {path}")
    return values["stop_reason"], [
        int(value) for value in values["token_ids"].split(",") if value
    ]


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
    vision_model: Path,
    language_model: Path,
    embeddings: Path,
    image: Path | None,
    generation_mode: str,
    max_new_tokens: int,
    l2m: str,
) -> None:
    printed: set[str] = set()
    for line in vision.startup_output + language.startup_output:
        if line not in printed and any(
            marker in line for marker in ("[UCP]", "[DNN]", "[BPU]")
        ):
            print(line)
            printed.add(line)
    print("[INFO] model_type=LocateAnything-3B")
    print("[INFO] backend=HBRT target=S600/Nash-P bpu_cores=0,1,2,3")
    print(f"[INFO] load visual graph success: {vision_model}")
    print(f"[INFO] load language graphs success: {language_model}")
    print(f"[INFO] load token embeddings success: {embeddings}")
    print(f"{SUCCESS} {GREEN}LocateAnything S600 Runtime is ready.{RESET}")
    print(LOGO)
    print(f"{CYAN}================== RUNTIME CONFIG =================={RESET}")
    print(f"{BOLD}模型{RESET}  LocateAnything-3B  |  图像、文本定位")
    print(f"{BOLD}Vision{RESET} {vision_model.name}  |  672x672, 576 visual tokens")
    print(f"{BOLD}Language{RESET} {language_model.name}  |  prefill=1024, PBD=6, AR=1, cache=4096")
    print(f"{BOLD}KV cache{RESET} UCP/DDR resident ring  |  Host only commits new rows")
    print(f"{BOLD}Embedding{RESET} {embeddings.name}")
    print(f"{BOLD}Generation{RESET} {generation_mode}  |  max_new_tokens={max_new_tokens}")
    print(f"{BOLD}BPU{RESET} Nash-P core 0,1,2,3  |  L2M={l2m}")
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
    print(f"  {GREEN}/box{RESET} /detect cat              {DIM}保存预测框图片到 ./output{RESET}")
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
    if args.prompt and args.image is None:
        raise ValueError("--prompt requires --image")

    current_image = args.image.expanduser().resolve() if args.image else None
    if current_image is not None and not current_image.is_file():
        raise FileNotFoundError(current_image)

    runtime_dir = Path(__file__).resolve().parent
    repo_root = runtime_dir.parent
    vision_runner = resolve_repo_path(
        repo_root,
        os.environ.get("LA_VISION_RUNNER", str(runtime_dir / "build" / "vision_hbm_runner")),
    )
    language_runner = resolve_repo_path(
        repo_root,
        os.environ.get(
            "LA_LANGUAGE_RUNNER", str(runtime_dir / "build" / "language_hbm_runner")
        ),
    )
    tokenizer_dir = runtime_dir / "tokenizer"
    vision_model = resolve_repo_path(repo_root, os.environ.get("LA_VISION_MODEL", VISION_MODEL))
    language_model = resolve_repo_path(repo_root, os.environ.get("LA_LANGUAGE_MODEL", LANGUAGE_MODEL))
    embeddings = resolve_repo_path(repo_root, os.environ.get("LA_EMBEDDINGS", EMBEDDINGS))
    required = (vision_runner, language_runner, tokenizer_dir, vision_model, language_model, embeddings)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    stream_tokenizer = load_tokenizer(tokenizer_dir)

    env = os.environ.copy()
    env.setdefault("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
    env.setdefault("LA_PROFILE_EXECUTION", "1")
    vision = HbmServer(
        [str(vision_runner), "--model", str(vision_model), "--server"], env, "vision"
    )
    try:
        language = HbmServer(
            [str(language_runner), "--model", str(language_model), "--embed", str(embeddings), "--server"],
            env,
            "language",
        )
    except Exception:
        vision.close()
        raise

    last_request: tuple[Path, str] | None = None
    vision_cache: tuple[tuple[str, int, int], bytes, dict[str, object]] | None = None
    request_index = 0
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print_runtime_info(
        vision,
        language,
        vision_model,
        language_model,
        embeddings,
        current_image,
        args.generation_mode,
        args.max_new_tokens,
        env["HB_DNN_USER_DEFINED_L2M_SIZES"],
    )
    dashboard = ResourceDashboard(enabled=not args.no_dashboard)
    dashboard.start()

    def infer(image: Path, prompt: str) -> None:
        nonlocal request_index, last_request, vision_cache
        if not image.is_file():
            raise FileNotFoundError(image)
        task_prompt, annotate = unwrap_box_command(prompt)
        normalized_prompt, task = normalize_prompt(task_prompt)
        request_index += 1
        last_request = (image, prompt)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="locateanything-interactive-") as raw_dir:
            work_dir = Path(raw_dir)
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
            dashboard.begin_request()
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

            stop_reason, token_ids = parse_generation(generation_path)
            text = decode_tokens(tokenizer_dir, token_ids, stream_tokenizer)
            if text.startswith(streamed_text) and len(text) > len(streamed_text):
                print(text[len(streamed_text):], end="", flush=True)
            elif text != streamed_text:
                print(f"\n[Assistant final] >>> {text}", end="", flush=True)
            print()
            for line in language_log:
                if line.startswith("[profile]"):
                    dashboard.observe_profile(line)
                    if args.show_profiles:
                        print(line)
                elif line.startswith(("[pbd]", "[hybrid:")):
                    print(line)
            raw_detections = parse_detections(text, transform)
            detections, suppressed_detections = postprocess_detections(
                raw_detections,
                task,
                iou_threshold=args.nms_iou,
                enabled=not args.no_nms,
            )
            points = parse_points(text, transform)
            annotated_image = None
            if annotate:
                annotated_image = annotated_output_path(image, task)
                save_annotated_image(image, detections, points, annotated_image)
            total_ms = (time.monotonic() - started) * 1000.0
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
            if output_dir:
                output_path = output_dir / f"request_{request_index:04d}.json"
                output_path.write_text(json.dumps({
                    "schema_version": 1,
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
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"[Output] {output_path}")

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
        dashboard.stop()
        language.close()
        vision.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
