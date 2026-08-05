#!/usr/bin/env python3
"""Interactive LocateAnything S600 frontend.

The HBM runners stay resident for the whole session.  This provides the same
interaction model as the Qwen ``vlm`` demo without routing LocateAnything
through libxlm's Qwen-specific preprocessing and decode loop.
"""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from console import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    WaitIndicator,
    banner_lines,
)
from runtime import (
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
    postprocess_detections,
    prepare_image,
    read_generation,
    require_runtime_paths,
    save_annotated_image,
    tokenize_prompt,
)
from telemetry import ResourceMonitor, ResourceSummary, resource_summary_lines
from video import (
    VideoFrameReader,
    VideoFrameWriter,
    VideoInfo,
    create_video_output_dir,
    probe_video,
)


VERSION = RUNTIME_VERSION

TASK_HELP = (
    ("/detect cat,dog", "目标检测"),
    ("/ground <phrase>", "指代表达，多目标"),
    ("/ground_single <phrase>", "指代表达，单目标"),
    ("/gui <element>", "GUI 点定位"),
    ("/gui_box <element>", "GUI 框定位"),
    ("/text", "文本 OCR"),
    ("/ground_text <text>", "指定文本定位"),
    ("/layout title,table,figure", "文档版面分析"),
    ("/point <target>", "通用点定位"),
)
SESSION_HELP = (
    ("/image <image_path>", "加载图片"),
    ("/video <video_path>", "加载视频并处理全部帧"),
    ("regen", "重跑上次请求"),
    ("reset", "清除当前媒体与缓存"),
    ("exit", "退出程序"),
)


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


def close_runtime_resources(
    monitor: ResourceMonitor | None,
    language: HbmServer | None,
    vision: HbmServer | None,
) -> None:
    """Attempt every shutdown step so one broken stream cannot leak a runner."""
    failures: list[tuple[str, BaseException]] = []
    resources = (
        ("resource monitor", monitor, "stop"),
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
                f"{YELLOW}[WARN]{RESET} failed to close {BOLD}{label}{RESET}: "
                f"{type(error).__name__}: {error}",
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
            "Task commands:\n  "
            + "\n  ".join(TASK_COMMANDS)
            + "\n\nSession commands:\n"
            "  /image IMAGE_PATH\n"
            "  /video VIDEO_PATH\n"
            "  regen\n"
            "  reset\n"
            "  exit"
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
        help="disable the fixed terminal header",
    )
    parser.add_argument(
        "--show-runner-details",
        action="store_true",
        help="print detailed runner timing lines",
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
    resources: ResourceSummary,
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
    print(f"{BOLD}{CYAN}Performance{RESET}")
    print(
        f"  {BOLD}Vision{RESET}       {YELLOW}{vit_ms:10.3f} ms{RESET}  "
        f"HBM {YELLOW}{vit_infer_ms:.3f} ms{RESET}  "
        f"cached={GREEN if vision_cached else DIM}"
        f"{'yes' if vision_cached else 'no'}{RESET}"
    )
    print(
        f"  {BOLD}Prefill{RESET}      {YELLOW}{prefill_ms:10.3f} ms{RESET}  "
        f"{GREEN}{prefill_tokens} tokens{RESET}  "
        f"{MAGENTA}{prefill_tps:.3f} tokens/s{RESET}"
    )
    print(
        f"  {BOLD}Decode{RESET}       {YELLOW}{decode_ms:10.3f} ms{RESET}  "
        f"{GREEN}{decode_tokens} tokens{RESET}  "
        f"{YELLOW}{decode_tpot:.3f} ms/token{RESET}  "
        f"{MAGENTA}{decode_tps:.3f} tokens/s{RESET}"
    )
    print(
        f"  {BOLD}End-to-end{RESET}   "
        f"{BOLD}{YELLOW}{total_ms:10.3f} ms{RESET}"
    )
    print(f"{BOLD}{CYAN}Resources{RESET}")
    for line in resource_summary_lines(resources):
        print(line)
    print(f"{BOLD}{CYAN}Result{RESET}")
    label_text = ", ".join(labels) if labels else "(none)"
    print(
        f"  Labels {MAGENTA}{label_text}{RESET}  |  "
        f"Boxes {GREEN}{len(detections)}{RESET}  |  "
        f"Points {GREEN}{len(points)}{RESET}"
    )


def print_command_help() -> None:
    print(f"{BOLD}{CYAN}Tasks{RESET}")
    for command, description in TASK_HELP:
        print(f"  {command:<30} {DIM}{description}{RESET}")
    print(f"{BOLD}{CYAN}Session{RESET}")
    for command, description in SESSION_HELP:
        print(f"  {command:<30} {DIM}{description}{RESET}")


def build_runtime_header(
    vision: HbmServer,
    language: HbmServer,
    generation_mode: str,
    max_new_tokens: int,
    show_runner_details: bool = False,
) -> tuple[list[str], list[str]]:
    details: list[str] = []
    if show_runner_details:
        printed: set[str] = set()
        for line in vision.startup_output + language.startup_output:
            if line not in printed and any(
                marker in line for marker in ("[UCP]", "[DNN]", "[BPU]")
            ):
                details.append(line)
                printed.add(line)

    header = banner_lines()
    header.extend((
        f"{BOLD}{GREEN}Ready{RESET}  S600/Nash-P  |  "
        f"{MAGENTA}{generation_mode}{RESET}  |  "
        f"max tokens {GREEN}{max_new_tokens}{RESET}",
        f"{BOLD}{CYAN}Tasks{RESET}    /detect  /ground  /ground_single  /gui  /gui_box",
        "         /text  /ground_text  /layout  /point  /help",
        f"{BOLD}{CYAN}Session{RESET}  /image  /video  regen  reset  exit",
        f"{DIM}{'─' * 72}{RESET}",
    ))
    return header, details


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
    current_video: tuple[Path, VideoInfo] | None = None

    runtime: RuntimeConfig = args.runtime
    require_runtime_paths(runtime)
    vision_runner = runtime.vision_runner
    tokenizer_dir = runtime.tokenizer_dir
    vision_model = runtime.vision_model
    stream_tokenizer = load_tokenizer(tokenizer_dir)
    output_dir = args.output_dir.resolve() if args.output_dir else None

    env = build_runtime_environment(runtime)
    if args.show_runner_details:
        env.setdefault("LA_PROFILE_EXECUTION", "1")
    with WaitIndicator(1, 2, "Load Vision model"):
        vision = HbmServer(
            [str(vision_runner), "--model", str(vision_model), "--server"],
            env,
            "vision",
            runtime.runner_startup_timeout_seconds,
        )
    try:
        with WaitIndicator(2, 2, "Load Language model"):
            language = HbmServer(
                language_runner_command(runtime) + ["--server"],
                env,
                "language",
                runtime.runner_startup_timeout_seconds,
            )
    except BaseException:
        close_runtime_resources(None, None, vision)
        raise

    last_request: tuple[str, Path, str] | None = None
    vision_cache: tuple[tuple[str, int, int], bytes, dict[str, object]] | None = None
    request_index = 0
    video_index = 0
    monitor: ResourceMonitor | None = None
    try:
        monitor = ResourceMonitor(
            visible=not args.no_dashboard,
            interval_seconds=runtime.telemetry_interval_seconds,
        )
        header_lines, startup_details = build_runtime_header(
            vision,
            language,
            args.generation_mode,
            args.max_new_tokens,
            show_runner_details=getattr(args, "show_runner_details", False),
        )
        monitor.start(header_lines)
        for line in startup_details:
            print(f"{DIM}{line}{RESET}")
        if current_image is not None:
            print(f"{GREEN}Image loaded{RESET}  {BLUE}{current_image}{RESET}")
    except BaseException:
        close_runtime_resources(monitor, language, vision)
        raise

    # The initialization guard above guarantees this before infer() is defined.
    assert monitor is not None

    def infer(
        image: Path,
        prompt: str,
        *,
        request_output_dir: Path | None = None,
        quiet: bool = False,
        remember: bool = True,
    ) -> dict[str, object]:
        nonlocal request_index, last_request, vision_cache
        if not image.is_file():
            raise FileNotFoundError(image)
        task_prompt = prompt
        normalized_prompt, task = normalize_prompt(task_prompt)
        request_index += 1
        if remember:
            last_request = ("image", image, prompt)
        started = time.monotonic()
        target_output_dir = (
            request_output_dir
            if request_output_dir is not None
            else output_dir / f"request_{request_index:04d}"
            if output_dir
            else None
        )
        paths = create_prediction_paths(
            runtime.layout_root,
            image,
            output_dir=target_output_dir,
        )
        with monitor.request_scope(), tempfile.TemporaryDirectory(
            prefix="locateanything-interactive-"
        ) as raw_dir:
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

            if not quiet:
                print(f"{BOLD}{MAGENTA}[Assistant] >>>{RESET} ", end="", flush=True)
            try:
                language_result, language_log = language.request([
                    "LAHBM/1", "RUN", str(request_index), str(token_path),
                    str(visual_output_path), str(generation_path),
                    str(args.max_new_tokens), args.generation_mode,
                    "1" if task == "object_detection" else "0",
                ], on_token=None if quiet else stream_token)
            except Exception:
                if not quiet:
                    print()
                raise
            language_fields = language_result.split("\t")
            if len(language_fields) < 8:
                raise RuntimeError(f"invalid Language response: {language_result}")
            decode_token_count = int(language_fields[4])
            prefill_tokens = int(language_fields[5])
            prefill_ms = float(language_fields[6])
            decode_ms = float(language_fields[7])
            response_executed_mode = (
                language_fields[8] if len(language_fields) > 8 else None
            )
            response_fallback_reason = (
                language_fields[9] if len(language_fields) > 9 else None
            )

            stop_reason, token_ids, file_executed_mode, file_fallback_reason = (
                read_generation(generation_path)
            )
            executed_mode = (
                response_executed_mode or file_executed_mode or args.generation_mode
            )
            fallback_reason = response_fallback_reason or file_fallback_reason
            text = decode_tokens(tokenizer_dir, token_ids, stream_tokenizer)
            if not quiet:
                if text.startswith(streamed_text) and len(text) > len(streamed_text):
                    print(text[len(streamed_text):], end="", flush=True)
                elif text != streamed_text:
                    print(
                        f"\n{BOLD}{MAGENTA}[Assistant final] >>>{RESET} "
                        f"{text}",
                        end="",
                        flush=True,
                    )
                print()
                for line in language_log:
                    if args.show_runner_details and line.startswith(
                        ("[profile]", "[pbd]", "[hybrid:")
                    ):
                        print(f"{DIM}{line}{RESET}")
            language_seconds = time.monotonic() - language_started

            postprocess_started = time.monotonic()
            output_complete = stop_reason == "im_end"
            raw_detections = (
                parse_detections(text, transform)
                if task != "object_detection" or output_complete
                else []
            )
            detections, suppressed_detections = postprocess_detections(
                raw_detections,
                task,
                iou_threshold=args.nms_iou,
                enabled=not args.no_nms,
            )
            points = parse_points(text, transform)
            annotated_image = None
            if detections or points:
                annotated_image = paths.annotated_image
                save_annotated_image(image, detections, points, annotated_image)
            postprocess_seconds = time.monotonic() - postprocess_started
            total_ms = (time.monotonic() - started) * 1000.0
            monitor.end_request()
            resource_summary = monitor.summary()
            if not quiet:
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
                    resource_summary,
                )
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
                "resources": resource_summary.as_dict(),
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
            prediction_data = {
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
                    "requested_mode": args.generation_mode,
                    "executed_mode": executed_mode,
                    "fallback_reason": fallback_reason,
                    "stop_reason": stop_reason,
                    "complete": output_complete,
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
                "resources": resource_summary.as_dict(),
                "runtime_log": str(paths.runtime_log),
            }
            paths.prediction.write_text(
                json.dumps(prediction_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if not quiet:
                print(f"{BOLD}{CYAN}Saved{RESET}")
                if annotated_image:
                    print(f"  {BOLD}Image{RESET}  {BLUE}{annotated_image}{RESET}")
                print(f"  {BOLD}JSON{RESET}   {BLUE}{paths.prediction}{RESET}")
                print()
            return prediction_data

    def infer_video(
        video_path: Path,
        prompt: str,
        info: VideoInfo,
        *,
        remember: bool = True,
    ) -> None:
        nonlocal last_request, video_index, vision_cache
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        normalized_prompt, task = normalize_prompt(prompt)
        video_index += 1
        if remember:
            last_request = ("video", video_path, prompt)
        vision_cache = None
        video_root = create_video_output_dir(
            runtime.layout_root,
            video_path,
            output_dir,
            sequence=video_index,
        )
        predictions_path = video_root / "predictions.jsonl"
        summary_path = video_root / "summary.json"
        output_video = video_root / "annotated.mp4"
        runtime_log = video_root / "logs" / "runtime.log"
        expected_frames = info.frame_count
        started = time.monotonic()
        total_boxes = 0
        total_points = 0
        processed_frames = 0

        try:
            with tempfile.TemporaryDirectory(prefix="locateanything-video-") as raw_dir:
                work_dir = Path(raw_dir)
                requests_dir = work_dir / "requests"
                requests_dir.mkdir()
                frame_path = work_dir / "frame.png"
                with (
                    VideoFrameReader(video_path, info) as frames,
                    VideoFrameWriter(output_video, video_path, info) as video_writer,
                    predictions_path.open("w", encoding="utf-8") as predictions,
                    runtime_log.open("w", encoding="utf-8") as combined_log,
                ):
                    for frame_index, rgb_frame in enumerate(frames, 1):
                        frame_started = time.monotonic()
                        Image.frombytes(
                            "RGB", (info.width, info.height), rgb_frame
                        ).save(frame_path, format="PNG")
                        vision_cache = None
                        frame_result = infer(
                            frame_path,
                            prompt,
                            request_output_dir=requests_dir / "frame",
                            quiet=True,
                            remember=False,
                        )
                        annotated_source = frame_result.get("annotated_image")
                        if annotated_source and Path(str(annotated_source)).is_file():
                            with Image.open(Path(str(annotated_source))) as annotated:
                                video_writer.write(annotated.convert("RGB").tobytes())
                        else:
                            video_writer.write(rgb_frame)

                        frame_log = frame_result.get("runtime_log")
                        if frame_log and Path(str(frame_log)).is_file():
                            combined_log.write(f"FRAME {frame_index}\n")
                            combined_log.write(
                                Path(str(frame_log)).read_text(
                                    encoding="utf-8", errors="replace"
                                )
                            )
                            combined_log.write("\n")

                        detections = frame_result.get("detections", [])
                        points = frame_result.get("points", [])
                        box_count = (
                            len(detections) if isinstance(detections, list) else 0
                        )
                        point_count = len(points) if isinstance(points, list) else 0
                        total_boxes += box_count
                        total_points += point_count
                        processed_frames += 1
                        source_frame_index = frame_index - 1
                        timestamp_seconds = (
                            source_frame_index / info.source_fps
                            if info.source_fps > 0
                            else 0.0
                        )
                        record = {
                            "frame_index": frame_index,
                            "source_frame_index": source_frame_index,
                            "timestamp_seconds": round(timestamp_seconds, 6),
                            "text": frame_result.get("text"),
                            "raw_detections": frame_result.get("raw_detections", []),
                            "detections": detections,
                            "suppressed_detections": frame_result.get(
                                "suppressed_detections", []
                            ),
                            "points": points,
                            "generation": frame_result.get("generation", {}),
                            "performance": frame_result.get("performance", {}),
                            "resources": frame_result.get("resources", {}),
                        }
                        predictions.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        predictions.flush()
                        frame_seconds = time.monotonic() - frame_started
                        print(
                            f"\r{BOLD}{CYAN}Video{RESET}  frame "
                            f"{GREEN}{frame_index}/{expected_frames}{RESET}  |  "
                            f"{timestamp_seconds:8.2f} s  |  "
                            f"boxes {GREEN}{box_count}{RESET}  |  "
                            f"{frame_seconds:.2f} s/frame",
                            end="",
                            flush=True,
                        )
                print()
                codec = video_writer.codec
        finally:
            vision_cache = None

        elapsed_seconds = time.monotonic() - started
        summary = {
            "schema_version": 1,
            "runtime": {
                "version": RUNTIME_VERSION,
                "model_type": runtime.model_type,
                "runtime_specification": runtime.specification(),
            },
            "video": {
                "source": str(video_path),
                "width": info.width,
                "height": info.height,
                "source_fps": round(info.source_fps, 6),
                "duration_seconds": round(info.duration_seconds, 6),
                "source_frame_count": info.frame_count,
                "source_frame_count_exact": info.frame_count_exact,
                "rotation_degrees": info.rotation_degrees,
                "has_audio": info.has_audio,
            },
            "inference": {
                "method": "independent_all_frames",
                "output_fps": round(info.output_fps, 6),
                "expected_frame_count": expected_frames,
                "processed_frame_count": processed_frames,
                "prompt": prompt,
                "normalized_prompt": normalized_prompt,
                "task": task,
                "generation_mode": args.generation_mode,
                "max_new_tokens": args.max_new_tokens,
                "total_boxes": total_boxes,
                "total_points": total_points,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "processing_fps": round(
                    processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                    6,
                ),
                "temporal_tracking": False,
            },
            "output": {
                "annotated_video": str(output_video),
                "predictions": str(predictions_path),
                "runtime_log": str(runtime_log),
                "video_codec": codec,
                "audio_preserved": video_writer.audio_preserved,
                "warning": video_writer.warning,
            },
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{BOLD}{CYAN}Video complete{RESET}")
        print(
            f"  Frames {GREEN}{processed_frames}{RESET}  |  "
            f"Boxes {GREEN}{total_boxes}{RESET}  |  "
            f"Points {GREEN}{total_points}{RESET}  |  "
            f"{elapsed_seconds:.2f} s"
        )
        print(f"  {BOLD}Video{RESET}   {BLUE}{output_video}{RESET}")
        print(f"  {BOLD}JSONL{RESET}   {BLUE}{predictions_path}{RESET}")
        print(f"  {BOLD}Summary{RESET} {BLUE}{summary_path}{RESET}")
        if video_writer.warning:
            print(f"  {YELLOW}Audio warning{RESET}  {video_writer.warning}")
        print()

    try:
        if current_image is not None and args.prompt:
            print(f"{BOLD}{BLUE}[User] <<<{RESET} {BLUE}{args.prompt}{RESET}")
            infer(current_image, args.prompt)
        while True:
            try:
                line = input(f"{BOLD}{BLUE}[User] <<<{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "exit":
                break
            if line in {"/help", "help", "?"}:
                print_command_help()
                continue
            if line == "reset":
                current_image = None
                current_video = None
                last_request = None
                vision_cache = None
                print(f"{GREEN}[Session]{RESET} media and cache cleared")
                continue
            if line == "regen":
                if last_request is None:
                    print(f"{YELLOW}[Session]{RESET} no previous request")
                else:
                    print(
                        f"{BOLD}{BLUE}[User] <<<{RESET} "
                        f"{BLUE}{last_request[2]}{RESET}"
                    )
                    if last_request[0] == "image":
                        infer(last_request[1], last_request[2], remember=False)
                    else:
                        infer_video(
                            last_request[1],
                            last_request[2],
                            probe_video(last_request[1]),
                            remember=False,
                        )
                continue
            if line.startswith("/image"):
                parts = shlex.split(line)
                if len(parts) != 2:
                    print(f"{YELLOW}Usage:{RESET} /image IMAGE_PATH")
                    continue
                candidate = Path(parts[1]).expanduser().resolve()
                if not candidate.is_file():
                    print(f"{RED}[Error]{RESET} image not found: {BLUE}{candidate}{RESET}")
                    continue
                current_image = candidate
                current_video = None
                vision_cache = None
                print(f"{GREEN}Image loaded{RESET}  {BLUE}{current_image}{RESET}")
                continue
            if line.startswith("/video"):
                parts = shlex.split(line)
                if len(parts) != 2:
                    print(f"{YELLOW}Usage:{RESET} /video VIDEO_PATH")
                    continue
                candidate = Path(parts[1]).expanduser().resolve()
                if not candidate.is_file():
                    print(f"{RED}[Error]{RESET} video not found: {BLUE}{candidate}{RESET}")
                    continue
                try:
                    info = probe_video(candidate)
                except (RuntimeError, ValueError) as error:
                    print(f"{RED}[Error]{RESET} {error}")
                    continue
                current_video = (candidate, info)
                current_image = None
                vision_cache = None
                print(f"{GREEN}Video loaded{RESET}  {BLUE}{candidate}{RESET}")
                frame_count = (
                    str(info.frame_count)
                    if info.frame_count_exact
                    else f"~{info.frame_count}"
                )
                print(
                    f"  {info.width}x{info.height}  |  "
                    f"source {info.source_fps:.3f} FPS  |  "
                    f"duration {info.duration_seconds:.2f} s  |  "
                    f"process all {frame_count} frames"
                )
                continue
            if current_image is None and current_video is None:
                print(
                    f"{YELLOW}[Session]{RESET} load media first with "
                    f"{BLUE}/image IMAGE_PATH{RESET} or "
                    f"{BLUE}/video VIDEO_PATH{RESET}"
                )
                continue
            try:
                if current_video is not None:
                    infer_video(current_video[0], line, current_video[1])
                else:
                    assert current_image is not None
                    infer(current_image, line)
            except Exception as error:
                print(f"{RED}[Request failed]{RESET} {error}", file=sys.stderr)
    finally:
        close_runtime_resources(monitor, language, vision)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"{RED}[FAIL]{RESET} {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
