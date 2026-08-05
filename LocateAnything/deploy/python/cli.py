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
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

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
    pad_visible,
    truncate_visible,
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
    unwrap_box_command,
)
from telemetry import ResourceMonitor, ResourceSummary, resource_summary_lines


VERSION = RUNTIME_VERSION


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
    print(f"{BOLD}{CYAN}Predictions{RESET}")
    label_text = ", ".join(labels) if labels else "(none)"
    print(f"  {BOLD}Labels{RESET}  {MAGENTA}{label_text}{RESET}")
    print(f"  {BOLD}Boxes{RESET}   {GREEN}{len(detections)}{RESET}")
    for index, item in enumerate(detections, 1):
        print(
            f"    {GREEN}{index}.{RESET} {BLUE}{item['label']!r}{RESET}  "
            f"normalized={YELLOW}{item['bbox_profile_1000']}{RESET}  "
            f"pixels={YELLOW}{item['bbox_xyxy']}{RESET}"
        )
    print(f"  {BOLD}Points{RESET}  {GREEN}{len(points)}{RESET}")
    for index, item in enumerate(points, 1):
        print(
            f"    {GREEN}{index}.{RESET} {BLUE}{item['label']!r}{RESET}  "
            f"normalized={YELLOW}{item['point_profile_1000']}{RESET}  "
            f"pixels={YELLOW}{item['point_xy']}{RESET}"
        )


def build_runtime_header(
    vision: HbmServer,
    language: HbmServer,
    runtime: RuntimeConfig,
    generation_mode: str,
    max_new_tokens: int,
    show_runner_details: bool = False,
) -> tuple[list[str], list[str]]:
    def command(value: str, description: str, color: str) -> str:
        label = pad_visible(f"  {color}{value}{RESET}", 32)
        return f"{label}{DIM}{description}{RESET}"

    def menu_rows(items: list[tuple[str, str, str]], columns: int) -> list[str]:
        entries = [command(*item) for item in items]
        if columns < 88:
            return entries
        column_width = max(42, (columns - 2) // 2)
        rows: list[str] = []
        midpoint = (len(entries) + 1) // 2
        for index in range(midpoint):
            left = truncate_visible(entries[index], column_width)
            right_index = index + midpoint
            right = entries[right_index] if right_index < len(entries) else ""
            rows.append(f"{pad_visible(left, column_width)}  {right}")
        return rows

    details: list[str] = []
    if show_runner_details:
        printed: set[str] = set()
        for line in vision.startup_output + language.startup_output:
            if line not in printed and any(
                marker in line for marker in ("[UCP]", "[DNN]", "[BPU]")
            ):
                details.append(line)
                printed.add(line)

    columns = shutil.get_terminal_size((120, 32)).columns
    core_text = ",".join(str(core) for core in runtime.bpu_cores)
    header = banner_lines()
    header.extend((
        f"{BOLD}{GREEN}Ready{RESET}  {BOLD}S600/Nash-P{RESET}  |  "
        f"{runtime.model_type}  |  BPU {GREEN}{core_text}{RESET}  |  "
        f"v{RUNTIME_VERSION}",
        f"{BOLD}{CYAN}Input{RESET} {GREEN}{runtime.image_width}x{runtime.image_height}{RESET}"
        f"  |  Graphs {MAGENTA}{runtime.language_graph_set}{RESET}"
        f"  |  Generation {MAGENTA}{generation_mode}{RESET}"
        f"  |  Max tokens {GREEN}{max_new_tokens}{RESET}",
        f"{BOLD}{CYAN}Tasks{RESET}",
    ))
    header.extend(menu_rows([
        ("/detect cat,dog", "目标检测", GREEN),
        ("/ground <phrase>", "指代表达，多目标", MAGENTA),
        ("/ground_single <phrase>", "指代表达，单目标", MAGENTA),
        ("/gui <element>", "GUI 点定位", BLUE),
        ("/gui_box <element>", "GUI 框定位", BLUE),
        ("/text", "文本 OCR", YELLOW),
        ("/ground_text <text>", "指定文本定位", YELLOW),
        ("/layout title,table,figure", "文档版面分析", CYAN),
        ("/point <target>", "通用点定位", CYAN),
        ("/box /detect cat", "保存预测框图片", GREEN),
    ], columns))
    header.append(f"{BOLD}{CYAN}Session{RESET}")
    header.extend(menu_rows([
        ("/image <image_path>", "加载图片", BLUE),
        ("regen", "重跑上次请求", BLUE),
        ("reset", "清除当前图片与缓存", BLUE),
    ], columns))
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

    last_request: tuple[Path, str] | None = None
    vision_cache: tuple[tuple[str, int, int], bytes, dict[str, object]] | None = None
    request_index = 0
    monitor: ResourceMonitor | None = None
    try:
        monitor = ResourceMonitor(
            visible=not args.no_dashboard,
            interval_seconds=runtime.telemetry_interval_seconds,
        )
        header_lines, startup_details = build_runtime_header(
            vision,
            language,
            runtime,
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
        with monitor.request_scope(), tempfile.TemporaryDirectory(
            prefix="locateanything-interactive-"
        ) as raw_dir:
            work_dir = Path(raw_dir)
            monitor.set_stage(1, 3, "Vision")
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

            monitor.set_stage(2, 3, "Language")
            language_started = time.monotonic()
            tokens = tokenize_prompt(tokenizer_dir, task_prompt, stream_tokenizer)
            tokens.tofile(token_path)

            streamed_ids: list[int] = []
            streamed_text = ""

            def stream_token(token: int) -> None:
                nonlocal streamed_text
                monitor.observe_token()
                streamed_ids.append(token)
                decoded = stream_tokenizer.decode(
                    streamed_ids, skip_special_tokens=False
                ).rstrip("\ufffd")
                if decoded.startswith(streamed_text):
                    print(
                        f"{MAGENTA}{decoded[len(streamed_text):]}{RESET}",
                        end="",
                        flush=True,
                    )
                    streamed_text = decoded

            if normalized_prompt != prompt:
                print(f"{CYAN}[Task]{RESET} {MAGENTA}{task}{RESET}")
                print(f"{CYAN}[Prompt]{RESET} {BLUE}{normalized_prompt}{RESET}")
            print(f"{BOLD}{MAGENTA}[Assistant] >>>{RESET} ", end="", flush=True)
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
                print(
                    f"{MAGENTA}{text[len(streamed_text):]}{RESET}",
                    end="",
                    flush=True,
                )
            elif text != streamed_text:
                print(
                    f"\n{BOLD}{MAGENTA}[Assistant final] >>>{RESET} "
                    f"{MAGENTA}{text}{RESET}",
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

            monitor.set_stage(3, 3, "Postprocess")
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
            monitor.end_request()
            resource_summary = monitor.summary()
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
            if suppressed_detections:
                print(
                    f"  {YELLOW}NMS removed{RESET} "
                    f"{GREEN}{len(suppressed_detections)}{RESET} same-label box(es) "
                    f"at IoU >= {YELLOW}{args.nms_iou:.2f}{RESET}"
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
                    "resources": resource_summary.as_dict(),
                    "runtime_log": str(paths.runtime_log),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{BOLD}{CYAN}Saved{RESET}")
            if annotated_image:
                print(f"  {BOLD}Annotated image{RESET}  {BLUE}{annotated_image}{RESET}")
            print(f"  {BOLD}Prediction{RESET}       {BLUE}{paths.prediction}{RESET}")
            print(f"  {BOLD}Timings{RESET}          {BLUE}{paths.timings}{RESET}")
            print(f"  {BOLD}Run directory{RESET}    {BLUE}{paths.root}{RESET}")

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
            if line == "reset":
                current_image = None
                last_request = None
                vision_cache = None
                print(f"{GREEN}[Session]{RESET} image and cache cleared")
                continue
            if line == "regen":
                if last_request is None:
                    print(f"{YELLOW}[Session]{RESET} no previous request")
                else:
                    print(
                        f"{BOLD}{BLUE}[User] <<<{RESET} "
                        f"{BLUE}{last_request[1]}{RESET}"
                    )
                    infer(*last_request)
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
                print(f"{GREEN}Image loaded{RESET}  {BLUE}{current_image}{RESET}")
                continue
            if current_image is None:
                print(
                    f"{YELLOW}[Session]{RESET} load an image first with "
                    f"{BLUE}/image IMAGE_PATH{RESET}"
                )
                continue
            try:
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
