#!/usr/bin/env python3
"""Repeat and monitor an S600 runtime command without inventing metrics.

The collector uses only the Python standard library. It writes one immutable
log per run, a run-level JSONL file, a resource-sample JSONL file, and an
aggregate JSON summary. Runtime-reported metrics and derived metrics carry an
explicit provenance/status field.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


SCHEMA_VERSION = "locateanything.s600-benchmark.v1"
PERF_RE = re.compile(
    r"\[perf\]\s+vit_infer=(?P<vit_infer>[-+0-9.eE]+)ms\s+"
    r"prefill_tokens=(?P<prefill_tokens>\d+)\s+"
    r"prefill_tps=(?P<prefill_tps>[-+0-9.eE]+)\s+"
    r"decode_tokens=(?P<decode_tokens>\d+)\s+"
    r"decode_tps=(?P<decode_tps>[-+0-9.eE]+)\s+"
    r"ttft=(?P<ttft>[-+0-9.eE]+)ms\s+"
    r"tpot=(?P<tpot>[-+0-9.eE]+)ms\s+"
    r"e2e=(?P<e2e>[-+0-9.eE]+)ms"
)
CLI_PERF_RE = re.compile(
    r"\[perf\]\s+vision=(?P<vision>[-+0-9.eE]+)ms\s+"
    r"language=(?P<language>[-+0-9.eE]+)ms\s+"
    r"e2e=(?P<e2e>[-+0-9.eE]+)ms"
)
BOX_RE = re.compile(r"<box>((?:<\d{1,4}>){2}(?:(?:<\d{1,4}>){2})?)</box>")
COORD_RE = re.compile(r"<(\d{1,4})>")
INFER_RET_RE = re.compile(r"\[demo\]\s+xlm_infer ret=(-?\d+)")
VENDOR_SECTION_RE = re.compile(r"^\s*(temperature|voltage)\s*-+>\s*$", re.IGNORECASE)
VENDOR_VALUE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9_. /-]+?)\s*[:=]\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<unit>m?°?C|℃|[mu]?V)\b",
    re.IGNORECASE,
)
VENDOR_DIRECT_RE = re.compile(
    r"\b(?P<kind>temperature|temp|voltage)\b"
    r"(?:\s+(?P<label>[A-Za-z0-9_.-]+))?\s*[:=]\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<unit>m?°?C|℃|[mu]?V)\b",
    re.IGNORECASE,
)
BPU_RATIO_RE = re.compile(
    r"\b(?P<name>bpu[0-3])\b[^\n]*?\bratio\b\s*[:=]?\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<percent>%)?",
    re.IGNORECASE,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def measured(value: Any, source: str, unit: str | None = None) -> dict[str, Any]:
    result = {"value": value, "status": "measured", "source": source}
    if unit:
        result["unit"] = unit
    return result


def derived(value: Any, source: str, unit: str | None = None) -> dict[str, Any]:
    result = {"value": value, "status": "derived", "source": source}
    if unit:
        result["unit"] = unit
    return result


def unavailable(reason: str, unit: str | None = None) -> dict[str, Any]:
    result = {"value": None, "status": "unavailable", "reason": reason}
    if unit:
        result["unit"] = unit
    return result


def parse_runtime_performance(log_text: str) -> dict[str, float | int] | None:
    matches = list(PERF_RE.finditer(log_text))
    if matches:
        values = matches[-1].groupdict()
        return {
            "vit_infer_ms": float(values["vit_infer"]),
            "prefill_tokens": int(values["prefill_tokens"]),
            "prefill_tps": float(values["prefill_tps"]),
            "decode_tokens": int(values["decode_tokens"]),
            "decode_tps": float(values["decode_tps"]),
            "ttft_ms": float(values["ttft"]),
            "tpot_ms": float(values["tpot"]),
            "e2e_ms": float(values["e2e"]),
        }
    cli_matches = list(CLI_PERF_RE.finditer(log_text))
    if not cli_matches:
        return None
    values = cli_matches[-1].groupdict()
    return {
        "vit_infer_ms": float(values["vision"]),
        "language_ms": float(values["language"]),
        "e2e_ms": float(values["e2e"]),
    }


def final_response(log_text: str) -> str | None:
    marker = "[callback] END:"
    lines = [line.split(marker, 1)[1].strip() for line in log_text.splitlines() if marker in line]
    if lines:
        return lines[-1]
    matches = re.findall(r"\[Assistant\]\s*>>>\s*(.+)", log_text)
    return matches[-1].strip() if matches else None


def count_structured_boxes(response: str | None) -> int | None:
    if response is None:
        return None
    count = 0
    for match in BOX_RE.finditer(response):
        coordinates = [int(value) for value in COORD_RE.findall(match.group(1))]
        if len(coordinates) in {2, 4} and all(0 <= value <= 1000 for value in coordinates):
            count += 1
    return count


def safe_ratio_ms(tokens: int, tps: float) -> float | None:
    if tokens < 0 or not math.isfinite(tps) or tps <= 0:
        return None
    return tokens * 1000.0 / tps


def build_runtime_metrics(log_text: str, wall_ms: float) -> dict[str, dict[str, Any]]:
    perf = parse_runtime_performance(log_text)
    metrics: dict[str, dict[str, Any]] = {
        "wall_latency_ms": measured(wall_ms, "benchmark subprocess monotonic clock", "ms")
    }
    if perf is None:
        reason = "no complete Runtime performance record in output"
        for name, unit in (
            ("vision_ms", "ms"), ("prefill_tokens", "tokens"),
            ("prefill_tps", "tokens/s"), ("prefill_ms", "ms"),
            ("decode_tokens", "tokens"), ("decode_tps", "tokens/s"),
            ("decode_ms", "ms"), ("ttft_ms", "ms"),
            ("tpot_ms", "ms"), ("language_ms", "ms"),
            ("runtime_e2e_ms", "ms"),
        ):
            metrics[name] = unavailable(reason, unit)
    else:
        runtime_source = (
            "xlm_result_s.performance"
            if "prefill_tokens" in perf
            else "LocateAnything Runtime output"
        )
        direct = {}
        for output_name, key, unit in (
            ("vision_ms", "vit_infer_ms", "ms"),
            ("language_ms", "language_ms", "ms"),
            ("prefill_tokens", "prefill_tokens", "tokens"),
            ("prefill_tps", "prefill_tps", "tokens/s"),
            ("decode_tokens", "decode_tokens", "tokens"),
            ("decode_tps", "decode_tps", "tokens/s"),
            ("ttft_ms", "ttft_ms", "ms"),
            ("tpot_ms", "tpot_ms", "ms"),
            ("runtime_e2e_ms", "e2e_ms", "ms"),
        ):
            if key in perf:
                direct[output_name] = (perf[key], unit)
            else:
                metrics[output_name] = unavailable(f"{key} is not reported by this Runtime", unit)
        for name, (value, unit) in direct.items():
            metrics[name] = measured(value, runtime_source, unit)
        for phase in ("prefill", "decode"):
            if f"{phase}_tokens" not in perf or f"{phase}_tps" not in perf:
                metrics[f"{phase}_ms"] = unavailable(
                    f"{phase} token count or speed is not reported by this Runtime", "ms"
                )
                continue
            duration = safe_ratio_ms(int(perf[f"{phase}_tokens"]), float(perf[f"{phase}_tps"]))
            metrics[f"{phase}_ms"] = (
                derived(duration, f"{phase}_tokens / {phase}_tps", "ms")
                if duration is not None
                else unavailable(f"{phase}_tps is absent, zero, or invalid", "ms")
            )

    response = final_response(log_text)
    box_count = count_structured_boxes(response)
    if box_count is None:
        metrics["structured_boxes"] = unavailable("no [callback] END response", "boxes")
        metrics["bps"] = unavailable("structured response and runtime e2e are required", "boxes/s")
    else:
        metrics["structured_boxes"] = measured(box_count, "validated <box> coordinate frames", "boxes")
        e2e = metrics.get("runtime_e2e_ms", {}).get("value")
        if box_count > 0 and isinstance(e2e, (int, float)) and e2e > 0:
            metrics["bps"] = derived(box_count * 1000.0 / e2e, "structured_boxes / runtime_e2e_ms", "boxes/s")
        else:
            metrics["bps"] = unavailable("no structured box or positive runtime e2e", "boxes/s")
    return metrics


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return None


def numeric_file(path: Path) -> float | None:
    text = read_text(path)
    if text is None:
        return None
    try:
        value = float(text.split()[0])
    except (ValueError, IndexError):
        return None
    return value if math.isfinite(value) else None


def normalized_metric_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return name or fallback


def temperature_c(value: float, unit: str) -> float | None:
    normalized = unit.lower().replace("°", "")
    if normalized in {"c", "℃"}:
        return value
    if normalized == "mc":
        return value / 1000.0
    return None


def voltage_v(value: float, unit: str) -> float | None:
    normalized = unit.lower()
    if normalized == "v":
        return value
    if normalized == "mv":
        return value / 1000.0
    if normalized == "uv":
        return value / 1_000_000.0
    return None


def parse_vendor_status(text: str) -> dict[str, Any]:
    """Parse hrut_somstatus fields while retaining ambiguous ratio units."""
    result: dict[str, Any] = {"temperature_c": {}, "voltage_v": {}, "bpu_ratio": {}}
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        header = VENDOR_SECTION_RE.match(line)
        if header:
            section = header.group(1).lower()
            continue
        if line.endswith("-->"):
            section = None

        for match in BPU_RATIO_RE.finditer(line):
            result["bpu_ratio"][match.group("name").lower()] = {
                "value": float(match.group("value")),
                "unit": "%" if match.group("percent") else "vendor_ratio",
            }

        value_match = VENDOR_VALUE_RE.match(line)
        if section and value_match:
            value = float(value_match.group("value"))
            unit = value_match.group("unit")
            label = normalized_metric_name(value_match.group("label"), section)
            converted = temperature_c(value, unit) if section == "temperature" else voltage_v(value, unit)
            if converted is not None:
                result[f"{section}_c" if section == "temperature" else "voltage_v"][label] = converted
            continue

        direct = VENDOR_DIRECT_RE.search(line)
        if direct:
            kind = direct.group("kind").lower()
            label = normalized_metric_name(direct.group("label") or "soc", "soc")
            value = float(direct.group("value"))
            unit = direct.group("unit")
            if kind in {"temperature", "temp"}:
                converted = temperature_c(value, unit)
                if converted is not None:
                    result["temperature_c"][label] = converted
            else:
                converted = voltage_v(value, unit)
                if converted is not None:
                    result["voltage_v"][label] = converted
    return result


def collect_vendor_status(command: Path | None, timeout_s: float) -> dict[str, Any]:
    if command is None:
        return {
            "status": "unavailable",
            "reason": "vendor status command disabled, absent, or not executable",
            "temperature_c": {}, "voltage_v": {}, "bpu_ratio": {},
        }
    try:
        completed = subprocess.run(
            [str(command)], capture_output=True, text=True, errors="replace",
            timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "unavailable", "reason": f"vendor status command failed: {exc}",
            "command": str(command), "temperature_c": {}, "voltage_v": {},
            "bpu_ratio": {},
        }
    raw_output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        return {
            "status": "unavailable",
            "reason": f"vendor status command exited with {completed.returncode}",
            "command": str(command), "exit_code": completed.returncode,
            "raw_output": raw_output, "temperature_c": {}, "voltage_v": {},
            "bpu_ratio": {},
        }
    parsed = parse_vendor_status(raw_output)
    parsed.update({"command": str(command), "exit_code": 0, "raw_output": raw_output})
    if any(parsed[name] for name in ("temperature_c", "voltage_v", "bpu_ratio")):
        parsed["status"] = "measured"
    else:
        parsed.update({
            "status": "unavailable",
            "reason": "vendor command succeeded but no supported fields were parsed",
        })
    return parsed


def resolve_vendor_status_command(raw: str | None, disabled: bool) -> Path | None:
    if disabled:
        return None
    path = Path(raw or "/usr/hobot/bin/hrut_somstatus")
    return path if path.is_file() and os.access(path, os.X_OK) else None


def discover_sensors() -> dict[str, list[tuple[str, Path]]]:
    thermal: list[tuple[str, Path]] = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        label = read_text(path.parent / "type") or "unknown"
        thermal.append((f"{path.parent.name}:{label}", path))
    power: list[tuple[str, Path]] = []
    for path in sorted(Path("/sys/class/hwmon").glob("hwmon*/power*_input")):
        chip = read_text(path.parent / "name") or path.parent.name
        power.append((f"{path.parent.name}:{chip}:{path.stem}", path))
    return {"thermal": thermal, "power": power}


def parse_custom_metrics(values: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"metric must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in names or not raw_path:
            raise ValueError(f"invalid or duplicate metric: {value}")
        names.add(name)
        result.append((name, Path(raw_path)))
    return result


def process_rows() -> dict[int, tuple[int, int, int]]:
    """Return pid -> (ppid, cpu_ticks, rss_bytes), tolerating process exit."""
    rows: dict[int, tuple[int, int, int]] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
            stat = (proc_dir / "stat").read_text().strip()
            tail = stat[stat.rfind(")") + 2 :].split()
            ppid = int(tail[1])
            ticks = int(tail[11]) + int(tail[12])
            rss = max(0, int(tail[21])) * page_size
            rows[pid] = (ppid, ticks, rss)
        except (OSError, ValueError, IndexError):
            continue
    return rows


def descendants(root_pid: int, rows: dict[int, tuple[int, int, int]]) -> set[int]:
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in rows.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def sample_resources(
    pid: int,
    elapsed_s: float,
    sensors: dict[str, list[tuple[str, Path]]],
    custom_metrics: list[tuple[str, Path]],
    vendor_status_command: Path | None,
    vendor_status_timeout_s: float,
    previous: tuple[float, int] | None,
) -> tuple[dict[str, Any], tuple[float, int] | None]:
    rows = process_rows() if Path("/proc").exists() else {}
    pids = descendants(pid, rows)
    live = [rows[p] for p in pids if p in rows]
    total_ticks = sum(row[1] for row in live)
    rss_bytes = sum(row[2] for row in live)
    cpu_percent = None
    if previous is not None and elapsed_s > previous[0]:
        ticks_per_s = os.sysconf("SC_CLK_TCK")
        cpu_percent = max(0.0, (total_ticks - previous[1]) / ticks_per_s / (elapsed_s - previous[0]) * 100.0)
    process = {
        "process_count": len(live),
        "rss_bytes": rss_bytes if live else None,
        "cpu_percent": cpu_percent,
    }
    thermal: dict[str, float] = {}
    for name, path in sensors["thermal"]:
        value = numeric_file(path)
        if value is not None:
            thermal[name] = value / 1000.0 if abs(value) >= 1000 else value
    power: dict[str, float] = {}
    for name, path in sensors["power"]:
        value = numeric_file(path)
        if value is not None:
            power[name] = value / 1_000_000.0
    custom = {name: numeric_file(path) for name, path in custom_metrics}
    vendor_status = collect_vendor_status(vendor_status_command, vendor_status_timeout_s)
    return {
        "elapsed_s": elapsed_s,
        "process": process,
        "thermal_c": thermal,
        "power_w": power,
        "custom_metrics": custom,
        "vendor_status": vendor_status,
    }, (elapsed_s, total_ticks) if live else previous


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def aggregate(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def summarize_runs(runs: list[dict[str, Any]], semantic_configured: bool) -> dict[str, Any]:
    measured_runs = [run for run in runs if run["run_kind"] == "measured"]
    process_ok = [run for run in measured_runs if run["success"]["process"]]
    valid_runs = [
        run for run in measured_runs
        if run["success"]["process"]
        and (not semantic_configured or run["success"]["semantic"] is True)
    ]
    metric_names = sorted({name for run in measured_runs for name in run["metrics"]})
    metrics: dict[str, Any] = {}
    for name in metric_names:
        vals = [
            float(run["metrics"][name]["value"])
            for run in valid_runs
            if run["metrics"].get(name, {}).get("status") in {"measured", "derived"}
            and isinstance(run["metrics"][name].get("value"), (int, float))
        ]
        metrics[name] = aggregate(vals)
        statuses = sorted({run["metrics"].get(name, {}).get("status", "missing") for run in measured_runs})
        metrics[name]["statuses"] = statuses
    semantic_values = [run["success"]["semantic"] for run in measured_runs if run["success"]["semantic"] is not None]
    return {
        "run_count": len(measured_runs),
        "process_success_count": len(process_ok),
        "process_success_rate": len(process_ok) / len(measured_runs) if measured_runs else None,
        "valid_run_count_for_metrics": len(valid_runs),
        "metric_population": "process-successful runs that also pass the configured semantic criterion",
        "semantic_success_count": sum(bool(value) for value in semantic_values) if semantic_configured else None,
        "semantic_evaluated_count": len(semantic_values),
        "semantic_success_rate": (sum(bool(value) for value in semantic_values) / len(semantic_values)) if semantic_values else None,
        "metrics": metrics,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_records(paths: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for raw in paths:
        path = Path(raw).resolve()
        record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
        records.append(record)
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run_once(
    command: list[str], run_id: str, run_kind: str, output_dir: Path,
    interval_s: float, timeout_s: float, semantic_re: re.Pattern[str] | None,
    sensors: dict[str, list[tuple[str, Path]]], custom_metrics: list[tuple[str, Path]],
    vendor_status_command: Path | None, vendor_status_timeout_s: float,
    samples_path: Path,
) -> dict[str, Any]:
    log_path = output_dir / "logs" / f"{run_id}.log"
    started_utc = utc_now()
    started = time.monotonic()
    timed_out = False
    samples: list[dict[str, Any]] = []
    previous: tuple[float, int] | None = None
    with log_path.open("wb") as log_handle:
        try:
            proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            proc = None
            log_handle.write(f"[benchmark] command launch failed: {exc}\n".encode("utf-8", errors="replace"))
        if proc is not None:
            while proc.poll() is None:
                sample_started = time.monotonic()
                elapsed = time.monotonic() - started
                sample, previous = sample_resources(
                    proc.pid, elapsed, sensors, custom_metrics,
                    vendor_status_command, vendor_status_timeout_s, previous,
                )
                sample.update({"schema_version": SCHEMA_VERSION, "run_id": run_id, "timestamp_utc": utc_now()})
                samples.append(sample)
                append_jsonl(samples_path, sample)
                if elapsed >= timeout_s:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except (OSError, AttributeError):
                        proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except (OSError, AttributeError):
                            proc.kill()
                    break
                sample_cost = time.monotonic() - sample_started
                time.sleep(max(0.0, interval_s - sample_cost))
            exit_code = proc.wait()
        else:
            exit_code = 127
    wall_ms = (time.monotonic() - started) * 1000.0
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    infer_returns = [int(value) for value in INFER_RET_RE.findall(log_text)]
    vendor_success = bool(infer_returns) and infer_returns[-1] == 0 and "[callback] END:" in log_text
    locateanything_success = "STATUS: COMPLETE" in log_text and "[Assistant] >>>" in log_text
    process_success = not timed_out and exit_code == 0 and (vendor_success or locateanything_success)
    response = final_response(log_text)
    semantic_success = bool(semantic_re.search(response or "")) if semantic_re else None
    rss_values = [sample["process"]["rss_bytes"] for sample in samples if sample["process"]["rss_bytes"] is not None]
    cpu_values = [sample["process"]["cpu_percent"] for sample in samples if sample["process"]["cpu_percent"] is not None]
    temps = [value for sample in samples for value in sample["thermal_c"].values()]
    powers = [value for sample in samples for value in sample["power_w"].values()]
    vendor_temperatures = [
        value for sample in samples
        for value in sample["vendor_status"].get("temperature_c", {}).values()
    ]
    vendor_voltages = [
        value for sample in samples
        for value in sample["vendor_status"].get("voltage_v", {}).values()
    ]
    resources = {
        "sample_count": len(samples),
        "peak_process_rss_bytes": measured(max(rss_values), "/proc process tree sampling", "bytes") if rss_values else unavailable("no readable /proc samples", "bytes"),
        "mean_process_cpu_percent": measured(statistics.fmean(cpu_values), "/proc process tree CPU ticks", "% of one CPU") if cpu_values else unavailable("fewer than two readable /proc samples", "% of one CPU"),
        "peak_temperature_c": measured(max(temps), "/sys/class/thermal", "C") if temps else unavailable("no readable thermal_zone temp nodes", "C"),
        "peak_power_sensor_w": measured(max(powers), "largest individual /sys/class/hwmon power*_input sample", "W") if powers else unavailable("no readable hwmon power input nodes", "W"),
        "vendor_peak_temperature_c": measured(max(vendor_temperatures), "hrut_somstatus temperature fields", "C") if vendor_temperatures else unavailable("no parsed vendor temperature samples", "C"),
        "vendor_voltage_range_v": measured({"min": min(vendor_voltages), "max": max(vendor_voltages)}, "hrut_somstatus voltage fields", "V") if vendor_voltages else unavailable("no parsed vendor voltage samples", "V"),
    }
    bpu_peaks: dict[str, float] = {}
    for name, _ in custom_metrics:
        values = [sample["custom_metrics"][name] for sample in samples if sample["custom_metrics"].get(name) is not None]
        if values:
            bpu_peaks[name] = max(values)
    vendor_bpu_peaks: dict[str, dict[str, Any]] = {}
    for core in ("bpu0", "bpu1", "bpu2", "bpu3"):
        observations = [
            sample["vendor_status"]["bpu_ratio"][core]
            for sample in samples if core in sample["vendor_status"].get("bpu_ratio", {})
        ]
        if observations:
            units = sorted({item["unit"] for item in observations})
            vendor_bpu_peaks[core] = {
                "value": max(float(item["value"]) for item in observations),
                "unit": units[0] if len(units) == 1 else "mixed",
            }
    resources["bpu_utilization"] = (
        measured(
            {"vendor_hrut_somstatus": vendor_bpu_peaks, "explicit_paths": bpu_peaks},
            "hrut_somstatus ratio fields and/or user-specified --bpu-metric paths",
        )
        if vendor_bpu_peaks or bpu_peaks
        else unavailable("no parsed hrut_somstatus ratio or readable --bpu-metric path")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "command": command,
        "log_path": str(log_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "success": {
            "process": process_success,
            "semantic": semantic_success,
            "semantic_criterion": semantic_re.pattern if semantic_re else None,
        },
        "metrics": build_runtime_metrics(log_text, wall_ms),
        "resources": resources,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="measured repetitions")
    parser.add_argument("--warmup", type=int, default=1, help="warm-up repetitions excluded from aggregates")
    parser.add_argument("--interval", type=float, default=0.2, help="resource sampling interval in seconds")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-run timeout in seconds")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--semantic-regex", help="optional regex that defines semantic success")
    parser.add_argument("--artifact", action="append", default=[], help="file to checksum in summary; repeatable")
    parser.add_argument("--bpu-metric", action="append", default=[], metavar="NAME=PATH", help="board-specific BPU utilization path; repeatable")
    parser.add_argument("--vendor-status-command", default="/usr/hobot/bin/hrut_somstatus", help="hrut_somstatus-compatible command path")
    parser.add_argument("--no-vendor-status", action="store_true", help="disable automatic vendor status sampling")
    parser.add_argument("--vendor-status-timeout", type=float, default=1.0, help="timeout per vendor status sample in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("a benchmark command is required after --")
    if args.runs < 1 or args.warmup < 0 or args.interval <= 0 or args.timeout <= 0 or args.vendor_status_timeout <= 0:
        raise SystemExit("runs >= 1, warmup >= 0, intervals/timeouts > 0 are required")
    try:
        custom_metrics = parse_custom_metrics(args.bpu_metric)
        semantic_re = re.compile(args.semantic_regex) if args.semantic_regex else None
    except (ValueError, re.error) as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to use non-empty benchmark directory: {output_dir}")
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.jsonl"
    samples_path = output_dir / "resource_samples.jsonl"
    for path in (runs_path, samples_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing benchmark evidence: {path}")
        path.touch(exist_ok=False)
    sensors = discover_sensors()
    vendor_status_command = resolve_vendor_status_command(
        args.vendor_status_command, args.no_vendor_status,
    )
    runs: list[dict[str, Any]] = []
    total = args.warmup + args.runs
    for index in range(total):
        kind = "warmup" if index < args.warmup else "measured"
        ordinal = index + 1 if kind == "warmup" else index - args.warmup + 1
        run_id = f"{kind}-{ordinal:04d}"
        print(f"[{index + 1}/{total}] {run_id}", flush=True)
        record = run_once(
            command, run_id, kind, output_dir, args.interval, args.timeout,
            semantic_re, sensors, custom_metrics, vendor_status_command,
            args.vendor_status_timeout, samples_path,
        )
        runs.append(record)
        append_jsonl(runs_path, record)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "platform": {"node": platform.node(), "system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "protocol": {
            "command": command, "warmup_runs": args.warmup, "measured_runs": args.runs,
            "sample_interval_s": args.interval, "timeout_s": args.timeout,
            "vendor_status_timeout_s": args.vendor_status_timeout,
            "semantic_regex": args.semantic_regex,
            "metric_provenance": "measured and derived statuses are not interchangeable",
        },
        "artifacts": artifact_records(args.artifact),
        "sensor_inventory": {
            "thermal": [{"name": name, "path": str(path)} for name, path in sensors["thermal"]],
            "power": [{"name": name, "path": str(path)} for name, path in sensors["power"]],
            "custom": [{"name": name, "path": str(path), "exists": path.is_file()} for name, path in custom_metrics],
            "vendor_status": {
                "requested_command": args.vendor_status_command,
                "resolved_command": str(vendor_status_command) if vendor_status_command else None,
                "enabled": not args.no_vendor_status,
                "available": vendor_status_command is not None,
            },
        },
        "aggregate": summarize_runs(runs, semantic_re is not None),
        "evidence_files": {"runs_jsonl": str(runs_path), "resource_samples_jsonl": str(samples_path), "log_dir": str(output_dir / "logs")},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")
    process_ok = summary["aggregate"]["process_success_rate"] == 1.0
    semantic_ok = semantic_re is None or summary["aggregate"]["semantic_success_rate"] == 1.0
    return 0 if process_ok and semantic_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
