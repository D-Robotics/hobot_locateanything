#!/usr/bin/env python3
"""Audit and enforce the compiler-host environment before long-running work.

The command writes one JSON document to stdout. Formal Prepare, Calibrate, and
Build profiles fail closed when the pinned software environment or the host
resource floor is not satisfied.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


GIB = 1024 ** 3

RUNTIME_DISTRIBUTIONS = {
    "torch": {"expected": "2.8.0", "allow_local_suffix": True},
    "torchvision": {"expected": "0.23.0", "allow_local_suffix": True},
    "transformers": {"expected": "4.57.6", "allow_local_suffix": False},
    "tokenizers": {"expected": "0.22.2", "allow_local_suffix": False},
}

BUILD_DISTRIBUTIONS = {
    "hbdk4-compiler": {
        "expected": "4.10.2a2.dev202603180400+4c23b55.develop",
        "allow_local_suffix": False,
    },
    "leap-llm": {"expected": "1.0.5", "allow_local_suffix": False},
}

RESOURCE_PROFILES = {
    "prepare": {
        "minimum_available_memory_bytes": 32 * GIB,
        "minimum_free_disk_bytes": 32 * GIB,
        "minimum_free_cuda_bytes": 16 * GIB,
        "minimum_idle_cpu_cores": 4,
    },
    "calibrate": {
        "minimum_available_memory_bytes": 48 * GIB,
        "minimum_free_disk_bytes": 16 * GIB,
        "minimum_free_cuda_bytes": 18 * GIB,
        "minimum_idle_cpu_cores": 8,
    },
    "build": {
        "minimum_available_memory_bytes": 96 * GIB,
        # Thirteen Language graph variants retain Exported BC, Converted BC,
        # HBO, and the linked HBM. Historical complete builds occupied about
        # 86 GiB even after intermediate cleanup, so 80 GiB is not a safe
        # launch floor for a fresh build.
        "minimum_free_disk_bytes": 160 * GIB,
        "minimum_free_cuda_bytes": 16 * GIB,
        "minimum_idle_cpu_cores": 16,
    },
}


def path_state(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_dir": resolved.is_dir(),
        "is_file": resolved.is_file(),
    }


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_matches(
    actual: str | None,
    expected: str,
    *,
    allow_local_suffix: bool,
) -> bool:
    if actual is None:
        return False
    if allow_local_suffix:
        return actual.split("+", 1)[0] == expected
    return actual == expected


def memory_state() -> dict[str, object]:
    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}
        try:
            with Path("/proc/meminfo").open("r", encoding="ascii") as stream:
                for line in stream:
                    key, raw = line.split(":", 1)
                    fields = raw.strip().split()
                    if fields:
                        values[key] = int(fields[0]) * 1024
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
            if total is not None and available is not None:
                return {
                    "available": True,
                    "source": "/proc/meminfo",
                    "total_bytes": total,
                    "available_bytes": available,
                    "used_bytes": total - available,
                }
        except (OSError, ValueError):
            pass

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {
                    "available": True,
                    "source": "GlobalMemoryStatusEx",
                    "total_bytes": status.total_physical,
                    "available_bytes": status.available_physical,
                    "used_bytes": status.total_physical - status.available_physical,
                }
        except (AttributeError, OSError):
            pass

    return {"available": False, "error": "physical memory metrics unavailable"}


def cpu_state() -> dict[str, object]:
    logical = os.cpu_count()
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    available_cores = affinity or logical
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        load_available = True
    except (AttributeError, OSError):
        load_1m = load_5m = load_15m = None
        load_available = False
    return {
        "logical_cores": logical,
        "affinity_cores": affinity,
        "available_cores": available_cores,
        "load_average_available": load_available,
        "load_1m": load_1m,
        "load_5m": load_5m,
        "load_15m": load_15m,
    }


def nearest_existing_path(path: Path) -> Path | None:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def disk_state(path: Path) -> dict[str, object]:
    probe = nearest_existing_path(path)
    if probe is None:
        return {
            "available": False,
            "requested_path": str(path.expanduser().resolve()),
            "error": "no existing parent for resource path",
        }
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {
            "available": False,
            "requested_path": str(path.expanduser().resolve()),
            "probe_path": str(probe),
            "error": repr(exc),
        }
    return {
        "available": True,
        "requested_path": str(path.expanduser().resolve()),
        "probe_path": str(probe),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def cuda_state(torch_installed: bool) -> dict[str, object]:
    if not torch_installed:
        return {"available": False, "error": "torch is not installed"}
    try:
        import torch

        available = torch.cuda.is_available()
        result: dict[str, object] = {
            "available": available,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if available else 0,
        }
        devices = []
        if available:
            for index in range(torch.cuda.device_count()):
                device: dict[str, object] = {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                }
                try:
                    free, total = torch.cuda.mem_get_info(index)
                    device.update({"memory_free_bytes": free, "memory_total_bytes": total})
                except (RuntimeError, TypeError) as exc:
                    device["memory_error"] = repr(exc)
                devices.append(device)
        result["devices"] = devices
        return result
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def import_probe(modules: list[str], timeout_seconds: int = 60) -> dict[str, object]:
    unique = list(dict.fromkeys(modules))
    source = "\n".join(f"import {name}" for name in unique)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "modules": unique, "error": repr(exc)}
    return {
        "passed": completed.returncode == 0,
        "modules": unique,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def command_probe(command: str, timeout_seconds: int = 60) -> dict[str, object]:
    executable = shutil.which(command)
    if executable is None:
        return {"passed": False, "command": command, "error": "command not found"}
    try:
        completed = subprocess.run(
            [executable, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "command": command,
            "path": executable,
            "error": repr(exc),
        }
    return {
        "passed": completed.returncode == 0,
        "command": command,
        "path": executable,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def resource_failures(
    profile: str,
    resources: dict[str, Any],
    *,
    requested_jobs: int | None,
    requested_cuda_index: int = 0,
) -> list[str]:
    if profile == "audit":
        return []
    requirements = RESOURCE_PROFILES[profile]
    failures: list[str] = []
    memory = resources["memory"]
    disk = resources["disk"]
    cpu = resources["cpu"]
    cuda = resources.get("cuda", {})

    if not memory.get("available"):
        failures.append("available host memory could not be measured")
    elif memory["available_bytes"] < requirements["minimum_available_memory_bytes"]:
        failures.append(
            "available host memory is below the "
            f"{requirements['minimum_available_memory_bytes'] / GIB:.0f} GiB "
            f"{profile} floor"
        )

    if not disk.get("available"):
        failures.append("free disk space could not be measured")
    elif disk["free_bytes"] < requirements["minimum_free_disk_bytes"]:
        failures.append(
            "free disk space is below the "
            f"{requirements['minimum_free_disk_bytes'] / GIB:.0f} GiB "
            f"{profile} floor"
        )

    devices = cuda.get("devices") if isinstance(cuda, dict) else None
    if not cuda.get("available") or not isinstance(devices, list) or not devices:
        failures.append("CUDA device memory could not be measured")
    else:
        requested_device = next(
            (
                device
                for position, device in enumerate(devices)
                if isinstance(device, dict)
                and int(device.get("index", position)) == requested_cuda_index
            ),
            None,
        )
        if requested_device is None:
            failures.append(
                f"requested CUDA device cuda:{requested_cuda_index} is unavailable"
            )
        elif not isinstance(requested_device.get("memory_free_bytes"), int):
            failures.append(
                f"free CUDA memory for cuda:{requested_cuda_index} could not be measured"
            )
        elif requested_device["memory_free_bytes"] < requirements["minimum_free_cuda_bytes"]:
            failures.append(
                f"free CUDA memory on cuda:{requested_cuda_index} is below the "
                f"{requirements['minimum_free_cuda_bytes'] / GIB:.0f} GiB "
                f"{profile} floor"
            )

    available_cores = cpu.get("available_cores")
    if not isinstance(available_cores, int) or available_cores <= 0:
        failures.append("available CPU cores could not be measured")
    else:
        if requested_jobs is not None and requested_jobs <= 0:
            failures.append(f"requested jobs must be positive; got {requested_jobs}")
        elif requested_jobs is not None and requested_jobs > available_cores:
            failures.append(
                f"requested jobs ({requested_jobs}) exceed available CPU cores "
                f"({available_cores})"
            )
        if not cpu.get("load_average_available"):
            failures.append("CPU load average could not be measured")
        else:
            load_1m = float(cpu["load_1m"])
            load_5m = float(cpu["load_5m"])
            required_idle = int(requirements["minimum_idle_cpu_cores"])
            if requested_jobs is not None:
                required_idle = max(required_idle, requested_jobs)
            maximum_launch_load = max(0, available_cores - required_idle)
            if load_1m > maximum_launch_load:
                failures.append(
                    f"1-minute CPU load ({load_1m:.2f}) leaves fewer than "
                    f"{required_idle} idle cores on a {available_cores}-core host"
                )
            if load_5m > available_cores * 1.25:
                failures.append(
                    f"5-minute CPU load ({load_5m:.2f}) exceeds 1.25x available "
                    f"cores ({available_cores})"
                )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("audit", "prepare", "calibrate", "build"),
        default="audit",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--selected-jsonl", type=Path)
    parser.add_argument("--upstream-repo", type=Path)
    parser.add_argument("--resource-path", type=Path, default=Path.cwd())
    parser.add_argument("--requested-jobs", type=int)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used by the pending Prepare, Calibrate, or Build job",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--required-module",
        action="append",
        default=[],
        help="Python module that must import successfully; repeat as needed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formal = args.profile != "audit"
    build = args.profile == "build"
    toolchain = args.profile in {"calibrate", "build"}
    required_modules = list(dict.fromkeys((
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        *args.required_module,
        *(("hbdk4.compiler", "leap_llm.apis.oellm_build") if toolchain else ()),
    )))
    paths = {
        "model": path_state(args.model_path),
        "selected_jsonl": path_state(args.selected_jsonl),
        "upstream_repo": path_state(args.upstream_repo),
    }
    modules = {name: module_available(name) for name in required_modules}
    version_requirements = dict(RUNTIME_DISTRIBUTIONS)
    if toolchain:
        version_requirements.update(BUILD_DISTRIBUTIONS)
    distributions = {
        name: {
            "expected": requirement["expected"],
            "actual": distribution_version(name),
            "allow_local_suffix": requirement["allow_local_suffix"],
        }
        for name, requirement in version_requirements.items()
    }
    for value in distributions.values():
        value["passed"] = version_matches(
            value["actual"],
            value["expected"],
            allow_local_suffix=value["allow_local_suffix"],
        )

    cuda = cuda_state(modules.get("torch", False))
    resources = {
        "cpu": cpu_state(),
        "memory": memory_state(),
        "disk": disk_state(args.resource_path),
        "cuda": cuda,
    }
    probe = import_probe(required_modules) if formal else None
    build_command = command_probe("oellm_build") if build else None

    failures: list[str] = []
    requested_cuda_index = 0
    if args.device == "cuda":
        requested_cuda_index = 0
    elif args.device.startswith("cuda:"):
        try:
            requested_cuda_index = int(args.device.split(":", 1)[1])
        except ValueError:
            failures.append(f"invalid CUDA device: {args.device}")
    elif formal or args.require_cuda:
        failures.append(
            f"formal compiler jobs require a CUDA device; got {args.device!r}"
        )
    missing_paths = [
        name for name, value in paths.items()
        if value is not None and not value["exists"]
    ]
    missing_modules = [name for name, available in modules.items() if not available]
    failures.extend(f"required path is missing: {name}" for name in missing_paths)
    failures.extend(f"required module is missing: {name}" for name in missing_modules)

    python_version = platform.python_version()
    python_passed = sys.version_info[:2] == (3, 10) if formal else True
    if not python_passed:
        failures.append(f"Python {python_version} does not match required Python 3.10")
    if formal:
        for name, value in distributions.items():
            if not value["passed"]:
                failures.append(
                    f"{name}=={value['actual']} does not match required "
                    f"{name}=={value['expected']}"
                )
        if probe is not None and not probe["passed"]:
            failures.append("one or more required Python modules failed an import probe")
    if args.require_cuda and not cuda.get("available"):
        failures.append("CUDA is required but torch.cuda.is_available() is false")
    if build and build_command is not None and not build_command["passed"]:
        failures.append("oellm_build is unavailable or its --help probe failed")
    failures.extend(
        resource_failures(
            args.profile,
            resources,
            requested_jobs=args.requested_jobs,
            requested_cuda_index=requested_cuda_index,
        )
    )

    result: dict[str, object] = {
        "schema_version": 2,
        "profile": args.profile,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "version": python_version,
            "executable": sys.executable,
            "expected_major_minor": "3.10" if formal else None,
            "passed": python_passed,
        },
        "cwd": os.getcwd(),
        "paths": paths,
        "resource_requirements": RESOURCE_PROFILES.get(args.profile),
        "resources": resources,
        "cuda": cuda,
        "distributions": distributions,
        "modules": modules,
        "import_probe": probe,
        "commands": {
            name: shutil.which(name)
            for name in ("nvidia-smi", "hbdk-cc", "hbrt4-run-model-nash", "cmake", "oellm_build")
        },
        "oellm_build_probe": build_command,
        "requested_jobs": args.requested_jobs,
        "requested_device": args.device,
        "requested_cuda_index": requested_cuda_index,
        "passed": not failures,
        "failures": failures,
        "missing_paths": missing_paths,
        "missing_modules": missing_modules,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
