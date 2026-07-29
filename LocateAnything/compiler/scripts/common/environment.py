#!/usr/bin/env python3
"""Check host prerequisites before calibration, compilation, or deployment.

The command prints one JSON object so logs can be archived and compared across
the Windows staging host, compiler host, and S600 runtime host.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--selected-jsonl", type=Path)
    parser.add_argument("--upstream-repo", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--required-module", action="append", default=[],
        help="Python module that must be importable; repeat as needed",
    )
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "paths": {
            "model": path_state(args.model_path),
            "selected_jsonl": path_state(args.selected_jsonl),
            "upstream_repo": path_state(args.upstream_repo),
        },
        "commands": {
            name: shutil.which(name)
            for name in ("nvidia-smi", "hbdk-cc", "hbrt4-run-model-nash", "cmake")
        },
        "modules": {
            name: importlib.util.find_spec(name) is not None
            for name in dict.fromkeys(
                ("torch", "transformers", "PIL", "numpy", *args.required_module)
            )
        },
    }

    cuda_available = False
    if result["modules"]["torch"]:  # type: ignore[index]
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            cuda = {
                "available": cuda_available,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count() if cuda_available else 0,
            }
            if cuda_available:
                free, total = torch.cuda.mem_get_info(0)
                cuda.update({
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_free_bytes": free,
                    "memory_total_bytes": total,
                })
            result["cuda"] = cuda
        except Exception as exc:
            result["cuda"] = {"available": False, "error": repr(exc)}
    else:
        result["cuda"] = {"available": False, "error": "torch not installed"}

    missing_paths = [
        name for name, value in result["paths"].items()  # type: ignore[union-attr]
        if value is not None and not value["exists"]
    ]
    missing_modules = [
        name for name in args.required_module
        if not result["modules"].get(name)  # type: ignore[union-attr]
    ]
    result["passed"] = (
        not missing_paths
        and not missing_modules
        and (cuda_available or not args.require_cuda)
    )
    result["missing_paths"] = missing_paths
    result["missing_modules"] = missing_modules
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
