#!/usr/bin/env python3
"""Materialize one LocateAnything image/prompt pair for repeatable S600 A/B tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from run_locateanything import normalize_prompt, prepare_image, tokenize_prompt


def _readable_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--vision-runner", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    image = _readable_file(args.image, "image")
    tokenizer = args.tokenizer.expanduser().resolve()
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer.json").is_file():
        raise ValueError(f"tokenizer directory is incomplete: {tokenizer}")
    vision_runner = _readable_file(args.vision_runner, "Vision runner", executable=True)
    vision_model = _readable_file(args.vision_model, "Vision HBM")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    normalized_prompt, task = normalize_prompt(args.prompt)
    vision_input, transform = prepare_image(image)
    prompt_tokens = tokenize_prompt(tokenizer, args.prompt)
    vision_input_path = output_dir / "vision_input.f16.bin"
    prompt_tokens_path = output_dir / "prompt_tokens.i32.bin"
    visual_features_path = output_dir / "visual_features.f16.bin"
    vision_log_path = output_dir / "vision.log"
    vision_input.tofile(vision_input_path)
    prompt_tokens.tofile(prompt_tokens_path)

    environment = os.environ.copy()
    environment.setdefault("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
    completed = subprocess.run(
        [
            str(vision_runner),
            "--model",
            str(vision_model),
            "--input",
            str(vision_input_path),
            "--output",
            str(visual_features_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    vision_log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Vision runner exited with code {completed.returncode}; log: {vision_log_path}"
        )
    expected_visual_bytes = 1 * 576 * 2048 * 2
    if not visual_features_path.is_file() or visual_features_path.stat().st_size != expected_visual_bytes:
        raise RuntimeError(
            f"visual feature size mismatch: expected {expected_visual_bytes} bytes"
        )

    files = (vision_input_path, prompt_tokens_path, visual_features_path)
    manifest = {
        "schema_version": 1,
        "image": str(image),
        "image_sha256": _sha256(image),
        "prompt": args.prompt,
        "normalized_prompt": normalized_prompt,
        "task": task,
        "image_transform": transform,
        "prompt_tokens": int(prompt_tokens.size),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] task={task} prompt_tokens={prompt_tokens.size}")
    print(f"[OUTPUT] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
