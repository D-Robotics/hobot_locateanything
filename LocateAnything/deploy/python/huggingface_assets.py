#!/usr/bin/env python3
"""Download and validate LocateAnything calibration data or compiled HBM assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


TASK_COUNTS = {
    "detection": 660,
    "gui": 150,
    "referring": 120,
    "ocr": 120,
    "layout": 90,
    "pointing": 60,
}
CALIBRATION_REQUIRED = (
    "current/source/selected.jsonl",
    "current/generated/generated.jsonl",
)
HBM_REQUIRED = (
    "LocateAnything-3B_vision.hbm",
    "LocateAnything-3B_language.hbm",
    "LocateAnything-3B_embed_tokens.bin",
)
EMBED_TOKENS_BYTES = 152681 * 2048 * 2


class AssetError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise AssetError("missing required files: " + ", ".join(missing))
    empty = [relative for relative in relative_paths if (root / relative).stat().st_size == 0]
    if empty:
        raise AssetError("empty required files: " + ", ".join(empty))


def validate_calibration(root: Path, *, verify_images: bool) -> dict[str, object]:
    require_files(root, CALIBRATION_REQUIRED)
    selected = root / "current" / "source" / "selected.jsonl"
    counts: Counter[str] = Counter()
    image_count = 0
    with selected.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssetError(f"invalid selected.jsonl line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise AssetError(f"selected.jsonl line {line_number} is not an object")
            task = str(record.get("task", ""))
            counts[task] += 1
            image_value = record.get("image")
            if not isinstance(image_value, str) or not image_value:
                raise AssetError(f"selected.jsonl line {line_number} has no image path")
            image = (selected.parent / image_value).resolve()
            try:
                image.relative_to(selected.parent.resolve())
            except ValueError as exc:
                raise AssetError(
                    f"selected.jsonl line {line_number} image escapes calibration root: "
                    f"{image_value}"
                ) from exc
            if not image.is_file():
                raise AssetError(f"selected.jsonl line {line_number} image is missing: {image_value}")
            if verify_images:
                expected = record.get("image_sha256")
                if isinstance(expected, str) and expected and sha256(image) != expected:
                    raise AssetError(
                        f"selected.jsonl line {line_number} image SHA256 mismatch: {image_value}"
                    )
            image_count += 1
    if image_count != 1200:
        raise AssetError(f"selected.jsonl contains {image_count} records; expected 1200")
    if dict(counts) != TASK_COUNTS:
        raise AssetError(f"task counts are {dict(counts)}; expected {TASK_COUNTS}")

    generated = root / "current" / "generated" / "generated.jsonl"
    generated_count = 0
    with generated.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssetError(f"invalid generated.jsonl line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise AssetError(f"generated.jsonl line {line_number} is not an object")
            tensor_value = record.get("tensor_file")
            if not isinstance(tensor_value, str) or not tensor_value:
                raise AssetError(f"generated.jsonl line {line_number} has no tensor_file")
            tensor = (generated.parent / tensor_value).resolve()
            try:
                tensor.relative_to(generated.parent.resolve())
            except ValueError as exc:
                raise AssetError(
                    f"generated.jsonl line {line_number} tensor escapes generated root: "
                    f"{tensor_value}"
                ) from exc
            if not tensor.is_file():
                raise AssetError(
                    f"generated.jsonl line {line_number} tensor is missing: {tensor_value}"
                )
            generated_count += 1
    if generated_count != 1200:
        raise AssetError(f"generated.jsonl contains {generated_count} records; expected 1200")
    return {
        "kind": "calibration",
        "root": str(root.resolve()),
        "sample_count": image_count,
        "generated_tensor_count": generated_count,
        "task_counts": dict(counts),
        "image_sha256_checked": verify_images,
    }


def validate_hbm(root: Path) -> dict[str, object]:
    require_files(root, HBM_REQUIRED)
    embeddings = root / "LocateAnything-3B_embed_tokens.bin"
    if embeddings.stat().st_size != EMBED_TOKENS_BYTES:
        raise AssetError(
            "embedding table size mismatch: "
            f"{embeddings.stat().st_size}; expected {EMBED_TOKENS_BYTES}"
        )
    return {
        "kind": "hbm",
        "root": str(root.resolve()),
        "files": {
            relative: (root / relative).stat().st_size for relative in HBM_REQUIRED
        },
        "embedding_specification": {
            "vocab_size": 152681,
            "embed_dim": 2048,
            "dtype": "float16",
        },
    }


def validate_checksums(root: Path, checksum_file: Path) -> dict[str, object]:
    checked = 0
    with checksum_file.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split(None, 1)
            if len(fields) != 2 or len(fields[0]) != 64:
                raise AssetError(f"invalid SHA256 checksum line {line_number}: {line!r}")
            relative = fields[1].lstrip("* ")
            path = (root / relative).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise AssetError(f"checksum path escapes asset root: {relative}") from exc
            if not path.is_file():
                raise AssetError(f"checksum target is missing: {relative}")
            if sha256(path) != fields[0].lower():
                raise AssetError(f"SHA256 mismatch: {relative}")
            checked += 1
    if checked == 0:
        raise AssetError(f"SHA256 checksum file has no entries: {checksum_file}")
    return {
        "checksums": str(checksum_file.resolve()),
        "checksum_files_checked": checked,
    }


def write_checksums(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    if output != root / "checksums.sha256":
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise AssetError("checksum output must be inside the asset root") from exc
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output and ".cache" not in path.parts
    )
    if not files:
        raise AssetError(f"asset root contains no files: {root}")
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {"checksums": str(output), "checksum_files": len(files)}


def validate(kind: str, local_dir: Path, *, verify_images: bool) -> dict[str, object]:
    results: list[dict[str, object]] = []
    kinds = ("calibration", "hbm") if kind == "all" else (kind,)
    for selected_kind in kinds:
        root = local_dir / selected_kind
        result = (
            validate_calibration(root, verify_images=verify_images)
            if selected_kind == "calibration"
            else validate_hbm(root)
        )
        checksum_file = root / "checksums.sha256"
        if not checksum_file.is_file():
            raise AssetError(f"SHA256 checksum file is missing: {checksum_file}")
        result.update(validate_checksums(root, checksum_file))
        results.append(result)
    return {"passed": True, "assets": results}


def download(args: argparse.Namespace) -> dict[str, object]:
    repo_id = args.repo_id or os.environ.get("LA_HF_REPO")
    if not repo_id:
        raise AssetError("set LA_HF_REPO or pass --repo-id ORGANIZATION/REPOSITORY")
    hf = shutil.which("hf")
    if hf is None:
        raise AssetError("the `hf` command is missing; install huggingface_hub first")
    includes = []
    if args.kind in {"calibration", "all"}:
        includes.append("calibration/**")
    if args.kind in {"hbm", "all"}:
        includes.append("hbm/**")
    command = [
        hf, "download", repo_id,
        "--repo-type", args.repo_type,
        "--revision", args.revision,
        "--include", *includes,
        "--local-dir", str(args.local_dir),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise AssetError(f"hf download failed with exit code {completed.returncode}")
    result = validate(args.kind, args.local_dir, verify_images=args.verify_images)
    result.update({"repo_id": repo_id, "revision": args.revision})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("download", help="download assets and validate them")
    fetch.add_argument("--repo-id", help="Hugging Face repository ID; or set LA_HF_REPO")
    fetch.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    fetch.add_argument("--revision", default="main")
    fetch.add_argument("--kind", choices=("calibration", "hbm", "all"), required=True)
    fetch.add_argument("--local-dir", type=Path, default=Path("artifacts/huggingface"))
    fetch.add_argument("--verify-images", action="store_true")

    check = subparsers.add_parser("validate", help="validate an existing local snapshot")
    check.add_argument("--kind", choices=("calibration", "hbm", "all"), required=True)
    check.add_argument("--local-dir", type=Path, default=Path("artifacts/huggingface"))
    check.add_argument("--verify-images", action="store_true")

    checksums = subparsers.add_parser(
        "checksums", help="write checksums.sha256 for upload"
    )
    checksums.add_argument("--root", type=Path, required=True)
    checksums.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "download":
            result = download(args)
        elif args.command == "validate":
            result = validate(args.kind, args.local_dir, verify_images=args.verify_images)
        else:
            output = args.output or args.root / "checksums.sha256"
            result = {"passed": True, **write_checksums(args.root, output)}
    except AssetError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
