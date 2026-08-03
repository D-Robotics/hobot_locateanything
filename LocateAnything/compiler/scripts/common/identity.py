"""Content identities used to decide whether long-running outputs are reusable."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SOURCE_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml"}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
)

RELEASE_CHECKPOINT_INDEX_SHA256 = (
    "2ecc63fee5f958ffc8142fa29ff7b704a58e80349e9c9ca155a9710d97700271"
)
RELEASE_CHECKPOINT_SHA256 = {
    "model-00001-of-00002.safetensors": (
        "923cfc10fed19808067da6df85a9a4220ddc1f9eb91ceee94c0fecd05d0f2d58"
    ),
    "model-00002-of-00002.safetensors": (
        "3459ba101f40594f3f62d3312014f1f8378b4ba3da3b1d562480045938fc7d47"
    ),
}

# These two Prepare implementations differ only in progress reporting. The
# tensor generation, model inputs, sampling, and serialized record schema are
# byte-for-byte unchanged. Compatibility remains directional and hash-pinned;
# any future Prepare source change is rejected until audited explicitly.
PREPARE_SOURCE_COMPATIBILITY = {
    "57cf747532c6d3453291c9fd76bf853a03e17bed0edfa8629636771c2765123d": {
        "30e036d39ce00b1cd3c510d91385bf5c2df10e88ba42f6c091684090c01fd271"
    },
}


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path: Path, *, normalize_text: bool = False) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"identity input is missing: {path}")
    content = path.read_bytes()
    if normalize_text:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def source_tree_identity(
    root: Path, suffixes: Iterable[str] = SOURCE_SUFFIXES
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"source directory is missing: {root}")
    allowed = {str(value).lower() for value in suffixes}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not {".git", "__pycache__", ".pytest_cache"} & set(path.parts)
    )
    digest = hashlib.sha256()
    normalized_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        normalized_bytes += len(content)
    if not files:
        raise RuntimeError(f"source directory contains no tracked source files: {root}")
    return {
        "file_count": len(files),
        "normalized_bytes": normalized_bytes,
        "sha256": digest.hexdigest(),
    }


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"checkpoint index is missing: {index_path}")
    index = read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"checkpoint index has no weight_map: {index_path}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shards = {}
    for name in shard_names:
        path = model_path / name
        shards[name] = file_identity(path)
    config_path = model_path / "config.json"
    return {
        "index": file_identity(index_path),
        "config": file_identity(config_path),
        "shards": shards,
    }


def release_checkpoint_errors(model_path: Path) -> list[str]:
    """Check the checkpoint files against the frozen release hashes."""

    try:
        identity = checkpoint_identity(model_path)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return [f"cannot identify release checkpoint: {exc}"]
    errors = []
    if identity.get("index", {}).get("sha256") != RELEASE_CHECKPOINT_INDEX_SHA256:
        errors.append("checkpoint index SHA256 does not match the release checkpoint")
    shards = identity.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(RELEASE_CHECKPOINT_SHA256):
        errors.append("checkpoint shard catalog does not match the release checkpoint")
        return errors
    for name, expected_sha in RELEASE_CHECKPOINT_SHA256.items():
        if shards.get(name, {}).get("sha256") != expected_sha:
            errors.append(f"checkpoint shard SHA256 mismatch: {name}")
    return errors


def tokenizer_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    files = {
        name: file_identity(model_path / name)
        for name in TOKENIZER_FILES
        if (model_path / name).is_file()
    }
    if "tokenizer_config.json" not in files:
        raise RuntimeError(f"tokenizer_config.json is missing from {model_path}")
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= files.keys():
        raise RuntimeError(f"model directory has no complete tokenizer files: {model_path}")
    custom_code = sorted(model_path.glob("*.py"))
    return {
        "files": files,
        "custom_code": {
            path.name: file_identity(path, normalize_text=True) for path in custom_code
        },
    }


def rotation_identity(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"mode": "built-in"}
    resolved = Path(path).expanduser().resolve()
    return {"mode": "file", "file": file_identity(resolved)}


def identity_mismatches(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                mismatches.append(path)
            else:
                mismatches.extend(identity_mismatches(expected[key], actual[key], path))
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [prefix]
        mismatches = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatches.extend(identity_mismatches(left, right, f"{prefix}[{index}]"))
        return mismatches
    return [] if expected == actual else [prefix or "<root>"]


def prepare_source_is_compatible(expected: Any, current: Any) -> bool:
    if not identity_mismatches(expected, current):
        return True
    if not isinstance(expected, dict) or not isinstance(current, dict):
        return False
    current_sha = current.get("sha256")
    expected_sha = expected.get("sha256")
    return expected_sha in PREPARE_SOURCE_COMPATIBILITY.get(current_sha, set())


def artifact_identities(paths: Iterable[Path]) -> dict[str, Any]:
    identities = {}
    for path in paths:
        resolved = path.resolve()
        identities[resolved.name] = file_identity(resolved)
    return identities


def prepared_bundle_identity_errors(
    *,
    selected_jsonl: Path,
    generated_jsonl: Path,
    model_path: Path,
    prepare_source_path: Path,
    upstream_repo: Path | None = None,
    expected_sample_count: int | None = None,
) -> list[str]:
    """Validate that prepared tensors still belong to the current inputs.

    Prepare records checkpoint, tokenizer, source, profile, and manifest
    identities before the expensive Float pass. Calibrate and Build call this
    function again so a model or source change cannot silently reuse tensors
    produced by an earlier run.
    """

    selected_jsonl = selected_jsonl.resolve()
    generated_jsonl = generated_jsonl.resolve()
    model_path = model_path.resolve()
    prepare_source_path = prepare_source_path.resolve()
    identity_path = generated_jsonl.parent / "prepare_run_identity.json"
    summary_path = generated_jsonl.parent / "generation_summary.json"

    missing = [
        ("selected manifest", selected_jsonl),
        ("generated manifest", generated_jsonl),
        ("Prepare identity", identity_path),
        ("Prepare summary", summary_path),
        ("Prepare source", prepare_source_path),
    ]
    errors = [f"{label} is missing: {path}" for label, path in missing if not path.is_file()]
    if errors:
        return errors

    try:
        prepare_identity = read_json(identity_path)
        summary = read_json(summary_path)
        if not isinstance(prepare_identity, dict):
            return ["Prepare identity must be a JSON object"]
        if not isinstance(summary, dict):
            return ["Prepare summary must be a JSON object"]

        current_checkpoint = checkpoint_identity(model_path)
        current_tokenizer = tokenizer_identity(model_path)
        current_prepare_source = file_identity(
            prepare_source_path, normalize_text=True
        )
        checks = (
            (
                "Prepare checkpoint does not match the current checkpoint",
                prepare_identity.get("checkpoint"),
                current_checkpoint,
            ),
            (
                "Prepare tokenizer does not match the current tokenizer",
                prepare_identity.get("tokenizer"),
                current_tokenizer,
            ),
        )
        for message, expected, actual in checks:
            mismatches = identity_mismatches(expected, actual)
            if mismatches:
                errors.append(f"{message}: {', '.join(mismatches[:8])}")
        if not prepare_source_is_compatible(
            prepare_identity.get("prepare_source"), current_prepare_source
        ):
            mismatches = identity_mismatches(
                prepare_identity.get("prepare_source"), current_prepare_source
            )
            errors.append(
                "Prepare source does not match the current prepare.py: "
                + ", ".join(mismatches[:8])
            )

        if upstream_repo is not None:
            current_upstream = source_tree_identity(upstream_repo, {".py"})
            mismatches = identity_mismatches(
                prepare_identity.get("upstream_source"), current_upstream
            )
            if mismatches:
                errors.append(
                    "Prepare upstream source does not match the current source: "
                    + ", ".join(mismatches[:8])
                )

        selected_sha = sha256_file(selected_jsonl)
        generated_sha = sha256_file(generated_jsonl)
        prepare_identity_sha = sha256_file(identity_path)
        if prepare_identity.get("selected_manifest_sha256") != selected_sha:
            errors.append("Prepare identity selected manifest SHA256 mismatch")
        if summary.get("selected_manifest_sha256") != selected_sha:
            errors.append("Prepare summary selected manifest SHA256 mismatch")
        if summary.get("generated_manifest_sha256") != generated_sha:
            errors.append("Prepare summary generated manifest SHA256 mismatch")
        if summary.get("prepare_run_identity") != identity_path.name:
            errors.append("Prepare summary identity filename mismatch")
        if summary.get("prepare_run_identity_sha256") != prepare_identity_sha:
            errors.append("Prepare summary identity SHA256 mismatch")
        if summary.get("generated_manifest") != generated_jsonl.name:
            errors.append("Prepare summary generated manifest filename mismatch")

        for field in ("fixed_profile", "generation_config"):
            mismatches = identity_mismatches(
                prepare_identity.get(field), summary.get(field)
            )
            if mismatches:
                errors.append(
                    f"Prepare identity/summary {field} mismatch: "
                    + ", ".join(mismatches[:8])
                )

        if expected_sample_count is not None:
            selected_count = sum(
                bool(line.strip())
                for line in selected_jsonl.read_text(encoding="utf-8").splitlines()
            )
            generated_count = sum(
                bool(line.strip())
                for line in generated_jsonl.read_text(encoding="utf-8").splitlines()
            )
            if selected_count != expected_sample_count:
                errors.append(
                    f"selected manifest has {selected_count} records; "
                    f"expected {expected_sample_count}"
                )
            if generated_count != expected_sample_count:
                errors.append(
                    f"generated manifest has {generated_count} records; "
                    f"expected {expected_sample_count}"
                )
            if summary.get("sample_count") != expected_sample_count:
                errors.append(
                    "Prepare summary sample_count does not match the release count"
                )
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        errors.append(f"cannot validate Prepare identity chain: {exc}")
    return errors
