import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "locateanything_hf_assets", ROOT / "scripts" / "hf_assets.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AssetError = MODULE.AssetError
TASK_COUNTS = MODULE.TASK_COUNTS
validate = MODULE.validate
validate_hbm = MODULE.validate_hbm
write_checksums = MODULE.write_checksums


def write_file(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_hbm_assets(root: Path, *, embedding_bytes: int = 16) -> None:
    write_file(root / "LocateAnything-3B_vision.hbm")
    write_file(root / "LocateAnything-3B_language.hbm")
    embeddings = root / "LocateAnything-3B_embed_tokens.bin"
    embeddings.parent.mkdir(parents=True, exist_ok=True)
    with embeddings.open("wb") as stream:
        stream.truncate(embedding_bytes)


def test_validate_hbm_specification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "EMBED_TOKENS_BYTES", 16)
    root = tmp_path / "hbm"
    make_hbm_assets(root)

    result = validate_hbm(root)

    assert result["embedding_specification"]["vocab_size"] == 152681


def test_validate_hbm_rejects_wrong_embedding_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "EMBED_TOKENS_BYTES", 16)
    root = tmp_path / "hbm"
    make_hbm_assets(root, embedding_bytes=1)

    with pytest.raises(AssetError, match="embedding table size mismatch"):
        validate_hbm(root)


def test_checksum_round_trip_with_hbm_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "EMBED_TOKENS_BYTES", 16)
    root = tmp_path / "hbm"
    make_hbm_assets(root)
    write_checksums(root, root / "checksums.sha256")

    result = validate("hbm", tmp_path, verify_images=False)

    assert result["passed"] is True
    assert result["assets"][0]["checksum_files_checked"] == 3


def test_validate_rejects_missing_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "EMBED_TOKENS_BYTES", 16)
    make_hbm_assets(tmp_path / "hbm")

    with pytest.raises(AssetError, match="SHA256 checksum file is missing"):
        validate("hbm", tmp_path, verify_images=False)


def test_task_counts_sum_to_1200() -> None:
    assert TASK_COUNTS == {
        "detection": 660,
        "gui": 150,
        "referring": 120,
        "ocr": 120,
        "layout": 90,
        "pointing": 60,
    }
    assert sum(TASK_COUNTS.values()) == 1200


def test_calibration_manifest_uses_the_public_source_layout() -> None:
    assert MODULE.CALIBRATION_REQUIRED[0] == "current/source/selected.jsonl"
