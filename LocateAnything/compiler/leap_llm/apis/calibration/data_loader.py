from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from leap_llm.apis.calibration.mmstar_process import (
    build_tsv_prompt,
    prepare_tsv_content,
)

__all__ = [
    "load_message_data",
    "load_tsv_data",
]


# Default mmstar cali data path
DEFAULT_MMSTAR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibration_data",
    "mmstar",
    "conversation.json",
)

def _normalize_media_paths(messages: list, json_dir: Path) -> list:
    """Normalize media paths to absolute paths in-memory.

    Args:
        messages (list): The messages data to normalize.
        json_dir (Path): The directory containing the JSON file.

    Returns:
        list: The normalized messages data.
    """
    repo_root = Path(__file__).resolve().parents[3]
    base_candidates: list[Path] = [json_dir, repo_root]

    media_keys = ("audio", "image", "video")
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for element in content:
            for key in media_keys:
                value = element.get(key)
                if not value:
                    continue
                raw = Path(os.path.expanduser(str(value)))
                if raw.is_absolute():
                    element[key] = str(raw)
                    continue

                resolved: Path | None = None
                for base in base_candidates:
                    candidate = (base / raw).resolve()
                    if candidate.exists():
                        resolved = candidate
                        break

                if resolved is None:
                    resolved = (json_dir / raw).resolve()

                element[key] = str(resolved)

    return messages


def load_message_data(
    calib_message_path: str | None = None,
    model_type: str | None = None,
) -> Iterator[list]:
    """
    Load message data for multimodal calibration from a specified path.

    Args:
        calib_message_path (str | None):
            Path to the message file. If None, uses the default
            message data.

    Returns:
        Iterator[list]: An iterator yielding each full message
            (list of messages).
    """
    if not calib_message_path:
        if model_type == "qwen2_5-vl-3b":
            path = Path(DEFAULT_MMSTAR_PATH).resolve()
        else:
            raise ValueError(
                f"calib_message_path is required for model_type={model_type!r}"
            )
    else:
        path = Path(os.path.expanduser(os.path.expandvars(calib_message_path))).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Message data path does not exist: {path}")

    json_files: list[Path]
    if path.is_dir():
        json_files = sorted(p for p in path.glob("*.json") if p.is_file())
        if not json_files:
            raise RuntimeError(f"No json files found in directory {path}")
    else:
        if path.suffix.lower() != ".json":
            raise ValueError("Message data path must point to a json file or directory.")
        json_files = [Path(path)]

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSON file {json_path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"Message data must be a list: {json_path}")

        for item in data:
            if not isinstance(item, dict) or "message" not in item:
                raise ValueError(f"Invalid message format in {json_path}")
            messages = item["message"]
            if not isinstance(messages, list):
                raise ValueError(f"Message must be a list: {json_path}")
            if not calib_message_path:
                yield _normalize_media_paths(messages, json_path.parent)
            else:
                yield messages


def load_tsv_data(
    calib_tsv_path: str,
) -> Iterator[list]:
    """
    Load tsv data for calibration from a specified path.
    """
    assert Path(calib_tsv_path).exists(), f"MMStar data path does not exist: {calib_tsv_path}"
    if Path(calib_tsv_path).is_dir():
        tsv_files = sorted(p for p in Path(calib_tsv_path).glob("*.tsv") if p.is_file())
    else:
        assert Path(calib_tsv_path).suffix.lower() == ".tsv", f"MMStar data path must be a tsv file: {calib_tsv_path}"
        tsv_files = [Path(calib_tsv_path)]
    for tsv_file in tsv_files:
        data_sets = pd.read_csv(tsv_file, sep="\t")
        for idx in range(len(data_sets)):
            raw = build_tsv_prompt(data_sets.iloc[idx])
            prepared = prepare_tsv_content(raw["content"])
            messages = [{"role": "user", "content": prepared}]
            yield messages
