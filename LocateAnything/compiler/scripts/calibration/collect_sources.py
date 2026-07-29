#!/usr/bin/env python3
"""Stream small training subsets for the six LocateAnything task domains.

Supports two modes:
  - Streaming (default): loads from Hugging Face datasets via the network.
  - Local (--local-source): reads from local directories only; network access is
    forbidden and will cause an immediate failure.

The ``--local-parquet`` flag is deprecated in favour of ``--local-source``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import traceback
import urllib.request
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Per-domain source specifications (streaming / HF defaults)
# ---------------------------------------------------------------------------

SOURCE_SPECS = {
    "detection": {
        "dataset": "benjamintli/sku110k",
        "config": "default",
        "split": "train",
        "revision": "41ffe587d4e568204bbf752ba8f18ea21e3d7b09",
        "source": "SKU110K",
        "license": "see SKU110K upstream terms",
        "shuffle_buffer": 32,
    },
    "gui": {
        "dataset": "likaixin/GroundCUA-train",
        "config": "train",
        "split": "train",
        "revision": "ec731b2225ec08db63aabb18a95137f4046a67b4",
        "source": "GroundCUA",
        "license": "MIT",
    },
    "referring": {
        "dataset": "sionic-ai/refcocog_object_detection",
        "config": "default",
        "split": "train",
        "revision": "efbdf2cc32689178bec9374f20798537421509ea",
        "source": "RefCOCOg",
        "license": "see RefCOCOg and COCO upstream terms",
    },
    "ocr": {
        "dataset": "Berzerker/ocr_hiertext",
        "config": "default",
        "split": "train",
        "revision": "7aef3f135b0fb5ae063174c0b4b8d760ebaa4992",
        "source": "HierText",
        "license": "see HierText upstream terms",
    },
    "layout": {
        "dataset": "docling-project/DocLayNet-v1.2",
        "config": "default",
        "split": "train",
        "revision": "0daf93102e2efce76c3e11a274a5e0d0969391d3",
        "source": "DocLayNet",
        "license": "CDLA-Permissive-1.0",
    },
    "pointing": {
        "dataset": "allenai/pixmo-points",
        "config": "default",
        "split": "train",
        "revision": "2b5c6931e790e00ae00d4a2857e5f95d88f09a66",
        "source": "PixMo-Points",
        "license": "ODC-BY-1.0",
    },
}

DEFAULT_COUNTS = {
    "detection": 130,
    "gui": 70,
    "referring": 50,
    "ocr": 40,
    "layout": 40,
    "pointing": 32,
}

# ---------------------------------------------------------------------------
# Per-domain local source provenance (local mode only)
# ---------------------------------------------------------------------------
# These are the *actual* source identities for local data.  They intentionally
# differ from ``SOURCE_SPECS`` above in several cases:
#
#   detection — local CSV+JPEG is the raw SKU110K_fixed release, not the HF
#     ``benjamintli/sku110k`` parquet that the streaming adapter expects.
#   referring — the local dataset is ``lmms-lab/RefCOCOg`` (val/test only),
#     not ``sionic-ai/refcocog_object_detection``.
#   pointing — images are remote URLs; provenance is the metadata-only
#     arrow shard, not a complete offline dataset.
#
# The output manifest's ``source_dataset`` and ``source_revision`` MUST
# reflect the actual local input, never the streaming reference.

LOCAL_SOURCE_SPECS: dict[str, dict[str, str]] = {
    "detection": {
        "dataset": "SKU110K_fixed",
        "revision": "local-csv-jpeg",
        "source": "SKU110K",
        "license": "see SKU110K upstream terms",
        "note": (
            "Raw SKU110K_fixed release (CSV annotations + JPEG files), "
            "not the HF benjamintli/sku110k parquet."
        ),
    },
    "gui": {
        "dataset": "likaixin/GroundCUA-train",
        "revision": "ec731b2225ec08db63aabb18a95137f4046a67b4",
        "source": "GroundCUA",
        "license": "MIT",
    },
    "referring": {
        "dataset": "lmms-lab/RefCOCOg",
        "revision": "local-arrow",
        "source": "RefCOCOg",
        "license": "see RefCOCOg and COCO upstream terms",
        "note": (
            "lmms-lab/RefCOCOg (val/test only, no train). "
            "For formal calibration, sionic-ai/refcocog_object_detection "
            "train is required."
        ),
    },
    "ocr": {
        "dataset": "Berzerker/ocr_hiertext",
        "revision": "7aef3f135b0fb5ae063174c0b4b8d760ebaa4992",
        "source": "HierText",
        "license": "see HierText upstream terms",
    },
    "layout": {
        "dataset": "docling-project/DocLayNet-v1.2",
        "revision": "0daf93102e2efce76c3e11a274a5e0d0969391d3",
        "source": "DocLayNet",
        "license": "CDLA-Permissive-1.0",
    },
    "pointing": {
        "dataset": "allenai/pixmo-points",
        "revision": "2b5c6931e790e00ae00d4a2857e5f95d88f09a66",
        "source": "PixMo-Points",
        "license": "ODC-BY-1.0",
        "note": "Metadata-only arrow shard; images are remote URLs.",
    },
}

# ---------------------------------------------------------------------------
# Local source identity registry.
#
# ``LOCAL_SOURCE_SPECS`` above pins the *default* local identity per task —
# i.e. the dataset currently sitting on disk under ``dataset/<name>``. Some
# tasks can legitimately have more than one local identity:
#
#   referring — the on-disk default is ``lmms-lab/RefCOCOg`` (val/test only,
#     audit/held-out), but formal calibration MUST use
#     ``sionic-ai/refcocog_object_detection`` train once it is populated under
#     ``dataset/RefCOCOg-train-sionic``.
#
# ``LOCAL_SOURCE_IDENTITIES`` binds each identity key to its full provenance
# (dataset / revision / split / source label / license) plus the schema the
# adapter expects and an ``enabled`` flag. ``enabled=False`` means
# "audit / held-out only" — ``collect_domain_local`` fail-closes rather than
# emit formal-calibration records. This is how lmms-lab/RefCOCOg val/test
# stays available for audit while being forbidden from calibration
# according to the train-only source contract.
#
# ``LOCAL_SOURCE_ALTERNATES`` maps a detected schema label to the identity
# key the collector must use when that schema is observed at load time. This
# prevents sionic-schema rows from being written out as the unrelated
# ``lmms-lab/RefCOCOg`` source. The detected schema
# selects the matching identity and the manifest's
# ``source_dataset`` / ``source_revision`` reflect the actual input.
# ---------------------------------------------------------------------------

LOCAL_SOURCE_IDENTITIES: dict[str, dict[str, Any]] = {
    "detection_sku110k_local": {
        "dataset": "SKU110K_fixed",
        "revision": "local-csv-jpeg",
        "split": "train",
        "source": "SKU110K",
        "license": "see SKU110K upstream terms",
        "schema": "sku110k_csv_jpeg",
        "adapter": "_local_detection_adapter",
        "enabled": True,
        "description": "Local SKU110K_fixed release (CSV + JPEG).",
    },
    "gui_groundcua_local": {
        "dataset": "likaixin/GroundCUA-train",
        "revision": "ec731b2225ec08db63aabb18a95137f4046a67b4",
        "split": "train",
        "source": "GroundCUA",
        "license": "MIT",
        "schema": "groundcua_arrow",
        "adapter": "_local_gui_adapter",
        "enabled": True,
        "description": "Local GroundCUA training split adapter.",
    },
    "referring_lmms_refcocog_audit": {
        # Identity for the lmms-lab/RefCOCOg snapshot currently on disk.
        # Verified by reading dataset/RefCOCOg/val/dataset_info.json:
        # dataset_name="ref_coc_og", splits=["val","test"], no train. This
        # is NOT sionic-ai/refcocog_object_detection. Keeping the identity
        # explicit prevents lmms data from ever being written out labelled
        # as sionic.
        "dataset": "lmms-lab/RefCOCOg",
        "revision": "local-arrow",
        "split": "val+test",
        "source": "RefCOCOg-lmms-audit",
        "license": "see RefCOCOg and COCO upstream terms",
        "schema": "lmms_refcocog_valtest",
        "adapter": "_local_referring_adapter",
        "enabled": False,
        "description": (
            "lmms-lab/RefCOCOg snapshot currently on disk (val/test only). "
            "Audit / held-out only — formal calibration requires train "
            "split per README §3. Fail closed."
        ),
    },
    "referring_sionic_train": {
        # Identity formal calibration MUST use for referring. The actual
        # The sionic training Arrow data may not be present locally. Once it is
        # populated under dataset/RefCOCOg-train-sionic, the
        # schema mirrors what the streaming referring_adapter already
        # expects from sionic-ai/refcocog_object_detection:
        #   question starts with '[detect]', answer contains a literal
        #   '<bbox>[x1,y1,x2,y2]</bbox>' token in 0-1000 normalized coords.
        "dataset": "sionic-ai/refcocog_object_detection",
        "revision": "efbdf2cc32689178bec9374f20798537421509ea",
        "split": "train",
        "source": "RefCOCOg",
        "license": "see RefCOCOg and COCO upstream terms",
        "schema": "sionic_refcocog_train",
        "adapter": "_local_referring_sionic_adapter",
        "enabled": True,
        "description": (
            "sionic-ai/refcocog_object_detection train (pinned revision). "
            "Local adapter matches the streaming referring_adapter schema. "
            "Status: code complete, not yet probed against real data "
            "(sionic train arrow not downloaded)."
        ),
    },
    "ocr_hiertext_local": {
        "dataset": "Berzerker/ocr_hiertext",
        "revision": "7aef3f135b0fb5ae063174c0b4b8d760ebaa4992",
        "split": "train",
        "source": "HierText",
        "license": "see HierText upstream terms",
        "schema": "hiertext_arrow",
        "adapter": "_local_ocr_adapter",
        "enabled": True,
        "description": "Local HierText arrow snapshot (single train split).",
    },
    "layout_doclaynet_local": {
        "dataset": "docling-project/DocLayNet-v1.2",
        "revision": "0daf93102e2efce76c3e11a274a5e0d0969391d3",
        "split": "train",
        "source": "DocLayNet",
        "license": "CDLA-Permissive-1.0",
        "schema": "doclaynet_arrow",
        "adapter": "_local_layout_adapter",
        "enabled": True,
        "description": "Local DocLayNet-v1.2 arrow snapshot.",
    },
    "pointing_pixmo_local": {
        "dataset": "allenai/pixmo-points",
        "revision": "2b5c6931e790e00ae00d4a2857e5f95d88f09a66",
        "split": "train",
        "source": "PixMo-Points",
        "license": "ODC-BY-1.0",
        "schema": "pixmo_points_url_only",
        "adapter": "_local_pointing_adapter",
        "enabled": False,
        "description": (
            "Local PixMo-Points metadata. Images are URL-only; fail closed."
        ),
    },
    "pointing_pixmo_cache": {
        "dataset": "allenai/pixmo-points",
        "revision": "2b5c6931e790e00ae00d4a2857e5f95d88f09a66",
        "split": "train",
        "source": "PixMo-Points",
        "license": "ODC-BY-1.0",
        "schema": "pixmo_points_local_cache",
        "adapter": "_local_pointing_adapter",
        "enabled": True,
        "description": (
            "Deterministic PixMo-Points candidate bundle with verified local "
            "image paths and original metadata SHA256 values."
        ),
    },
}

# Default identity key per task (the dataset currently sitting on disk).
LOCAL_SOURCE_DEFAULT_KEYS: dict[str, str] = {
    "detection": "detection_sku110k_local",
    "gui": "gui_groundcua_local",
    "referring": "referring_lmms_refcocog_audit",
    "ocr": "ocr_hiertext_local",
    "layout": "layout_doclaynet_local",
    "pointing": "pointing_pixmo_local",
}

# Schema-label → identity-key bindings used by referring auto-detection.
# When the loader observes ``answer`` as a str-with-<bbox>, that signals the
# sionic schema and the collector MUST use the sionic identity (and write
# sionic provenance), not the lmms-lab default. This is the fix for the
# This prevents source-provenance mis-attribution.
LOCAL_SOURCE_ALTERNATES: dict[str, dict[str, str]] = {
    "referring": {
        # schema_label → identity_key
        "sionic-ai/refcocog_object_detection": "referring_sionic_train",
        "lmms-lab/RefCOCOg": "referring_lmms_refcocog_audit",
    },
    "pointing": {
        "pixmo_points_local_cache": "pointing_pixmo_cache",
    },
}

# Inverse map: adapter function name → identity key (used by tests and by
# the explicit --local-source-task-key selector).
_IDENTITY_KEY_BY_ADAPTER = {
    identity["adapter"]: key for key, identity in LOCAL_SOURCE_IDENTITIES.items()
}


def select_local_source_identity(
    task: str,
    *,
    schema_label: str | None = None,
    explicit_key: str | None = None,
) -> dict[str, Any]:
    """Resolve the local source identity for *task*.

    Selection precedence:
      1. ``explicit_key`` (from ``--local-source-task-key``) if provided;
         must be a valid identity key registered for *task*.
      2. ``schema_label`` (from referring auto-detection) if provided;
         must be present in ``LOCAL_SOURCE_ALTERNATES[task]``.
      3. ``LOCAL_SOURCE_DEFAULT_KEYS[task]``.

    Returns the full identity dict from ``LOCAL_SOURCE_IDENTITIES``. Raises
    ``ValueError`` for unknown keys / unknown schema labels / unknown tasks.
    Does NOT consult ``enabled`` — callers must check that separately via
    ``ensure_identity_enabled`` so they can produce task-specific error
    messages.
    """
    if task not in LOCAL_SOURCE_DEFAULT_KEYS:
        raise ValueError(f"unknown local source task: {task!r}")

    if explicit_key is not None:
        if explicit_key not in LOCAL_SOURCE_IDENTITIES:
            raise ValueError(
                f"unknown local source identity key: {explicit_key!r}"
            )
        # Validate the key is registered as a valid identity for this task.
        valid_for_task = {LOCAL_SOURCE_DEFAULT_KEYS[task]}
        valid_for_task.update(LOCAL_SOURCE_ALTERNATES.get(task, {}).values())
        if explicit_key not in valid_for_task:
            raise ValueError(
                f"identity key {explicit_key!r} is not registered for task "
                f"{task!r}. Valid keys: {sorted(valid_for_task)}"
            )
        return dict(LOCAL_SOURCE_IDENTITIES[explicit_key])

    if schema_label is not None:
        alternates = LOCAL_SOURCE_ALTERNATES.get(task, {})
        if schema_label not in alternates:
            raise ValueError(
                f"unknown schema label {schema_label!r} for task {task!r}"
            )
        return dict(LOCAL_SOURCE_IDENTITIES[alternates[schema_label]])

    return dict(LOCAL_SOURCE_IDENTITIES[LOCAL_SOURCE_DEFAULT_KEYS[task]])


def ensure_identity_enabled(task: str, identity: dict[str, Any]) -> None:
    """Fail closed if *identity* is disabled (audit / held-out only).

    Disabled identities are kept in the registry so audit code can still
    load and inspect the data, but ``collect_domain_local`` must refuse to
    emit formal-calibration records from them.
    """
    if not identity.get("enabled", False):
        raise RuntimeError(
            f"local source identity for task {task!r} is disabled "
            f"(audit/held-out only): dataset={identity.get('dataset')!r}, "
            f"split={identity.get('split')!r}. "
            f"{identity.get('description', '')}"
        )


def resolve_adapter_for_identity(identity: dict[str, Any]) -> Callable[..., tuple[Any, dict[str, Any]]]:
    """Return the local adapter function named by *identity*['adapter']."""
    adapter_name = identity.get("adapter")
    if not adapter_name:
        raise ValueError(
            f"identity {identity.get('dataset')!r} has no adapter binding"
        )
    # LOCAL_ADAPTERS is defined below; look it up lazily so this helper can
    # be called after the registry is built.
    adapter = LOCAL_ADAPTERS_BY_NAME.get(adapter_name)
    if adapter is None:
        raise ValueError(
            f"identity {identity.get('dataset')!r} references unknown "
            f"adapter {adapter_name!r}"
        )
    return adapter

DOCLAYNET_CATEGORIES = [
    "caption",
    "footnote",
    "formula",
    "list item",
    "page footer",
    "page header",
    "picture",
    "section header",
    "table",
    "text",
    "title",
]


def doclaynet_label(category_id: Any) -> str:
    """Map a DocLayNet ``category_id`` to its human-readable label.

    DocLayNet ships COCO-style 1-indexed ``category_id`` values (1..11),
    while ``DOCLAYNET_CATEGORIES`` is a 0-indexed Python list (0..10).
    Without this offset, raw id 6 ("page header" per the DocLayNet spec) would
    be mislabelled as "picture" (index 6 of the 0-indexed list). G1 confirmed
    the local train shard uses 1-indexed ids (row0 category_id[0]=6 -> "page
    header"), so we subtract 1 before indexing. Unknown ids fall back to a
    generic label.
    """

    try:
        cat = int(category_id)
    except (TypeError, ValueError):
        return f"layout {category_id}"
    # 1-indexed COCO id → 0-indexed list position.
    index = cat - 1
    if 0 <= index < len(DOCLAYNET_CATEGORIES):
        return DOCLAYNET_CATEGORIES[index]
    return f"layout {cat}"

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def normalized_coordinate(value: float, extent: float) -> int:
    if extent <= 0:
        raise ValueError(f"invalid coordinate extent: {extent}")
    return max(0, min(1000, int(round(float(value) / float(extent) * 1000))))


def normalized_percent(value: float) -> int:
    return max(0, min(1000, int(round(float(value) * 10))))


def xywh_box(box: Iterable[float], width: float, height: float) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = [float(value) for value in box]
    return (
        normalized_coordinate(x, width),
        normalized_coordinate(y, height),
        normalized_coordinate(x + box_width, width),
        normalized_coordinate(y + box_height, height),
    )


def xyxy_box(box: Iterable[float], width: float, height: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        normalized_coordinate(x1, width),
        normalized_coordinate(y1, height),
        normalized_coordinate(x2, width),
        normalized_coordinate(y2, height),
    )


def box_token(box: tuple[int, int, int, int]) -> str:
    return "<box>" + "".join(f"<{coordinate}>" for coordinate in box) + "</box>"


def point_token(point: tuple[int, int]) -> str:
    return f"<box><{point[0]}><{point[1]}></box>"


def grouped_response(items: list[tuple[str, str]]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for label, geometry in items:
        grouped[label].append(geometry)
    return "".join(
        f"<ref>{label}</ref>{''.join(geometries)}"
        for label, geometries in grouped.items()
    )


def sequence_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        keys = list(value)
        if not keys:
            return []
        length = len(value[keys[0]])
        return [{key: value[key][index] for key in keys} for index in range(length)]
    return []


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def image_from_value(
    value: Any,
    dataset_id: str,
    *,
    allow_network: bool = True,
) -> Any:
    """Materialize a PIL image from a row value.

    ``allow_network`` is the single hard gate for network access in this
    collector. Local loaders always call with ``allow_network=False``; the
    network-streaming path uses the default ``allow_network=True``. In local
    mode a URL-valued image is rejected explicitly instead of being fetched,
    which is the invariant tested by the "no network in local mode" test
    suite.
    """

    from PIL import Image

    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).copy()
        if value.get("path"):
            return Image.open(value["path"]).copy()
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            if not allow_network:
                raise ValueError(
                    "remote image URL is not allowed in local offline mode: "
                    f"{value[:80]}"
                )
            with urllib.request.urlopen(value, timeout=60) as response:
                return Image.open(io.BytesIO(response.read())).copy()
        if not allow_network:
            # Non-URL string with allow_network=False: treat strictly as a
            # local filesystem path; HF hub download is forbidden.
            if not os.path.exists(value):
                raise FileNotFoundError(
                    f"local image path does not exist (offline mode): {value}"
                )
            return Image.open(value).copy()
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=dataset_id,
            filename=value,
            repo_type="dataset",
        )
        return Image.open(local_path).copy()
    raise TypeError(f"cannot materialize image from {type(value).__name__}")


def store_image(image: Any, image_dir: Path) -> tuple[str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0, optimize=False)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    destination = image_dir / f"{digest}.jpg"
    if not destination.exists():
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    return destination.name, digest


# ---------------------------------------------------------------------------
# Box-size stratification helpers for detection
# ---------------------------------------------------------------------------

_BOX_AREA_QUANTILES = [0.0, 0.01, 0.05, 0.25, 1.0]


def _box_area(box: tuple[int, int, int, int]) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _stratify_boxes(
    boxes: list[tuple[tuple[int, int, int, int], str]],
    max_boxes: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Select up to *max_boxes* stratified by small/medium/large area."""
    if len(boxes) <= max_boxes:
        return boxes

    areas = [_box_area(box) for box, _ in boxes]
    sorted_areas = sorted(areas)
    n = len(sorted_areas)
    thresholds = [
        sorted_areas[max(0, min(n - 1, int(q * (n - 1))))]
        for q in _BOX_AREA_QUANTILES
    ]
    # small: [0, q20), medium: [q20, q80), large: [q80, inf)
    # If thresholds collapse (all boxes same area), shuffle and take first N
    if thresholds[1] == thresholds[3]:
        rng.shuffle(boxes)
        return boxes[:max_boxes]

    small = [(box, label) for (box, label), a in zip(boxes, areas) if a < thresholds[1]]
    medium = [
        (box, label)
        for (box, label), a in zip(boxes, areas)
        if thresholds[1] <= a < thresholds[3]
    ]
    large = [(box, label) for (box, label), a in zip(boxes, areas) if a >= thresholds[3]]

    # Sort each group by area descending within group
    small.sort(key=lambda x: _box_area(x[0]), reverse=True)
    medium.sort(key=lambda x: _box_area(x[0]), reverse=True)
    large.sort(key=lambda x: _box_area(x[0]), reverse=True)

    # Weighted allocation: 40% medium, 35% small, 25% large (preferring medium/mid-size)
    alloc_small = max(1, int(max_boxes * 0.35))
    alloc_medium = max(1, int(max_boxes * 0.40))
    alloc_large = max(1, int(max_boxes * 0.25))

    # Adjust to exactly max_boxes
    total = alloc_small + alloc_medium + alloc_large
    if total < max_boxes:
        alloc_medium += max_boxes - total
    elif total > max_boxes:
        alloc_small = max(1, alloc_small - (total - max_boxes))

    selected = (
        small[:alloc_small] + medium[:alloc_medium] + large[:alloc_large]
    )
    rng.shuffle(selected)
    return selected[:max_boxes]


# ---------------------------------------------------------------------------
# Streaming (HF) adapters — unchanged from original
# ---------------------------------------------------------------------------


def detection_adapter(
    row: dict[str, Any], index: int, dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    del dataset
    image = image_from_value(row["image"], SOURCE_SPECS["detection"]["dataset"])
    width, height = image.size
    objects = sequence_rows(row.get("objects"))
    objects = [item for item in objects if len(item.get("bbox") or []) == 4]
    if not objects:
        raise ValueError("SKU110K row has no usable objects")
    objects.sort(
        key=lambda item: float(item["bbox"][2]) * float(item["bbox"][3]),
        reverse=True,
    )
    items = [
        ("object", box_token(xywh_box(item["bbox"], width, height)))
        for item in objects[:48]
    ]
    return image, {
        "sample_id": f"sku110k-{index}",
        "categories": ["object"],
        "target_response": grouped_response(items),
        "metadata": {"target_count": len(items), "image_size": [int(width), int(height)]},
        "source_width": int(width),
        "source_height": int(height),
    }


def gui_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    instructions = row.get("instructions") or []
    bboxes = row.get("bboxes") or []
    if not instructions or len(instructions) != len(bboxes):
        raise ValueError("GroundCUA row has no aligned instructions and boxes")
    choice = rng.randrange(len(instructions))
    phrase = str(instructions[choice]).strip()
    image = image_from_value(row["image"], SOURCE_SPECS["gui"]["dataset"])
    width, height = image.size
    box = xyxy_box(bboxes[choice], width, height)
    if index % 2 == 0:
        point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        geometry = point_token(point)
        output_type = "point"
    else:
        geometry = box_token(box)
        output_type = "box"
    return image, {
        "sample_id": f"groundcua-{row.get('image_id', index)}-{choice}",
        "phrase": phrase,
        "output_type": output_type,
        "target_response": f"<ref>{phrase}</ref>{geometry}",
        "metadata": {
            "target_count": 1,
            "instruction_type": (row.get("inst_type") or [None] * len(instructions))[choice],
            "software": row.get("software"),
        },
        "source_width": int(width),
        "source_height": int(height),
    }


def referring_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    del rng
    phrase = re.sub(r"^\[detect\]\s*", "", str(row.get("question") or "")).strip()
    if not phrase:
        raise ValueError("RefCOCOg row has no referring expression")
    match = re.search(
        r"<bbox>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</bbox>",
        str(row.get("answer") or ""),
    )
    if not match:
        raise ValueError("RefCOCOg row has no normalized bbox answer")
    box = tuple(int(match.group(group)) for group in range(1, 5))
    image = image_from_value(row["image"], SOURCE_SPECS["referring"]["dataset"])
    return image, {
        "sample_id": f"refcocog-{Path(str(row.get('image_path', index))).stem}-{index}",
        "phrase": phrase,
        "target_response": f"<ref>{phrase}</ref>{box_token(box)}",
        "metadata": {"target_count": 1, "image_size": list(image.size)},
        "source_width": int(image.size[0]),
        "source_height": int(image.size[1]),
    }


def parse_hiertext_with_stats(
    value: Any,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    text = str(value or "")
    if text.startswith('"'):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            pass
    items = []
    stats = {
        "parsed_word_boxes": 0,
        "dropped_non_positive_extent": 0,
        "dropped_degenerate_after_normalization": 0,
    }
    for line in text.splitlines():
        match = re.match(
            r"^-\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
            r"(?:True|False)\s+(?:True|False)\s+(.+)$",
            line.strip(),
        )
        if not match:
            continue
        x, y, width, height = [float(match.group(i)) for i in range(1, 5)]
        label = match.group(5).strip()
        if not label:
            continue
        stats["parsed_word_boxes"] += 1
        if width <= 0 or height <= 0:
            stats["dropped_non_positive_extent"] += 1
            continue
        normalized = (
            normalized_percent(x),
            normalized_percent(y),
            normalized_percent(x + width),
            normalized_percent(y + height),
        )
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            stats["dropped_degenerate_after_normalization"] += 1
            continue
        if len(items) < 48:
            items.append((label, box_token(normalized)))
    return items, stats


def parse_hiertext(value: Any) -> list[tuple[str, str]]:
    items, _ = parse_hiertext_with_stats(value)
    return items


def ensure_lossless_ocr_record(record: dict[str, Any]) -> None:
    filter_stats = (record.get("metadata") or {}).get("hiertext_filter") or {}
    if (
        filter_stats.get("dropped_non_positive_extent", 0)
        or filter_stats.get("dropped_degenerate_after_normalization", 0)
        or filter_stats.get("parsed_word_boxes", 0) > 48
    ):
        raise ValueError(
            "OCR row is not lossless under the 48-target calibration contract: "
            f"{filter_stats}"
        )


def ensure_lossless_layout_record(record: dict[str, Any]) -> None:
    filter_stats = (record.get("metadata") or {}).get("layout_filter") or {}
    if (
        filter_stats.get("invalid_source_boxes", 0)
        or filter_stats.get("degenerate_after_normalization", 0)
        or filter_stats.get("unique_valid_boxes", 0) > 48
    ):
        raise ValueError(
            "Layout row is not lossless under the 48-target calibration contract: "
            f"{filter_stats}"
        )


def ocr_adapter(
    row: dict[str, Any], index: int, _dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    items, parse_stats = parse_hiertext_with_stats(row.get("output_json_dumpsed"))
    if not items:
        raise ValueError("HierText row has no parsed word boxes")
    image = image_from_value(row["image"], SOURCE_SPECS["ocr"]["dataset"])
    width, height = image.size
    return image, {
        "sample_id": f"hiertext-{index}",
        "target_response": grouped_response(items),
        "metadata": {"target_count": len(items), "hiertext_filter": parse_stats},
        "source_width": int(width),
        "source_height": int(height),
    }


def layout_adapter(
    row: dict[str, Any], index: int, _dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    image = image_from_value(row["image"], SOURCE_SPECS["layout"]["dataset"])
    width, height = image.size
    grouped_boxes: dict[tuple[int, int, int, int, int], None] = {}
    for category, box in zip(row["category_id"], row["bboxes"]):
        normalized = xywh_box(box, width, height)
        grouped_boxes[(int(category), *normalized)] = None
    items = []
    categories = []
    for category, x1, y1, x2, y2 in list(grouped_boxes)[:48]:
        label = doclaynet_label(category)
        categories.append(label)
        items.append((label, box_token((x1, y1, x2, y2))))
    if not items:
        raise ValueError("DocLayNet row has no layout boxes")
    source_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return image, {
        "sample_id": f"doclaynet-{source_metadata.get('page_hash', index)}",
        "categories": list(dict.fromkeys(categories)),
        "target_response": grouped_response(items),
        "metadata": {
            "target_count": len(items),
            "document_category": source_metadata.get("collection"),
        },
        "source_width": int(width),
        "source_height": int(height),
    }


def pointing_adapter(
    row: dict[str, Any], index: int, _dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    label = str(row.get("label") or "").strip()
    points = row.get("points") or []
    if not label or not points:
        raise ValueError("PixMo row has no label or points")
    geometries = []
    for point in points[:48]:
        geometries.append(
            point_token((normalized_percent(point["x"]), normalized_percent(point["y"])))
        )
    image = image_from_value(row["image_url"], SOURCE_SPECS["pointing"]["dataset"])
    width, height = image.size
    return image, {
        "sample_id": f"pixmo-{row.get('image_sha256', index)}-{label}",
        "phrase": label,
        "target_response": f"<ref>{label}</ref>{''.join(geometries)}",
        "metadata": {"target_count": len(geometries), "collection_method": row.get("collection_method")},
        "source_width": int(width),
        "source_height": int(height),
    }


# ---------------------------------------------------------------------------
# Local (offline) adapters
# ---------------------------------------------------------------------------

# --- Detection: CSV + JPEG ---


def _local_detection_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    """Local SKU110K CSV+JPEG adapter.

    *row* is produced by the local detection loader and contains:
      image_path, image_name, boxes (list of (x1,y1,x2,y2)),
      image_width, image_height.

    ``allow_network=False`` is passed to ``image_from_value`` so this adapter
    can never trigger a network fetch even if the row's image path were a URL.
    """
    image_path = Path(row["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"detection image not found: {image_path}")
    # Pass through image_from_value with allow_network=False for the hard
    # gate; it falls through to the local-path branch.
    image = image_from_value(str(image_path), SOURCE_SPECS["detection"]["dataset"], allow_network=False)
    width = int(row["image_width"])
    height = int(row["image_height"])
    if image.size != (width, height):
        raise ValueError(
            f"detection image size mismatch: {image_path} reports "
            f"{image.size}, CSV has {width}x{height}"
        )

    boxes = row["boxes"]
    if not boxes:
        raise ValueError(f"SKU110K row {row['image_name']} has no usable boxes")

    # Stratify: pick up to 48 boxes by small/medium/large
    labelled = [(box, "object") for box in boxes]
    selected = _stratify_boxes(labelled, 48, rng)

    items = [
        (label, box_token(xyxy_box(box, width, height)))
        for box, label in selected
    ]
    return image, {
        "sample_id": f"sku110k-local-{Path(row['image_name']).stem}",
        "categories": ["object"],
        "target_response": grouped_response(items),
        "metadata": {
            "target_count": len(items),
            "image_size": [width, height],
        },
        "source_width": width,
        "source_height": height,
        "prompt": "detect all objects",
    }


# --- Referring: lmms-lab/RefCOCOg ---


def _local_referring_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    """Local lmms-lab/RefCOCOg adapter.

    Expected row fields (from Arrow dataset):
      image (PIL Image, embedded), question, answer (List[str]),
      bbox (List[float] length 4, COCO-style [x,y,w,h] in pixel coords),
      file_name, question_id.
    """
    del rng
    image = row.get("image")
    # Embedded PIL Image is the common case; fall back to image_from_value
    # with allow_network=False so URL images raise instead of fetching.
    if not hasattr(image, "size"):
        image = image_from_value(image, SOURCE_SPECS["referring"]["dataset"], allow_network=False)
    width, height = image.size

    bbox = row.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"RefCOCOg local row has invalid bbox: {bbox}")
    # bbox is [x, y, w, h] in pixel coords → convert to xyxy then normalize
    box = xywh_box(bbox, width, height)

    # Use answer[0] as referring phrase (local lmms-lab/RefCOCOg ships
    # natural-language captions, not the <bbox> token form expected by the
    # streaming adapter).
    answer = row.get("answer")
    if isinstance(answer, list) and len(answer) > 0:
        phrase = str(answer[0]).strip()
    elif isinstance(answer, str):
        phrase = answer.strip()
    else:
        phrase = ""

    if not phrase:
        raise ValueError(f"RefCOCOg local row has no referring phrase")

    return image, {
        "sample_id": f"refcocog-local-{row.get('question_id', index)}",
        "phrase": phrase,
        "target_response": f"<ref>{phrase}</ref>{box_token(box)}",
        "metadata": {"target_count": 1, "image_size": [int(width), int(height)]},
        "source_width": int(width),
        "source_height": int(height),
        "prompt": phrase,
    }


# --- Referring: sionic-ai/refcocog_object_detection ---


def _local_referring_sionic_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    """Local sionic-ai/refcocog_object_detection adapter.

    Expected row fields (from pinned HF dataset):
      image (PIL Image, embedded), question (str starting with ``[detect]``),
      answer (str containing ``<bbox>[x1,y1,x2,y2]</bbox>`` tokens in
      normalized 0--1000 coordinates).

    This is the same schema as the streaming ``referring_adapter`` but
    forces ``allow_network=False`` on ``image_from_value`` and uses
    ``LOCAL_SOURCE_SPECS`` provenance.
    """
    del rng
    phrase = re.sub(r"^\[detect\]\s*", "", str(row.get("question") or "")).strip()
    if not phrase:
        raise ValueError("sionic RefCOCOg row has no referring expression")
    match = re.search(
        r"<bbox>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</bbox>",
        str(row.get("answer") or ""),
    )
    if not match:
        raise ValueError("sionic RefCOCOg row has no normalized bbox answer")
    box = tuple(int(match.group(group)) for group in range(1, 5))
    # Coordinates are already normalized to [0,1000] in the answer text.
    for coord in box:
        if not 0 <= coord <= 1000:
            raise ValueError(
                f"sionic RefCOCOg bbox coordinate out of [0,1000]: {box}"
            )
    image = image_from_value(
        row.get("image"),
        SOURCE_SPECS["referring"]["dataset"],
        allow_network=False,
    )
    width, height = image.size
    return image, {
        "sample_id": f"refcocog-sionic-{Path(str(row.get('image_path', index))).stem}-{index}",
        "phrase": phrase,
        "target_response": f"<ref>{phrase}</ref>{box_token(box)}",
        "metadata": {"target_count": 1, "image_size": [int(width), int(height)]},
        "source_width": int(width),
        "source_height": int(height),
        "prompt": phrase,
    }


# --- Pointing: fail closed ---


def _local_pointing_adapter(
    row: dict[str, Any], index: int, _dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    """Convert a materialized PixMo candidate using a local image path only."""
    image_path = row.get("image_path")
    if not image_path:
        raise RuntimeError(
            "PixMo-Points remote URLs have no local image_path. URL-only metadata "
            "cannot be used for offline calibration."
        )
    image = image_from_value(
        str(image_path), SOURCE_SPECS["pointing"]["dataset"], allow_network=False
    )

    labels = row.get("label") or []
    point_groups = row.get("points") or []
    methods = row.get("collection_method") or []
    if isinstance(labels, str):
        labels = [labels]
    if point_groups and isinstance(point_groups[0], dict):
        point_groups = [point_groups]
    if isinstance(methods, str):
        methods = [methods]
    if not labels or len(labels) != len(point_groups):
        raise ValueError(
            "PixMo local row must have aligned non-empty label and points lists"
        )

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for label, points in zip(labels, point_groups):
        phrase = str(label).strip()
        if not phrase or not points:
            continue
        groups.append((phrase, list(points)))
    items: list[tuple[str, str]] = []
    point_index = 0
    while groups and len(items) < 48:
        added = False
        for phrase, points in groups:
            if point_index >= len(points):
                continue
            point = points[point_index]
            x = normalized_percent(point["x"])
            y = normalized_percent(point["y"])
            items.append((phrase, point_token((x, y))))
            added = True
            if len(items) >= 48:
                break
        if not added:
            break
        point_index += 1
    if not items:
        raise ValueError("PixMo local row has no usable label/point pairs")

    width, height = image.size
    image_sha256 = str(row.get("image_sha256") or "").strip()
    return image, {
        "sample_id": f"pixmo-local-{image_sha256 or index}",
        "target_response": grouped_response(items),
        "metadata": {
            "target_count": len(items),
            "collection_method": methods,
            "candidate_id": row.get("candidate_id"),
            "expected_source_sha256": image_sha256,
        },
        "source_width": int(width),
        "source_height": int(height),
        "prompt": "Point to the requested objects in the image.",
    }


def _local_pointing_single_label_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    image_path = row.get("image_path")
    if not image_path:
        raise RuntimeError("PixMo-Points row has no verified local image_path")
    image = image_from_value(
        str(image_path), SOURCE_SPECS["pointing"]["dataset"], allow_network=False
    )
    labels = row.get("label") or []
    point_groups = row.get("points") or []
    methods = row.get("collection_method") or []
    if isinstance(labels, str):
        labels = [labels]
    if point_groups and isinstance(point_groups[0], dict):
        point_groups = [point_groups]
    if isinstance(methods, str):
        methods = [methods]
    if not labels or len(labels) != len(point_groups):
        raise ValueError("PixMo row has misaligned label and point groups")

    candidates = []
    for group_index, (label, points) in enumerate(zip(labels, point_groups)):
        phrase = str(label).strip()
        if phrase and 1 <= len(points) <= 48:
            candidates.append((group_index, phrase, list(points)))
    if not candidates:
        raise ValueError("PixMo row has no complete single-label group within 48 points")
    group_index, phrase, points = candidates[rng.randrange(len(candidates))]
    items = [
        (
            phrase,
            point_token(
                (normalized_percent(point["x"]), normalized_percent(point["y"]))
            ),
        )
        for point in points
    ]
    width, height = image.size
    image_sha256 = str(row.get("image_sha256") or "").strip()
    method = methods[group_index] if group_index < len(methods) else None
    return image, {
        "sample_id": f"pixmo-local-{image_sha256 or index}",
        "phrase": phrase,
        "target_response": grouped_response(items),
        "metadata": {
            "target_count": len(items),
            "collection_method": method,
            "candidate_id": row.get("candidate_id"),
            "expected_source_sha256": image_sha256,
            "source_label_group_count": len(labels),
            "selected_label_index": group_index,
        },
        "source_width": int(width),
        "source_height": int(height),
        "prompt": f"Point to: {phrase}.",
    }


# --- GUI: local GroundCUA adapter ---


def _local_gui_adapter(
    row: dict[str, Any], index: int, _dataset: Any, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    """Local GroundCUA-train adapter.

    Expected row fields (from Arrow dataset):
      image (PIL Image, embedded or path), instructions (List[str]),
      bboxes (List[[x1,y1,x2,y2]]), image_id, inst_type, software.

    The adapter randomly selects one instruction/box pair, alternating
    between point (center of box) and box output on even/odd indices.
    Forced offline via ``allow_network=False``.
    """
    instructions = row.get("instructions") or []
    bboxes = row.get("bboxes") or []
    if not instructions:
        raise ValueError("GroundCUA local row has no instructions")
    if len(instructions) != len(bboxes):
        raise ValueError(
            f"GroundCUA local row has {len(instructions)} instructions "
            f"but {len(bboxes)} boxes"
        )
    choice = rng.randrange(len(instructions))
    phrase = str(instructions[choice]).strip()
    if not phrase:
        raise ValueError("GroundCUA local row has empty instruction phrase")

    image = image_from_value(
        row.get("image"), SOURCE_SPECS["gui"]["dataset"], allow_network=False
    )
    width, height = image.size
    # Validate box coordinates are within image bounds before normalization.
    # Also reject inverted boxes (x1 > x2 or y1 > y2): these produce a
    # nonsensical center point in point mode and
    # an inside-out box in box mode. Failing closed here is the explicit
    # Degenerate boxes must not silently produce an
    # invalid geometry.
    raw_x1, raw_y1, raw_x2, raw_y2 = [float(v) for v in bboxes[choice]]
    if raw_x1 < 0 or raw_y1 < 0 or raw_x2 > width or raw_y2 > height:
        raise ValueError(
            f"GroundCUA local row box {bboxes[choice]} outside image "
            f"{width}x{height}"
        )
    if raw_x1 > raw_x2 or raw_y1 > raw_y2:
        raise ValueError(
            f"GroundCUA local row box {bboxes[choice]} is inverted "
            f"(x1>x2 or y1>y2); cannot derive a valid point or box"
        )

    box = xyxy_box(bboxes[choice], width, height)
    if index % 2 == 0:
        point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        geometry = point_token(point)
        output_type = "point"
    else:
        geometry = box_token(box)
        output_type = "box"

    inst_types = row.get("inst_type") or []
    return image, {
        "sample_id": f"groundcua-local-{row.get('image_id', index)}-{choice}",
        "phrase": phrase,
        "output_type": output_type,
        "target_response": f"<ref>{phrase}</ref>{geometry}",
        "metadata": {
            "target_count": 1,
            "instruction_type": inst_types[choice] if choice < len(inst_types) else None,
            "software": row.get("software"),
        },
        "source_width": int(width),
        "source_height": int(height),
        "prompt": f"Locate the region that matches the following description: {phrase}.",
    }


# --- OCR/Layout: local versions of the streaming adapters that enforce
# allow_network=False and add the canonical ``prompt`` field. The local
# arrow datasets ship embedded PIL Image dicts (bytes), so the image gate is
# belt-and-braces rather than load-bearing.


def _local_ocr_adapter(
    row: dict[str, Any], index: int, _dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    items, parse_stats = parse_hiertext_with_stats(row.get("output_json_dumpsed"))
    if not items:
        raise ValueError("HierText local row has no parsed word boxes")
    image = image_from_value(row.get("image"), SOURCE_SPECS["ocr"]["dataset"], allow_network=False)
    width, height = image.size
    return image, {
        "sample_id": f"hiertext-local-{index}",
        "target_response": grouped_response(items),
        "metadata": {"target_count": len(items), "hiertext_filter": parse_stats},
        "source_width": int(width),
        "source_height": int(height),
        "prompt": "recognize text",
    }


def _local_layout_adapter(
    row: dict[str, Any], index: int, _dataset: Any, _: random.Random
) -> tuple[Any, dict[str, Any]]:
    image = image_from_value(row.get("image"), SOURCE_SPECS["layout"]["dataset"], allow_network=False)
    width, height = image.size
    grouped_boxes: dict[tuple[int, int, int, int, int], None] = {}
    bboxes = row.get("bboxes") or []
    category_ids = row.get("category_id") or []
    invalid_source_boxes = abs(len(category_ids) - len(bboxes))
    degenerate_after_normalization = 0
    for category, box in zip(category_ids, bboxes):
        if not box or len(box) != 4:
            invalid_source_boxes += 1
            continue
        normalized = xywh_box(box, width, height)
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            degenerate_after_normalization += 1
            continue
        grouped_boxes[(int(category), *normalized)] = None
    items: list[tuple[str, str]] = []
    categories: list[str] = []
    for category, x1, y1, x2, y2 in list(grouped_boxes)[:48]:
        label = doclaynet_label(category)
        categories.append(label)
        items.append((label, box_token((x1, y1, x2, y2))))
    if not items:
        raise ValueError("DocLayNet local row has no layout boxes")
    source_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return image, {
        "sample_id": f"doclaynet-local-{source_metadata.get('page_hash', index)}",
        "categories": list(dict.fromkeys(categories)),
        "target_response": grouped_response(items),
        "metadata": {
            "target_count": len(items),
            "document_category": source_metadata.get("collection"),
            "layout_filter": {
                "invalid_source_boxes": invalid_source_boxes,
                "degenerate_after_normalization": degenerate_after_normalization,
                "unique_valid_boxes": len(grouped_boxes),
            },
        },
        "source_width": int(width),
        "source_height": int(height),
        "prompt": "detect document layout elements",
    }

# ---------------------------------------------------------------------------
# Adapter registries
# ---------------------------------------------------------------------------

STREAMING_ADAPTERS: dict[str, Callable[..., tuple[Any, dict[str, Any]]]] = {
    "detection": detection_adapter,
    "gui": gui_adapter,
    "referring": referring_adapter,
    "ocr": ocr_adapter,
    "layout": layout_adapter,
    "pointing": pointing_adapter,
}

LOCAL_ADAPTERS: dict[str, Callable[..., tuple[Any, dict[str, Any]]]] = {
    "detection": _local_detection_adapter,
    "gui": _local_gui_adapter,
    "referring": _local_referring_adapter,
    "ocr": _local_ocr_adapter,
    "layout": _local_layout_adapter,
    "pointing": _local_pointing_adapter,
}

# Function-name → function map so ``LOCAL_SOURCE_IDENTITIES`` can bind an
# identity to its adapter by name (and so ``--local-source-task-key`` can
# select an alternate adapter without hard-coding task-specific switches).
# The sionic referring adapter lives outside ``LOCAL_ADAPTERS`` because the
# default referring adapter is the lmms-lab audit one; the sionic adapter is
# only swapped in when the sionic identity is selected (auto-detection or
# explicit ``--local-source-task-key``).
LOCAL_ADAPTERS_BY_NAME: dict[str, Callable[..., tuple[Any, dict[str, Any]]]] = {
    "_local_detection_adapter": _local_detection_adapter,
    "_local_gui_adapter": _local_gui_adapter,
    "_local_referring_adapter": _local_referring_adapter,
    "_local_referring_sionic_adapter": _local_referring_sionic_adapter,
    "_local_ocr_adapter": _local_ocr_adapter,
    "_local_layout_adapter": _local_layout_adapter,
    "_local_pointing_adapter": _local_pointing_adapter,
}

# Backwards-compat alias kept for any caller that referenced the sionic
# adapter by the old private name. The identity registry is now the single
# source of truth; this alias only avoids breaking external callers.
_SIONIC_REFERRING_ADAPTER = _local_referring_sionic_adapter

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_counts(values: list[str] | None) -> dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"invalid --count {value!r}; expected task=number")
        task, raw_count = value.split("=", 1)
        task = task.strip().lower()
        if task not in SOURCE_SPECS:
            raise ValueError(f"unsupported count task: {task}")
        counts[task] = int(raw_count)
    return counts


def parse_local_sources(
    values: list[str] | None,
) -> dict[str, Path]:
    """Parse --local-source task=path entries."""
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"--local-source must use TASK=PATH syntax, got: {value!r}"
            )
        task, raw_path = value.split("=", 1)
        task = task.strip().lower()
        if task not in SOURCE_SPECS:
            raise ValueError(f"unknown local source task: {task}")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"local source path for {task} does not exist: {path}"
            )
        if task in result:
            raise ValueError(f"duplicate --local-source for task: {task}")
        result[task] = path
    return result


def parse_local_source_task_keys(
    values: list[str] | None,
) -> dict[str, str]:
    """Parse --local-source-task-key task=identity_key entries.

    Returns a map of task → identity_key. Validates that each key is a
    registered identity for the task (the identity must be the task default
    or one of its alternates in ``LOCAL_SOURCE_ALTERNATES``). Raises
    ``ValueError`` for malformed entries, unknown tasks, unknown keys, or
    keys that are not valid for the named task.
    """
    result: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"--local-source-task-key must use TASK=KEY syntax, got: {value!r}"
            )
        task, raw_key = value.split("=", 1)
        task = task.strip().lower()
        raw_key = raw_key.strip()
        if task not in SOURCE_SPECS:
            raise ValueError(f"unknown local source task: {task}")
        if raw_key not in LOCAL_SOURCE_IDENTITIES:
            raise ValueError(
                f"unknown local source identity key: {raw_key!r}"
            )
        valid_for_task = {LOCAL_SOURCE_DEFAULT_KEYS[task]}
        valid_for_task.update(LOCAL_SOURCE_ALTERNATES.get(task, {}).values())
        if raw_key not in valid_for_task:
            raise ValueError(
                f"identity key {raw_key!r} is not valid for task {task!r}. "
                f"Valid keys: {sorted(valid_for_task)}"
            )
        if task in result:
            raise ValueError(
                f"duplicate --local-source-task-key for task: {task}"
            )
        result[task] = raw_key
    return result


def shutdown_fsspec_io() -> None:
    try:
        import fsspec.asyn as fsspec_async
    except ImportError:
        return

    loop = fsspec_async.loop[0]
    thread = fsspec_async.iothread[0]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive():
        thread.join(timeout=10)
    if thread is not None and thread.is_alive():
        print("[collect] warning: fsspec IO thread did not stop")
        return
    fsspec_async.reset_after_fork()


# ---------------------------------------------------------------------------
# Local data loaders (one per domain)
# ---------------------------------------------------------------------------


def _load_local_detection(
    source_dir: Path, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load SKU110K from CSV + JPEG files.

    Returns (rows, inventory) where each row is a dict with:
      image_path, image_name, boxes, image_width, image_height.

    The SKU110K_fixed release ships a headerless CSV (verified by G1 audit and
    by re-reading ``annotations/readme.txt``): columns are
    ``image_name,x1,y1,x2,y2,class,image_width,image_height``. We detect the
    headerless form by checking whether the first cell parses as a JPEG
    filename (``<split>_<n>.jpg``); if it does, the file is headerless and we
    consume the row as data. If the first row's first cell is literally
    ``image_name`` we treat the file as having a header (kept for the test
    fixture and any future re-export with a header).
    """

    # The SKU110K_fixed dataset has a triple-nested directory structure
    candidates = list(source_dir.glob("**/annotations_train.csv"))
    if not candidates:
        candidates = list(source_dir.glob("**/annotations/annotations_train.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"annotations_train.csv not found under {source_dir}"
        )
    csv_path = candidates[0]
    images_dir = csv_path.parent.parent / "images"
    if not images_dir.is_dir():
        # Try same-level images/
        images_dir = csv_path.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"images directory not found relative to {csv_path}"
        )

    # Column order is fixed by the upstream SKU110K readme.
    columns = [
        "image_name", "x1", "y1", "x2", "y2", "class",
        "image_width", "image_height",
    ]
    grouped: dict[str, dict[str, Any]] = {}
    header_seen = False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row or len(raw_row) < len(columns):
                continue
            first = raw_row[0].strip()
            # Detect a header row: the first cell is literally "image_name".
            if first == "image_name":
                header_seen = True
                continue
            # Reject obviously non-data rows.
            if not first or not first.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_name = first
            # Only train split: image_name starts with "train_"
            if not image_name.startswith("train_"):
                continue
            try:
                x1 = int(float(raw_row[1]))
                y1 = int(float(raw_row[2]))
                x2 = int(float(raw_row[3]))
                y2 = int(float(raw_row[4]))
                image_width = int(float(raw_row[6]))
                image_height = int(float(raw_row[7]))
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"could not parse SKU110K CSV row for {image_name}: {raw_row}"
                ) from exc
            entry = grouped.setdefault(
                image_name,
                {
                    "image_name": image_name,
                    "image_path": str(images_dir / image_name),
                    "image_width": image_width,
                    "image_height": image_height,
                    "boxes": [],
                },
            )
            entry["boxes"].append((x1, y1, x2, y2))

    rows = list(grouped.values())
    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(rows)

    # Verify at least one image exists
    existing = [r for r in rows if Path(r["image_path"]).is_file()]
    if not existing:
        raise FileNotFoundError(
            f"no train images found under {images_dir} (checked {len(rows)} CSV entries)"
        )

    inventory = {
        "csv_path": str(csv_path),
        "images_dir": str(images_dir),
        "csv_has_header": header_seen,
        "total_train_images_csv": len(rows),
        "images_verified_on_disk": len(existing),
        "total_boxes_csv": sum(len(r["boxes"]) for r in rows),
    }
    return existing, inventory


def _load_local_arrow(
    source_dir: Path, task: str, seed: int
) -> tuple[Any, dict[str, Any]]:
    """Load an Arrow dataset via datasets.load_from_disk(), enforce train split.

    Returns (dataset, inventory).

    README §3 forbids mixing val/test into formal calibration. A single-split
    dataset whose only split is named ``val``/``test``/``validation`` is
    therefore rejected — the fallback is only allowed for splits that are
    plausibly train (e.g. ``ocr_hiertext`` and ``pixmo-points`` ship a single
    ``train`` split).
    """
    from datasets import load_from_disk

    dataset = load_from_disk(str(source_dir))
    splits = list(dataset.keys()) if hasattr(dataset, "keys") else []
    inventory = {
        "source_dir": str(source_dir),
        "splits": splits,
        "loaded_split": None,
        "num_examples": 0,
    }

    _non_train_names = {"val", "validation", "test", "dev", "holdout"}

    if "train" in splits:
        ds = dataset["train"]
        inventory["loaded_split"] = "train"
    elif len(splits) == 1 and splits[0] not in _non_train_names:
        # Single-split dataset whose split name is not a known val/test alias —
        # treat it as train (e.g. ocr_hiertext, pixmo-points).
        ds = dataset[splits[0]]
        inventory["loaded_split"] = splits[0]
        inventory["warning"] = (
            f"single split {splits[0]!r} used as train (no explicit train split)"
        )
    else:
        available = ", ".join(splits) if splits else "none"
        raise ValueError(
            f"{task} dataset at {source_dir} has no train split. "
            f"Available splits: {available}. Formal calibration is train-only "
            "per README §3; val/test must not be mixed in."
        )

    inventory["num_examples"] = len(ds)
    inventory["features"] = list(ds.features.keys()) if hasattr(ds, "features") else []

    # Shuffle deterministically
    ds = ds.shuffle(seed=seed)
    return ds, inventory


def _load_local_gui(source_dir: Path, seed: int) -> tuple[Any, dict[str, Any]]:
    """Load GroundCUA-train from a local Arrow dataset.

    If the directory is empty the loader raises (fail-closed with a clear
    message).  When data is present it delegates to ``_load_local_arrow``
    which enforces train-only.

    Required-features check: after loading, the dataset
    MUST expose the fields the GroundCUA adapter expects — ``image``,
    ``instructions``, ``bboxes``. Without this check a malformed or wrong-schema
    snapshot (e.g. a partial download, or a different GroundCUA revision with
    renamed columns) would only blow up inside the adapter on the first row,
    masking the real problem as a per-row ``KeyError``. Failing at load time
    with an explicit message is the required fail-closed behaviour.
    """
    # Check for a dataset_dict.json first — the definitive sign of a
    # loadable Arrow dataset.
    if (source_dir / "dataset_dict.json").exists():
        ds, inventory = _load_local_arrow(source_dir, "gui", seed)
    else:
        contents = list(source_dir.iterdir()) if source_dir.is_dir() else []
        if not contents:
            raise RuntimeError(
                f"GroundCUA-train directory is empty ({source_dir}). "
                f"Run sync_hf_datasets.py to download the data first."
            )
        # Has files but no dataset_dict.json — try load_from_disk anyway
        ds, inventory = _load_local_arrow(source_dir, "gui", seed)

    # Validate the GroundCUA schema. ``image`` / ``instructions`` / ``bboxes``
    # are load-bearing for the adapter; ``image_id`` / ``inst_type`` /
    # ``software`` are optional metadata (the adapter tolerates their absence).
    required_features = ("image", "instructions", "bboxes")
    available = set(inventory.get("features") or [])
    missing = [name for name in required_features if name not in available]
    if missing:
        raise ValueError(
            f"GroundCUA-train at {source_dir} is missing required features "
            f"{missing}. Available features: {sorted(available) or 'none'}. "
            f"The local GUI adapter requires image/instructions/bboxes; "
            f"this snapshot is not the expected likaixin/GroundCUA-train schema."
        )
    return ds, inventory


def _load_local_pointing(source_dir: Path, seed: int) -> tuple[Any, dict[str, Any]]:
    """Load a small PixMo bundle whose rows reference verified local images."""
    ds, inventory = _load_local_arrow(source_dir, "pointing", seed)
    required = {"image_path", "image_sha256", "label", "points"}
    available = set(inventory.get("features") or [])
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "PixMo-Points images are remote URLs only or the local bundle is "
            f"incomplete. Missing local-cache fields: {missing}."
        )
    if len(ds) == 0:
        raise RuntimeError("PixMo local calibration bundle is empty")
    sample_path = Path(str(ds[0].get("image_path") or ""))
    if not sample_path.is_file():
        raise RuntimeError(
            f"PixMo local image cache is missing file: {sample_path}"
        )
    inventory["image_storage"] = "local_path"
    inventory["status"] = "ready"
    return ds, inventory


# ---------------------------------------------------------------------------
# Local collection entry point
# ---------------------------------------------------------------------------


def collect_domain_local(
    task: str,
    count: int,
    output_dir: Path,
    seed: int,
    resume: bool,
    source_dir: Path,
    *,
    local_source_identity_key: str | None = None,
    ocr_lossless: bool = False,
    layout_lossless: bool = False,
    pointing_single_label: bool = False,
) -> dict[str, Any]:
    """Collect *count* samples from a local source directory.

    Network access is never attempted.  Unavailable domains (GUI, Pointing)
    raise an error immediately.

    Safety: this entry point forces ``HF_HUB_OFFLINE=1`` and
    ``HF_DATASETS_OFFLINE=1`` before any loader runs, so even if a loader
    accidentally called a network API, the underlying libraries would refuse
    to fetch. The local adapters additionally call ``image_from_value`` with
    ``allow_network=False`` so URL-valued images raise instead of downloading.

    Provenance contract: the emitted ``source_dataset`` / ``source_revision``
    / ``split`` / ``source`` / ``license`` come from the *selected local
    source identity* (``LOCAL_SOURCE_IDENTITIES``), NOT from the streaming
    ``SOURCE_SPECS``. This prevents the source mis-attribution where
    lmms-lab/RefCOCOg data was written out labelled as
    ``sionic-ai/refcocog_object_detection``.

    Identity selection:
      * ``local_source_identity_key`` (from ``--local-source-task-key``)
        wins if provided; must be a valid key for *task*.
      * For ``referring`` the loader auto-detects the schema (sionic vs
        lmms-lab) and selects the matching identity. This binds the schema
        to the provenance so a sionic-schema row is never written out
        labelled as lmms-lab.
      * Otherwise the task default identity is used.

    Disabled identities (``enabled=False``) fail closed: audit/held-out
    data cannot enter formal calibration (README §3).
    """
    # Belt-and-braces: force offline mode for the duration of this call.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest_path = output_dir / f"{task}.jsonl"
    existing = read_jsonl(manifest_path) if resume else []
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"manifest already exists; use --resume: {manifest_path}")
    existing_ids = {record["sample_id"] for record in existing}
    accepted = len(existing)
    if accepted >= count:
        return {
            "task": task,
            "requested": count,
            "accepted": accepted,
            "failures": 0,
            "local_source": str(source_dir),
            "local_source_identity_key": local_source_identity_key,
        }

    spec = SOURCE_SPECS[task]
    rng = random.Random(seed)
    failures = Counter()
    scanned = 0
    inventory: dict[str, Any] = {}

    # Resolve the initial identity (explicit override or task default). For
    # referring this may be re-bound below after schema auto-detection.
    identity = select_local_source_identity(
        task, explicit_key=local_source_identity_key
    )
    adapter = resolve_adapter_for_identity(identity)
    if task == "pointing" and pointing_single_label:
        adapter = _local_pointing_single_label_adapter
    selected_identity_key = (
        local_source_identity_key or LOCAL_SOURCE_DEFAULT_KEYS[task]
    )

    # --- Load the dataset ---
    if task == "detection":
        rows, inventory = _load_local_detection(source_dir, seed)
        dataset_iter = enumerate(rows)
    elif task == "gui":
        ds, inventory = _load_local_gui(source_dir, seed)
        dataset_iter = enumerate(ds)
    elif task == "pointing":
        ds, inventory = _load_local_pointing(source_dir, seed)
        dataset_iter = enumerate(ds)
    else:
        # Arrow-based: referring, ocr, layout
        ds, inventory = _load_local_arrow(source_dir, task, seed)
        dataset_iter = enumerate(ds)

        # Auto-detect referring schema: sionic vs lmms-lab.
        # sionic-ai/refcocog_object_detection has ``answer`` as a str
        # containing ``<bbox>`` tokens; lmms-lab/RefCOCOg has ``answer``
        # as a List[str] of natural-language captions.
        #
        # The detected schema MUST re-bind the identity (and therefore
        # the provenance written to the manifest). If the user explicitly
        # passed --local-source-task-key we still validate that it is
        # consistent with the detected schema (a mismatch is a hard error
        # rather than a silent mis-attribution).
        if task == "referring":
            features = inventory.get("features", [])
            sample = ds[0] if len(ds) > 0 else {}
            answer_sample = sample.get("answer")
            if isinstance(answer_sample, str) and "<bbox>" in answer_sample:
                detected_schema_label = "sionic-ai/refcocog_object_detection"
            elif isinstance(answer_sample, list):
                detected_schema_label = "lmms-lab/RefCOCOg"
            else:
                raise ValueError(
                    f"referring dataset at {source_dir} has unrecognized schema. "
                    f"answer type: {type(answer_sample).__name__}. "
                    f"Expected sionic-ai (str with <bbox>) or lmms-lab (List[str])."
                )

            # Re-resolve the identity from the detected schema label, unless
            # the user explicitly pinned one — in which case it must match.
            if local_source_identity_key is not None:
                # User-pinned identity must be consistent with detected schema.
                expected_key = LOCAL_SOURCE_ALTERNATES["referring"].get(
                    detected_schema_label
                )
                if expected_key is not None and local_source_identity_key != expected_key:
                    raise ValueError(
                        f"--local-source-task-key={local_source_identity_key!r} "
                        f"disagrees with detected referring schema "
                        f"{detected_schema_label!r}. Expected identity "
                        f"{expected_key!r}. Refusing to write a "
                        f"mis-attributed manifest."
                    )
                # identity already resolved above; keep it.
            else:
                identity = select_local_source_identity(
                    "referring", schema_label=detected_schema_label
                )
                adapter = resolve_adapter_for_identity(identity)
                selected_identity_key = LOCAL_SOURCE_ALTERNATES["referring"][
                    detected_schema_label
                ]
            inventory["referring_schema"] = detected_schema_label
            inventory["referring_identity_key"] = selected_identity_key

    # Fail closed if the resolved identity is disabled.
    # (audit/held-out only). This keeps lmms-lab/RefCOCOg val/test
    # available for inspection while forbidding it from calibration.
    ensure_identity_enabled(task, identity)

    provenance = {
        "source": identity["source"],
        "source_dataset": identity["dataset"],
        "source_revision": identity["revision"],
        "split": identity["split"],
        "license": identity["license"],
    }

    for index, row in dataset_iter:
        if accepted >= count:
            break
        scanned += 1
        try:
            image, record = adapter(row, index, None, rng)
            if task == "ocr" and ocr_lossless:
                ensure_lossless_ocr_record(record)
            if task == "layout" and layout_lossless:
                ensure_lossless_layout_record(record)
            sample_id = str(record["sample_id"])
            if sample_id in existing_ids:
                continue
            image_name, image_sha256 = store_image(image, output_dir / "images")
            # Canonical source schema. Local provenance uses the resolved
            # identity so the manifest truthfully records the actual input
            # dataset (and its real revision), never the streaming reference.
            prompt = record.pop("prompt", "")
            output_record = {
                "schema_version": 1,
                "sample_id": sample_id,
                "task": task,
                "source": provenance["source"],
                "source_dataset": provenance["source_dataset"],
                "source_revision": provenance["source_revision"],
                "split": provenance["split"],
                "license": provenance["license"],
                "image": f"images/{image_name}",
                "image_sha256": image_sha256,
                "prompt": prompt,
                "source_width": record.pop("source_width", None),
                "source_height": record.pop("source_height", None),
                "acquisition": "local",
                "local_source": str(source_dir),
                "local_source_identity_key": selected_identity_key,
                **{key: value for key, value in record.items() if key != "sample_id"},
            }
            append_jsonl(manifest_path, output_record)
            existing_ids.add(sample_id)
            accepted += 1
            print(f"[collect:{task}] {accepted}/{count} {sample_id}")
        except Exception as exc:
            failures[type(exc).__name__] += 1
            append_jsonl(
                output_dir / "rejected.jsonl",
                {
                    "schema_version": 1,
                    "task": task,
                    "row_index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "local_source": str(source_dir),
                    "local_source_identity_key": selected_identity_key,
                },
            )
            print(f"[collect:{task}] skip row={index}: {type(exc).__name__}: {exc}")
            if sum(failures.values()) > max(100, count * 5):
                raise RuntimeError(
                    f"{task} exceeded failure budget: {dict(failures)}"
                ) from exc

    if accepted < count:
        raise RuntimeError(
            f"{task} collected {accepted}/{count} after scanning {scanned} rows"
        )
    return {
        "task": task,
        "requested": count,
        "accepted": accepted,
        "scanned": scanned,
        "failures": dict(failures),
        "manifest": manifest_path.name,
        "source_spec": spec,
        "local_source": str(source_dir),
        "local_source_identity_key": selected_identity_key,
        "local_provenance": provenance,
        "inventory": inventory,
    }


# ---------------------------------------------------------------------------
# Streaming (HF) collection — unchanged
# ---------------------------------------------------------------------------


def collect_domain(
    task: str,
    count: int,
    output_dir: Path,
    seed: int,
    shuffle_buffer: int,
    resume: bool,
    local_parquet: Path | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    spec = SOURCE_SPECS[task]
    manifest_path = output_dir / f"{task}.jsonl"
    existing = read_jsonl(manifest_path) if resume else []
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"manifest already exists; use --resume: {manifest_path}")
    existing_ids = {record["sample_id"] for record in existing}
    accepted = len(existing)
    if accepted >= count:
        return {"task": task, "requested": count, "accepted": accepted, "failures": 0}

    if local_parquet is not None:
        if task != "detection":
            raise ValueError("--local-parquet currently supports detection only")
        if not local_parquet.is_file():
            raise FileNotFoundError(f"local parquet does not exist: {local_parquet}")
        dataset = load_dataset(
            "parquet",
            data_files={"train": str(local_parquet)},
            split="train",
        ).shuffle(seed=seed)
        source_shuffle_buffer = None
    else:
        dataset = load_dataset(
            spec["dataset"],
            spec["config"],
            split=spec["split"],
            revision=spec["revision"],
            streaming=True,
        )
        source_shuffle_buffer = min(
            shuffle_buffer, int(spec.get("shuffle_buffer", shuffle_buffer))
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=source_shuffle_buffer)
    rng = random.Random(seed)
    failures = Counter()
    scanned = 0
    adapter = STREAMING_ADAPTERS[task]

    for index, row in enumerate(dataset):
        if accepted >= count:
            break
        scanned += 1
        try:
            image, record = adapter(row, index, dataset, rng)
            sample_id = str(record["sample_id"])
            if sample_id in existing_ids:
                continue
            image_name, image_sha256 = store_image(image, output_dir / "images")
            output_record = {
                "schema_version": 1,
                "sample_id": sample_id,
                "task": task,
                "source": spec["source"],
                "source_dataset": spec["dataset"],
                "source_revision": spec["revision"],
                "split": spec["split"],
                "license": spec["license"],
                "image": f"images/{image_name}",
                "image_sha256": image_sha256,
                "source_width": record.pop("source_width", None),
                "source_height": record.pop("source_height", None),
                **{key: value for key, value in record.items() if key != "sample_id"},
            }
            append_jsonl(manifest_path, output_record)
            existing_ids.add(sample_id)
            accepted += 1
            print(f"[collect:{task}] {accepted}/{count} {sample_id}")
        except Exception as exc:
            failures[type(exc).__name__] += 1
            print(f"[collect:{task}] skip row={index}: {type(exc).__name__}: {exc}")
            if sum(failures.values()) > max(100, count * 5):
                raise RuntimeError(
                    f"{task} exceeded failure budget: {dict(failures)}"
                ) from exc

    if accepted < count:
        raise RuntimeError(f"{task} collected {accepted}/{count} after scanning {scanned} rows")
    return {
        "task": task,
        "requested": count,
        "accepted": accepted,
        "scanned": scanned,
        "failures": dict(failures),
        "manifest": manifest_path.name,
        "source_spec": spec,
        "shuffle_buffer": source_shuffle_buffer,
        "local_parquet": str(local_parquet) if local_parquet else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(SOURCE_SPECS),
        default=list(SOURCE_SPECS),
    )
    parser.add_argument(
        "--count",
        action="append",
        help="override a collection count, for example --count detection=150",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument(
        "--local-source",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "use a local directory for a domain (offline, no network); repeatable. "
            "Example: --local-source detection=D:\\dataset\\SKU"
        ),
    )
    parser.add_argument(
        "--local-source-task-key",
        action="append",
        default=[],
        metavar="TASK=KEY",
        help=(
            "explicitly select a local source identity key for a task "
            "(source provenance override). Only meaningful together with "
            "--local-source. For referring, use "
            "'referring=referring_sionic_train' once the pinned sionic train "
            "arrow is on disk. If omitted, referring schema is auto-detected "
            "and the matching identity is selected."
        ),
    )
    parser.add_argument(
        "--local-parquet",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "[DEPRECATED] use --local-source instead. "
            "Currently detection only; repeatable."
        ),
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="optional Hugging Face endpoint, for example https://hf-mirror.com",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ocr-lossless",
        action="store_true",
        help=(
            "in local OCR mode, reject any row with malformed/collapsed labeled "
            "boxes or more than 48 labeled boxes instead of dropping labels"
        ),
    )
    parser.add_argument(
        "--layout-lossless",
        action="store_true",
        help="reject layout rows with invalid, collapsed, or more than 48 boxes",
    )
    parser.add_argument(
        "--pointing-single-label",
        action="store_true",
        help="emit one complete PixMo label group with an exact Point to prompt",
    )
    parser.add_argument(
        "--skip-python-finalization",
        action="store_true",
        help="work around PyArrow thread-state crashes after all outputs are fsynced",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = parse_counts(args.count)

    # --- Resolve local sources ---
    local_sources = parse_local_sources(args.local_source)
    # Resolve explicit local source identity keys.
    local_source_task_keys = parse_local_source_task_keys(args.local_source_task_key)
    # An identity key without a matching --local-source is a usage error.
    for task in local_source_task_keys:
        if task not in local_sources:
            raise ValueError(
                f"--local-source-task-key {task}=... requires a matching "
                f"--local-source {task}=PATH"
            )

    # --- Deprecated --local-parquet handling ---
    local_parquet: dict[str, Path] = {}
    for item in args.local_parquet:
        warnings.warn(
            "--local-parquet is deprecated; use --local-source instead. "
            "For detection, provide the CSV+JPEG root directory.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "=" not in item:
            raise ValueError("--local-parquet must use TASK=PATH syntax")
        task, raw_path = item.split("=", 1)
        if task not in SOURCE_SPECS:
            raise ValueError(f"unknown local parquet task: {task}")
        local_parquet[task] = Path(raw_path).expanduser().resolve()

    # --- Determine which mode to use ---
    summaries = []
    for domain_index, task in enumerate(args.domains):
        domain_seed = args.seed + domain_index * 1009

        if task in local_sources:
            # --- Local mode ---
            summaries.append(
                collect_domain_local(
                    task=task,
                    count=counts[task],
                    output_dir=output_dir,
                    seed=domain_seed,
                    resume=args.resume,
                    source_dir=local_sources[task],
                    local_source_identity_key=local_source_task_keys.get(task),
                    ocr_lossless=args.ocr_lossless,
                    layout_lossless=args.layout_lossless,
                    pointing_single_label=args.pointing_single_label,
                )
            )
        else:
            # --- Streaming mode ---
            summaries.append(
                collect_domain(
                    task=task,
                    count=counts[task],
                    output_dir=output_dir,
                    seed=domain_seed,
                    shuffle_buffer=args.shuffle_buffer,
                    resume=args.resume,
                    local_parquet=local_parquet.get(task),
                )
            )

    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "hf_endpoint": args.hf_endpoint or "https://huggingface.co",
        "local_sources": {task: str(path) for task, path in local_sources.items()},
        "local_source_identity_keys": local_source_task_keys,
        "domains": summaries,
    }
    atomic_write_json(output_dir / "collection_summary.json", summary)
    print(f"[collect] summary -> {output_dir / 'collection_summary.json'}")
    return 0


if __name__ == "__main__":
    skip_finalization = "--skip-python-finalization" in sys.argv
    try:
        exit_code = main()
    except BaseException:
        if skip_finalization:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        shutdown_fsspec_io()
    if skip_finalization:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
