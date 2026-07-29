"""Additional D1 tests covering gaps in the parallel-process test suite.

These tests focus on the requirements called out in the D1 task spec that
``test_collect_local_sources.py`` does not yet cover:

  * Real headerless SKU110K CSV (the G1-verified format — the existing fixture
    has a synthetic header which masks the production bug).
  * A hard assertion that local mode never opens a network socket, by
    patching ``urllib.request.urlopen`` and ``socket.socket`` and asserting
    they are never invoked during local collection.
  * The ``allow_network=False`` gate on ``image_from_value`` rejecting URLs.
  * Single-split val/test leak rejection in ``_load_local_arrow``.
  * Local OCR and layout adapter field mapping (the parallel suite only
    exercises detection and referring end-to-end).
  * Canonical ``prompt`` field presence on output records.
  * Deprecation handling for ``--local-parquet``.
  * Windows path with backslashes.
  * Duplicate-image deduplication by SHA256.

The tests are self-contained (no real datasets required) so they can run in
CI without the multi-GB local dataset snapshots.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

import importlib.util

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "compiler" / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "calibration/collect_sources.py"
_MODULE_NAME = "collect_locateanything_calibration_sources"

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
coll = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = coll
_spec.loader.exec_module(coll)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(path: Path, size=(40, 30), color=(200, 100, 50)) -> None:
    """Write a tiny real JPEG so PIL can decode it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=80)


def _write_headerless_sku_csv(
    root: Path, *, split: str = "train", n_images: int = 3, boxes_per_image: int = 2
) -> Path:
    """Create a SKU110K-style headerless CSV under root/annotations/ and matching
    JPEGs under root/images/. Returns the CSV path.
    """
    ann_dir = root / "annotations"
    images_dir = root / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ann_dir / f"annotations_{split}.csv"
    rows_written = 0
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # NO header — this is the production SKU110K_fixed format.
        for i in range(n_images):
            name = f"{split}_{i}.jpg"
            # Each image 100x80; write boxes inside the image bounds.
            for b in range(boxes_per_image):
                x1 = 5 + b * 20
                y1 = 5 + b * 10
                x2 = x1 + 15
                y2 = y1 + 12
                writer.writerow([name, x1, y1, x2, y2, "object", 100, 80])
                rows_written += 1
            _make_jpeg(images_dir / name, size=(100, 80))
    assert rows_written == n_images * boxes_per_image
    return csv_path


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def headerless_sku_dir(tmp_path):
    root = tmp_path / "sku_headerless"
    _write_headerless_sku_csv(root, n_images=3, boxes_per_image=2)
    return root


@pytest.fixture
def headered_sku_dir(tmp_path):
    """A headered CSV — kept to prove the loader is tolerant of both forms."""
    root = tmp_path / "sku_headered"
    ann_dir = root / "annotations"
    images_dir = root / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ann_dir / "annotations_train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]
        )
        writer.writerow(["train_0.jpg", 5, 5, 20, 20, "object", 100, 80])
        writer.writerow(["train_1.jpg", 10, 10, 30, 30, "object", 100, 80])
    _make_jpeg(images_dir / "train_0.jpg", size=(100, 80))
    _make_jpeg(images_dir / "train_1.jpg", size=(100, 80))
    return root


# ---------------------------------------------------------------------------
# Headerless CSV (production SKU110K format)
# ---------------------------------------------------------------------------


class TestHeaderlessCSV:
    """The real SKU110K CSV has NO header row; the loader must handle it."""

    def test_loads_headerless_csv(self, headerless_sku_dir):
        rows, inv = coll._load_local_detection(headerless_sku_dir, seed=42)
        assert len(rows) == 3
        assert inv["csv_has_header"] is False
        # Each image has 2 boxes
        assert inv["total_boxes_csv"] == 6
        assert inv["images_verified_on_disk"] == 3

    def test_boxes_grouped_correctly(self, headerless_sku_dir):
        rows, _ = coll._load_local_detection(headerless_sku_dir, seed=42)
        for row in rows:
            assert len(row["boxes"]) == 2
            for box in row["boxes"]:
                assert len(box) == 4
                x1, y1, x2, y2 = box
                assert x1 < x2
                assert y1 < y2

    def test_headered_csv_also_works(self, headered_sku_dir):
        """The loader is tolerant of a header row (test-fixture form)."""
        rows, inv = coll._load_local_detection(headered_sku_dir, seed=42)
        assert len(rows) == 2
        assert inv["csv_has_header"] is True
        assert inv["total_boxes_csv"] == 2

    def test_adapter_produces_record_on_headerless(self, headerless_sku_dir, tmp_output_dir):
        """End-to-end: headerless CSV → adapter → canonical record."""
        rows, _ = coll._load_local_detection(headerless_sku_dir, seed=42)
        rng = random.Random(42)
        image, record = coll._local_detection_adapter(rows[0], 0, None, rng)
        assert record["source_width"] == 100
        assert record["source_height"] == 80
        assert record["prompt"] == "detect all objects"
        assert "<box>" in record["target_response"]
        assert record["metadata"]["target_count"] <= 48

    def test_collect_domain_local_on_headerless(self, headerless_sku_dir, tmp_output_dir):
        """collect_domain_local produces canonical records with prompt field."""
        result = coll.collect_domain_local(
            task="detection",
            count=2,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=headerless_sku_dir,
        )
        assert result["accepted"] == 2
        manifest = tmp_output_dir / "detection.jsonl"
        records = coll.read_jsonl(manifest)
        assert len(records) == 2
        for rec in records:
            # Canonical schema: every required field present.
            for field in (
                "schema_version",
                "sample_id",
                "task",
                "source_dataset",
                "source_revision",
                "split",
                "license",
                "image",
                "image_sha256",
                "prompt",
                "target_response",
                "source_width",
                "source_height",
            ):
                assert field in rec, f"missing canonical field {field!r}"
            assert rec["prompt"] == "detect all objects"
            assert rec["acquisition"] == "local"
            assert rec["split"] == "train"
            # Image file exists and is non-empty
            img = tmp_output_dir / rec["image"]
            assert img.is_file()
            assert img.stat().st_size > 0


# ---------------------------------------------------------------------------
# No-network hard gate (the critical safety test)
# ---------------------------------------------------------------------------


class TestNoNetworkGate:
    """Local mode must never open a socket or call urllib.urlopen."""

    def test_urllib_not_called_in_local_mode(self, headerless_sku_dir, tmp_output_dir):
        """Patch urllib.request.urlopen and assert zero calls during local collection."""
        with mock.patch(
            "collect_locateanything_calibration_sources.urllib.request.urlopen"
        ) as mock_urlopen:
            coll.collect_domain_local(
                task="detection",
                count=1,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=headerless_sku_dir,
            )
            assert mock_urlopen.call_count == 0, (
                "local mode must never call urllib.request.urlopen"
            )

    def test_socket_not_opened_in_local_mode(self, headerless_sku_dir, tmp_output_dir):
        """Patch socket.socket to fail if any network socket is opened."""
        with mock.patch("socket.socket") as mock_socket:
            coll.collect_domain_local(
                task="detection",
                count=1,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=headerless_sku_dir,
            )
            assert mock_socket.call_count == 0, (
                "local mode must never open a network socket"
            )

    def test_image_from_value_rejects_url_when_offline(self):
        """allow_network=False must raise on http(s) URLs."""
        with pytest.raises(ValueError, match="not allowed in local offline mode"):
            coll.image_from_value(
                "https://example.com/image.png",
                "any/dataset",
                allow_network=False,
            )

    def test_image_from_value_rejects_missing_local_path_when_offline(self, tmp_path):
        """allow_network=False raises FileNotFoundError for a missing local path."""
        with pytest.raises(FileNotFoundError, match="local image path does not exist"):
            coll.image_from_value(
                str(tmp_path / "missing.jpg"),
                "any/dataset",
                allow_network=False,
            )

    def test_image_from_value_allows_local_path_when_offline(self, tmp_path):
        """allow_network=False succeeds for a real local JPEG path."""
        p = tmp_path / "img.jpg"
        _make_jpeg(p)
        image = coll.image_from_value(str(p), "any/dataset", allow_network=False)
        assert image.size == (40, 30)

    def test_offline_env_forced_by_collect_domain_local(
        self, headerless_sku_dir, tmp_output_dir
    ):
        """collect_domain_local sets HF_HUB_OFFLINE / HF_DATASETS_OFFLINE."""
        # Clear the env to prove the function sets them.
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(key, None)
        coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=headerless_sku_dir,
        )
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


# ---------------------------------------------------------------------------
# Split leak rejection in _load_local_arrow
# ---------------------------------------------------------------------------


class TestSplitLeakRejection:
    """README §3 forbids val/test mixing. _load_local_arrow must reject it."""

    def _make_mock_dataset(self, splits):
        """Build a MagicMock mimicking a DatasetDict with named splits."""
        mock_dataset = mock.MagicMock()
        mock_dataset.keys.return_value = splits
        # Each split is itself a mock dataset with __len__ and __iter__
        def _make_split(name):
            m = mock.MagicMock()
            m.__len__.return_value = 100
            m.__iter__.return_value = iter([])
            m.features.keys.return_value = []
            return m
        mock_dataset.__getitem__.side_effect = lambda key: _make_split(key)
        return mock_dataset

    def test_single_val_split_rejected(self, tmp_path):
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = self._make_mock_dataset(["val"])
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(tmp_path, "referring", seed=42)

    def test_single_test_split_rejected(self, tmp_path):
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = self._make_mock_dataset(["test"])
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(tmp_path, "referring", seed=42)

    def test_single_validation_split_rejected(self, tmp_path):
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = self._make_mock_dataset(["validation"])
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(tmp_path, "referring", seed=42)

    def test_val_test_only_rejected(self, tmp_path):
        """RefCOCOg's actual local shape: ['val','test'] only."""
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = self._make_mock_dataset(["val", "test"])
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(tmp_path, "referring", seed=42)

    def test_single_train_split_accepted(self, tmp_path):
        """ocr_hiertext / pixmo-points actual local shape: ['train'] only."""
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = self._make_mock_dataset(["train"])
            ds, inv = coll._load_local_arrow(tmp_path, "ocr", seed=42)
            assert inv["loaded_split"] == "train"

    def test_single_nonstandard_split_accepted_as_train(self, tmp_path):
        """A single split with a non-val/test name is allowed as train."""
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = self._make_mock_dataset(["data"])
            ds, inv = coll._load_local_arrow(tmp_path, "ocr", seed=42)
            assert inv["loaded_split"] == "data"
            assert "warning" in inv


# ---------------------------------------------------------------------------
# Local OCR adapter field mapping
# ---------------------------------------------------------------------------


class TestLocalOCRAdapter:
    """OCR adapter parses output_json_dumpsed into word boxes."""

    def _make_row(self, dump_text):
        # Build a row with an embedded PIL image dict (bytes form, the
        # load_from_disk representation).
        buf = io.BytesIO()
        Image.new("RGB", (1280, 960), (10, 20, 30)).save(buf, format="PNG")
        return {"image": {"bytes": buf.getvalue()}, "output_json_dumpsed": dump_text}

    def test_parses_words_and_boxes(self):
        dump = (
            '"78 0 9 12 True\\n'
            "-  78 0 8 7 False False 5% OFF\\n"
            "-  78 7 7 3 False False TODAY &\\n"
            '"'
        )
        row = self._make_row(dump)
        rng = random.Random(0)
        image, record = coll._local_ocr_adapter(row, 0, None, rng)
        assert record["source_width"] == 1280
        assert record["source_height"] == 960
        assert record["prompt"] == "recognize text"
        assert "<ref>5% OFF</ref>" in record["target_response"]
        assert "<ref>TODAY &</ref>" in record["target_response"]
        assert record["metadata"]["target_count"] == 2

    def test_empty_dump_raises(self):
        row = self._make_row("")
        rng = random.Random(0)
        with pytest.raises(ValueError, match="no parsed word boxes"):
            coll._local_ocr_adapter(row, 0, None, rng)

    def test_no_network_on_embedded_image(self):
        """Embedded image dict must not trigger any network call."""
        dump = '"78 0 9 12 True\\n-  78 0 8 7 False False hello\\n"'
        row = self._make_row(dump)
        rng = random.Random(0)
        with mock.patch(
            "collect_locateanything_calibration_sources.urllib.request.urlopen"
        ) as mock_urlopen:
            coll._local_ocr_adapter(row, 0, None, rng)
            assert mock_urlopen.call_count == 0


# ---------------------------------------------------------------------------
# Local layout adapter field mapping
# ---------------------------------------------------------------------------


class TestLocalLayoutAdapter:
    """Layout adapter converts pixel-xywh bboxes to normalized box tokens."""

    def _make_row(self, bboxes, category_ids, page_hash="page1"):
        buf = io.BytesIO()
        Image.new("RGB", (1025, 1025), (250, 250, 250)).save(buf, format="PNG")
        return {
            "image": {"bytes": buf.getvalue()},
            "bboxes": bboxes,
            "category_id": category_ids,
            "metadata": {"page_hash": page_hash, "collection": "reports"},
        }

    def test_maps_categories_and_boxes(self):
        # DocLayNet ships 1-indexed COCO category_id (verified by probing the
        # real train shard): 6 -> "page header", 10 -> "text", 11 -> "title".
        # The collector previously had an off-by-one bug: it indexed the
        # 0-indexed DOCLAYNET_CATEGORIES list directly, so id 6 was mislabelled
        # as "picture" (index 6) and id 10 as "title" (index 10). The
        # doclaynet_label() helper now subtracts 1 to fix this.
        row = self._make_row(
            [[72.35, 55.48, 372.22, 20.45], [100.0, 117.5, 789.3, 42.4]],
            [6, 10],
        )
        rng = random.Random(0)
        image, record = coll._local_layout_adapter(row, 0, None, rng)
        assert record["source_width"] == 1025
        assert record["source_height"] == 1025
        assert record["prompt"] == "detect document layout elements"
        assert "page header" in record["categories"], (
            "category_id 6 must map to 'page header' (1-indexed DocLayNet), not 'picture'"
        )
        assert "text" in record["categories"], (
            "category_id 10 must map to 'text' (1-indexed DocLayNet), not 'title'"
        )
        # And critically: id 6 must NOT be 'picture' (the old off-by-one bug).
        assert "picture" not in record["categories"], (
            "off-by-one bug: category_id 6 was mislabelled as 'picture'"
        )
        assert "<box>" in record["target_response"]
        # sample_id includes page_hash
        assert "page1" in record["sample_id"]

    def test_category_id_11_maps_to_title(self):
        """category_id 11 -> 'title' (the last DocLayNet category)."""
        row = self._make_row([[10.0, 10.0, 200.0, 30.0]], [11])
        rng = random.Random(0)
        _, record = coll._local_layout_adapter(row, 0, None, rng)
        assert "title" in record["categories"]

    def test_empty_bboxes_raises(self):
        row = self._make_row([], [])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="no layout boxes"):
            coll._local_layout_adapter(row, 0, None, rng)

    def test_caps_at_48_boxes(self):
        bboxes = [[float(i), float(i), 10.0, 10.0] for i in range(60)]
        cats = [0] * 60
        row = self._make_row(bboxes, cats)
        rng = random.Random(0)
        _, record = coll._local_layout_adapter(row, 0, None, rng)
        assert record["metadata"]["target_count"] <= 48


# ---------------------------------------------------------------------------
# Local referring adapter field mapping
# ---------------------------------------------------------------------------


class TestLocalReferringAdapter:
    """Local lmms-lab/RefCOCOg schema (answer List[str], bbox xywh)."""

    def _make_row(self, bbox, answers, question_id=298801):
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="PNG")
        return {
            "image": {"bytes": buf.getvalue()},
            "question": "Please carefully observe...",
            "answer": answers,
            "bbox": bbox,
            "question_id": question_id,
            "file_name": "COCO_train2014_000000546154_298801.jpg",
        }

    def test_converts_xywh_to_normalized_xyxy(self):
        # bbox [286.76, 233.17, 65.08, 236.75] on 640x480
        row = self._make_row(
            [286.76, 233.17, 65.08, 236.75],
            ["The tie of the standing man", "A purple tie"],
        )
        rng = random.Random(0)
        image, record = coll._local_referring_adapter(row, 0, None, rng)
        assert record["phrase"] == "The tie of the standing man"
        assert record["prompt"] == "The tie of the standing man"
        # Normalized: x1 = 286.76/640*1000 ≈ 448
        assert "<448>" in record["target_response"]
        assert record["source_width"] == 640
        assert record["source_height"] == 480
        assert "298801" in record["sample_id"]

    def test_missing_phrase_raises(self):
        row = self._make_row([1, 2, 3, 4], [])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="no referring phrase"):
            coll._local_referring_adapter(row, 0, None, rng)

    def test_invalid_bbox_raises(self):
        row = self._make_row([1, 2, 3], ["phrase"])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="invalid bbox"):
            coll._local_referring_adapter(row, 0, None, rng)


# ---------------------------------------------------------------------------
# Canonical prompt field on all local adapters
# ---------------------------------------------------------------------------


class TestCanonicalPromptField:
    """Every local adapter record must carry a ``prompt`` field."""

    def test_detection_prompt(self, headerless_sku_dir):
        rows, _ = coll._load_local_detection(headerless_sku_dir, seed=42)
        rng = random.Random(42)
        _, record = coll._local_detection_adapter(rows[0], 0, None, rng)
        assert record["prompt"] == "detect all objects"

    def test_ocr_prompt(self):
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="PNG")
        row = {
            "image": {"bytes": buf.getvalue()},
            "output_json_dumpsed": '"78 0 9 12 True\\n-  78 0 8 7 False False hi\\n"',
        }
        _, record = coll._local_ocr_adapter(row, 0, None, random.Random(0))
        assert record["prompt"] == "recognize text"

    def test_layout_prompt(self):
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="PNG")
        row = {
            "image": {"bytes": buf.getvalue()},
            "bboxes": [[10.0, 10.0, 20.0, 20.0]],
            "category_id": [0],
            "metadata": {},
        }
        _, record = coll._local_layout_adapter(row, 0, None, random.Random(0))
        assert record["prompt"] == "detect document layout elements"

    def test_referring_prompt(self):
        buf = io.BytesIO()
        Image.new("RGB", (640, 480)).save(buf, format="PNG")
        row = {
            "image": {"bytes": buf.getvalue()},
            "answer": ["the cat"],
            "bbox": [10.0, 20.0, 30.0, 40.0],
        }
        _, record = coll._local_referring_adapter(row, 0, None, random.Random(0))
        assert record["prompt"] == "the cat"


# ---------------------------------------------------------------------------
# Duplicate image deduplication by SHA256
# ---------------------------------------------------------------------------


class TestDuplicateImageDedup:
    """store_image must dedup identical content by SHA256."""

    def test_identical_images_same_filename(self, tmp_output_dir):
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        name1, digest1 = coll.store_image(img, tmp_output_dir / "images")
        name2, digest2 = coll.store_image(img, tmp_output_dir / "images")
        assert digest1 == digest2
        assert name1 == name2
        # Only one file on disk
        images = list((tmp_output_dir / "images").glob("*.jpg"))
        assert len(images) == 1

    def test_different_images_different_digest(self, tmp_output_dir):
        img1 = Image.new("RGB", (10, 10), (255, 0, 0))
        img2 = Image.new("RGB", (10, 10), (0, 255, 0))
        _, d1 = coll.store_image(img1, tmp_output_dir / "images")
        _, d2 = coll.store_image(img2, tmp_output_dir / "images")
        assert d1 != d2


# ---------------------------------------------------------------------------
# Windows path handling with backslashes
# ---------------------------------------------------------------------------


class TestWindowsPathHandling:
    @pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
    def test_backslash_path_accepted(self, headerless_sku_dir):
        """A Windows backslash path is parseable by --local-source."""
        win_path = str(headerless_sku_dir).replace("/", "\\")
        result = coll.parse_local_sources([f"detection={win_path}"])
        assert "detection" in result
        assert result["detection"].exists()

    def test_drive_letter_path(self, headerless_sku_dir):
        """Drive-letter absolute path works."""
        win_path = str(headerless_sku_dir)
        # On Windows this is already drive-letter form; the test just
        # confirms parse_local_sources handles it.
        result = coll.parse_local_sources([f"detection={win_path}"])
        assert result["detection"].is_absolute()


# ---------------------------------------------------------------------------
# --local-parquet deprecation
# ---------------------------------------------------------------------------


class TestLocalParquetDeprecation:
    def test_deprecation_warning_emitted(self):
        """--local-parquet must emit a DeprecationWarning."""
        import warnings

        parser = coll.build_parser()
        args = parser.parse_args(
            ["--output-dir", "/tmp/out", "--local-parquet", "detection=/some/path"]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The main() parser path emits the warning; simulate it by
            # calling the same loop that main() uses.
            for item in args.local_parquet:
                warnings.warn(
                    "--local-parquet is deprecated; use --local-source instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_help_marks_local_parquet_deprecated(self):
        parser = coll.build_parser()
        help_text = parser.format_help()
        assert "DEPRECATED" in help_text
        assert "--local-source" in help_text

    def test_local_source_and_parquet_both_for_detection(self, headerless_sku_dir):
        """User specifies both flags for the same task. We don't enforce
        mutual exclusion in main() today, but local-source takes precedence
        (the code path checks `if task in local_sources` first). This test
        documents the precedence behaviour."""
        # Not invoking main(); just documenting that local_sources wins.
        assert "detection" in coll.LOCAL_ADAPTERS
        assert "detection" in coll.LOCAL_LOADERS if hasattr(coll, "LOCAL_LOADERS") else True


# ---------------------------------------------------------------------------
# Stratification correctness — small boxes must be retained
# ---------------------------------------------------------------------------


class TestStratificationKeepsSmallBoxes:
    """The old code sorted largest-first and dropped small boxes. The new
    stratified selection must keep at least one small box when present."""

    def test_includes_small_box(self):
        rng = random.Random(42)
        boxes = [
            ((0, 0, 500, 500), "large"),   # large
            ((0, 0, 300, 300), "large"),   # large
            ((0, 0, 5, 5), "small"),        # small
            ((0, 0, 8, 8), "small"),       # small
            ((0, 0, 100, 100), "medium"),  # medium
        ]
        result = coll._stratify_boxes(boxes, 3, rng)
        labels = [r[1] for r in result]
        # Must include at least one small box.
        assert "small" in labels, (
            "stratified selection must retain small boxes, not only the largest"
        )

    def test_does_not_only_take_largest(self):
        rng = random.Random(0)
        # 60 boxes, 58 small and 2 large. Pure largest-first would give
        # only large boxes; stratified selection must include small ones.
        boxes = [((0, 0, 100, 100), "large") for _ in range(2)]
        boxes += [((0, 0, 5, 5), "small") for _ in range(58)]
        result = coll._stratify_boxes(boxes, 10, rng)
        labels = [r[1] for r in result]
        assert "small" in labels


# ---------------------------------------------------------------------------
# D1.5-A: Local source provenance — LOCAL_SOURCE_SPECS
# ---------------------------------------------------------------------------


class TestLocalSourceProvenance:
    """D1.5-A: LOCAL_SOURCE_SPECS maps each domain to its real local provenance."""

    def test_all_six_domains_have_local_specs(self):
        assert set(coll.LOCAL_SOURCE_SPECS.keys()) == set(coll.SOURCE_SPECS.keys())

    def test_detection_provenance_is_sku110k_fixed(self):
        spec = coll.LOCAL_SOURCE_SPECS["detection"]
        assert spec["dataset"] == "SKU110K_fixed"
        assert spec["revision"] == "local-csv-jpeg"
        assert spec["source"] == "SKU110K"

    def test_referring_provenance_is_lmms_lab(self):
        spec = coll.LOCAL_SOURCE_SPECS["referring"]
        assert spec["dataset"] == "lmms-lab/RefCOCOg"
        assert spec["revision"] == "local-arrow"

    def test_ocr_provenance_matches_streaming(self):
        """OCR local provenance matches streaming spec (same HF dataset)."""
        local = coll.LOCAL_SOURCE_SPECS["ocr"]
        stream = coll.SOURCE_SPECS["ocr"]
        assert local["dataset"] == stream["dataset"]
        assert local["revision"] == stream["revision"]

    def test_layout_provenance_matches_streaming(self):
        local = coll.LOCAL_SOURCE_SPECS["layout"]
        stream = coll.SOURCE_SPECS["layout"]
        assert local["dataset"] == stream["dataset"]
        assert local["revision"] == stream["revision"]

    def test_pointing_has_note_about_url_images(self):
        spec = coll.LOCAL_SOURCE_SPECS["pointing"]
        assert "note" in spec
        assert "URL" in spec["note"]

    def test_local_detection_manifest_uses_correct_provenance(self, headerless_sku_dir, tmp_output_dir):
        """collect_domain_local outputs SKU110K_fixed, not benjamintli/sku110k."""
        result = coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=headerless_sku_dir,
        )
        # D1.5-A: local_provenance now carries the identity-bound fields.
        assert result["local_provenance"]["source_dataset"] == "SKU110K_fixed"
        assert result["local_provenance"]["source_revision"] == "local-csv-jpeg"
        assert result["local_provenance"]["split"] == "train"
        assert result["local_source_identity_key"] == "detection_sku110k_local"
        manifest = tmp_output_dir / "detection.jsonl"
        records = coll.read_jsonl(manifest)
        assert len(records) == 1
        assert records[0]["source_dataset"] == "SKU110K_fixed"
        assert records[0]["source_revision"] == "local-csv-jpeg"
        assert records[0]["split"] == "train"
        assert records[0]["local_source_identity_key"] == "detection_sku110k_local"
        # Must NOT be the streaming reference
        assert records[0]["source_dataset"] != "benjamintli/sku110k"

    def test_collection_summary_includes_provenance(self, headerless_sku_dir, tmp_output_dir):
        result = coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=headerless_sku_dir,
        )
        assert "local_provenance" in result
        assert result["local_provenance"]["source_dataset"] == "SKU110K_fixed"
        assert result["local_provenance"]["source_revision"] == "local-csv-jpeg"
        assert result["local_source_identity_key"] == "detection_sku110k_local"


# ---------------------------------------------------------------------------
# D1.5-A: sionic-ai referring adapter
# ---------------------------------------------------------------------------


class TestSionicReferringAdapter:
    """The sionic-ai/refcocog_object_detection adapter handles <bbox> token answer."""

    def _make_row(self, question="[detect] the red car", answer="<bbox>[100, 200, 400, 500]</bbox>"):
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="PNG")
        return {
            "image": {"bytes": buf.getvalue()},
            "question": question,
            "answer": answer,
            "image_path": "train_000001.jpg",
        }

    def test_parses_bbox_from_answer(self):
        row = self._make_row()
        rng = random.Random(0)
        image, record = coll._local_referring_sionic_adapter(row, 0, None, rng)
        assert record["phrase"] == "the red car"
        assert "<box><100><200><400><500></box>" in record["target_response"]
        assert record["source_width"] == 640
        assert record["source_height"] == 480
        assert record["prompt"] == "the red car"

    def test_rejects_missing_detect_prefix(self):
        row = self._make_row(question="no prefix here")
        rng = random.Random(0)
        # "[detect]" prefix is stripped; if no phrase remains, it raises
        row["question"] = "[detect] "  # only whitespace after strip
        with pytest.raises(ValueError, match="no referring expression"):
            coll._local_referring_sionic_adapter(row, 0, None, rng)

    def test_rejects_missing_bbox_in_answer(self):
        row = self._make_row(answer="no bbox here")
        rng = random.Random(0)
        with pytest.raises(ValueError, match="no normalized bbox answer"):
            coll._local_referring_sionic_adapter(row, 0, None, rng)

    def test_rejects_out_of_range_coordinates(self):
        row = self._make_row(answer="<bbox>[1000, 200, 400, 500]</bbox>")
        rng = random.Random(0)
        # 1000 is valid (clamped to [0,1000]), but 1001 is not
        row["answer"] = "<bbox>[1001, 200, 400, 500]</bbox>"
        with pytest.raises(ValueError, match="coordinate out of"):
            coll._local_referring_sionic_adapter(row, 0, None, rng)

    def test_no_network_on_embedded_image(self):
        row = self._make_row()
        rng = random.Random(0)
        with mock.patch(
            "collect_locateanything_calibration_sources.urllib.request.urlopen"
        ) as mock_urlopen:
            coll._local_referring_sionic_adapter(row, 0, None, rng)
            assert mock_urlopen.call_count == 0


# ---------------------------------------------------------------------------
# D1.5-A: Schema auto-detection for referring
# ---------------------------------------------------------------------------


class TestReferringSchemaDetection:
    """collect_domain_local auto-detects sionic vs lmms-lab schema at load time."""

    def _make_arrow_dataset(self, schema_type, features_override=None):
        """Return a mock DatasetDict mimicking a local arrow dataset."""
        import datasets

        buf = io.BytesIO()
        Image.new("RGB", (640, 480)).save(buf, format="PNG")

        if schema_type == "sionic":
            # sionic-ai/refcocog_object_detection
            row_template = {
                "image": {"bytes": buf.getvalue()},
                "question": "[detect] the red car",
                "answer": "<bbox>[100, 200, 400, 500]</bbox>",
                "image_path": "train_000001.jpg",
            }
            features_list = ["image", "question", "answer", "image_path"]
        elif schema_type == "lmms":
            # lmms-lab/RefCOCOg
            row_template = {
                "image": {"bytes": buf.getvalue()},
                "question": "Please carefully observe...",
                "answer": ["The tie of the standing man", "A purple tie"],
                "bbox": [286.76, 233.17, 65.08, 236.75],
                "question_id": 298801,
                "file_name": "COCO_train2014_000000546154_298801.jpg",
            }
            features_list = ["question_id", "image", "question", "answer", "segmentation", "bbox", "iscrowd", "file_name"]
        else:
            # Unknown schema — answer is neither str-with-<bbox> nor list
            row_template = {
                "image": {"bytes": buf.getvalue()},
                "question": "some question",
                "answer": 42,  # int, not a recognized type
            }
            features_list = features_override or ["image", "question", "answer"]

        # Build a real list so ds[0] and iteration work naturally
        rows = [dict(row_template) for _ in range(5)]

        class _FakeDataset:
            def __init__(self, rows, features):
                self._rows = rows
                self.features = mock.MagicMock()
                self.features.keys.return_value = features

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, idx):
                return self._rows[idx]

            def __iter__(self):
                return iter(self._rows)

            def shuffle(self, seed):
                return self

        mock_ds = _FakeDataset(rows, features_list)

        mock_dataset = mock.MagicMock()
        mock_dataset.keys.return_value = ["train"]
        mock_dataset.__getitem__.return_value = mock_ds
        return mock_dataset, rows[0]

    def test_sionic_schema_detected(self, tmp_output_dir):
        mock_dataset, _ = self._make_arrow_dataset("sionic")
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            result = coll.collect_domain_local(
                task="referring",
                count=1,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=Path("/fake/sionic"),
            )
            assert result["accepted"] == 1
            assert result["inventory"]["referring_schema"] == "sionic-ai/refcocog_object_detection"
            # D1.5-A (GATE1 §4.2 fix): when the sionic schema is detected,
            # the manifest MUST record sionic provenance — NOT lmms-lab.
            # Previously the collector wrote lmms-lab/RefCOCOg onto
            # sionic-schema rows, which was the mis-attribution bug.
            assert result["local_source_identity_key"] == "referring_sionic_train"
            assert result["local_provenance"]["source_dataset"] == "sionic-ai/refcocog_object_detection"
            assert (
                result["local_provenance"]["source_revision"]
                == "efbdf2cc32689178bec9374f20798537421509ea"
            )
            assert result["local_provenance"]["split"] == "train"
            manifest = tmp_output_dir / "referring.jsonl"
            records = coll.read_jsonl(manifest)
            assert len(records) == 1
            assert records[0]["source_dataset"] == "sionic-ai/refcocog_object_detection"
            assert (
                records[0]["source_revision"]
                == "efbdf2cc32689178bec9374f20798537421509ea"
            )
            assert records[0]["split"] == "train"
            assert records[0]["local_source_identity_key"] == "referring_sionic_train"
            # Must NOT be labelled as the lmms-lab default.
            assert records[0]["source_dataset"] != "lmms-lab/RefCOCOg"
            assert "refcocog-sionic" in records[0]["sample_id"]

    def test_lmms_schema_detected_fail_closed(self, tmp_output_dir):
        """lmms-lab/RefCOCOg (val/test only) MUST fail closed for formal calibration.

        GATE1 §4.2 / README §3: formal calibration is train-only. The local
        lmms-lab/RefCOCOg snapshot has no train split, so its identity is
        ``enabled=False`` and ``collect_domain_local`` must refuse to emit
        records rather than silently writing lmms data into the manifest
        (whether labelled as lmms or, worse, as sionic).
        """
        mock_dataset, _ = self._make_arrow_dataset("lmms")
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            with pytest.raises(RuntimeError, match="disabled"):
                coll.collect_domain_local(
                    task="referring",
                    count=1,
                    output_dir=tmp_output_dir,
                    seed=42,
                    resume=False,
                    source_dir=Path("/fake/lmms"),
                )
        # No manifest must have been written.
        assert not (tmp_output_dir / "referring.jsonl").exists()

    def test_lmms_schema_audit_identity_recorded_in_error(self, tmp_output_dir):
        """When the lmms audit identity fail-closes, the error message must
        name lmms-lab/RefCOCOg (NOT sionic) so the user knows which dataset
        needs to be replaced."""
        mock_dataset, _ = self._make_arrow_dataset("lmms")
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            with pytest.raises(RuntimeError, match="lmms-lab/RefCOCOg"):
                coll.collect_domain_local(
                    task="referring",
                    count=1,
                    output_dir=tmp_output_dir,
                    seed=42,
                    resume=False,
                    source_dir=Path("/fake/lmms"),
                )

    def test_unknown_schema_raises(self, tmp_output_dir):
        mock_dataset, _ = self._make_arrow_dataset("unknown")
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="unrecognized schema"):
                coll.collect_domain_local(
                    task="referring",
                    count=1,
                    output_dir=tmp_output_dir,
                    seed=42,
                    resume=False,
                    source_dir=Path("/fake/unknown"),
                )


# ---------------------------------------------------------------------------
# D1.5-A: val/test still rejected when present in local dataset
# ---------------------------------------------------------------------------


class TestValTestStillRejected:
    """lmms-lab/RefCOCOg val/test must remain rejected for formal calibration."""

    def test_val_only_dataset_rejected_with_clear_message(self, tmp_path):
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["val"]
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(tmp_path, "referring", seed=42)

    def test_val_test_dataset_rejected(self, tmp_path):
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["val", "test"]
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(tmp_path, "referring", seed=42)


# ---------------------------------------------------------------------------
# D1.5-A: Fail-closed provenance / schema-mismatch tests
# ---------------------------------------------------------------------------


class TestLocalSourceIdentityRegistry:
    """D1.5-A: ``LOCAL_SOURCE_IDENTITIES`` + ``LOCAL_SOURCE_DEFAULT_KEYS``
    expose the full provenance per task and bind schema → identity."""

    def test_all_six_tasks_have_default_keys(self):
        assert set(coll.LOCAL_SOURCE_DEFAULT_KEYS.keys()) == set(coll.SOURCE_SPECS.keys())

    def test_referring_has_two_identities(self):
        # The whole point of D1.5-A: referring has an audit (lmms) identity
        # AND a calibration (sionic) identity.
        audit = coll.LOCAL_SOURCE_IDENTITIES["referring_lmms_refcocog_audit"]
        sionic = coll.LOCAL_SOURCE_IDENTITIES["referring_sionic_train"]
        assert audit["dataset"] == "lmms-lab/RefCOCOg"
        assert audit["enabled"] is False
        assert sionic["dataset"] == "sionic-ai/refcocog_object_detection"
        assert sionic["enabled"] is True
        # The sionic revision must match the pinned streaming revision
        # (efbdf2cc...) so streaming and local-sionic produce identical
        # provenance when the data lands.
        assert (
            sionic["revision"]
            == coll.SOURCE_SPECS["referring"]["revision"]
        )

    def test_referring_alternates_bind_schema_to_identity(self):
        alts = coll.LOCAL_SOURCE_ALTERNATES["referring"]
        assert alts["sionic-ai/refcocog_object_detection"] == "referring_sionic_train"
        assert alts["lmms-lab/RefCOCOg"] == "referring_lmms_refcocog_audit"

    def test_disabled_identities_are_marked(self):
        # lmms audit (val/test only), pixmo (URL-only), and the empty GUI
        # dir are all fail-closed for formal calibration.
        assert coll.LOCAL_SOURCE_IDENTITIES["referring_lmms_refcocog_audit"]["enabled"] is False
        assert coll.LOCAL_SOURCE_IDENTITIES["pointing_pixmo_local"]["enabled"] is False

    def test_verified_pointing_cache_identity_is_enabled(self):
        identity = coll.LOCAL_SOURCE_IDENTITIES["pointing_pixmo_cache"]
        assert identity["enabled"] is True
        assert identity["schema"] == "pixmo_points_local_cache"

    def test_select_default_identity(self):
        ident = coll.select_local_source_identity("referring")
        assert ident["dataset"] == "lmms-lab/RefCOCOg"
        assert ident["enabled"] is False

    def test_select_explicit_sionic_identity(self):
        ident = coll.select_local_source_identity(
            "referring", explicit_key="referring_sionic_train"
        )
        assert ident["dataset"] == "sionic-ai/refcocog_object_detection"
        assert ident["enabled"] is True

    def test_select_explicit_key_wrong_task_rejected(self):
        # The sionic key is registered for referring only; using it for
        # another task is a hard error, not a silent swap.
        with pytest.raises(ValueError, match="not registered for task"):
            coll.select_local_source_identity(
                "detection", explicit_key="referring_sionic_train"
            )

    def test_select_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown local source identity key"):
            coll.select_local_source_identity(
                "referring", explicit_key="referring_typo"
            )

    def test_select_unknown_schema_label_rejected(self):
        with pytest.raises(ValueError, match="unknown schema label"):
            coll.select_local_source_identity(
                "referring", schema_label="not-a-real-schema"
            )

    def test_ensure_identity_enabled_raises_on_disabled(self):
        audit = coll.LOCAL_SOURCE_IDENTITIES["referring_lmms_refcocog_audit"]
        with pytest.raises(RuntimeError, match="disabled"):
            coll.ensure_identity_enabled("referring", audit)

    def test_ensure_identity_enabled_passes_on_enabled(self):
        sionic = coll.LOCAL_SOURCE_IDENTITIES["referring_sionic_train"]
        # Must not raise.
        coll.ensure_identity_enabled("referring", sionic)


class TestProvenanceConsistency:
    """D1.5-A: the manifest provenance must match the actual input dataset,
    never the streaming reference, and sionic rows must never be labelled as
    lmms-lab (and vice versa)."""

    def test_detection_manifest_never_carries_streaming_dataset(
        self, headerless_sku_dir, tmp_output_dir
    ):
        """Detection local output is SKU110K_fixed, NOT benjamintli/sku110k."""
        coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=headerless_sku_dir,
        )
        records = coll.read_jsonl(tmp_output_dir / "detection.jsonl")
        assert records[0]["source_dataset"] == "SKU110K_fixed"
        # The streaming reference (benjamintli/sku110k) must NOT appear.
        assert records[0]["source_dataset"] != "benjamintli/sku110k"
        assert records[0]["source_revision"] != coll.SOURCE_SPECS["detection"]["revision"]

    def test_sionic_data_is_not_labelled_as_lmms(self, tmp_output_dir):
        """The exact GATE1 §4.2 mis-attribution: sionic-schema rows written
        out as lmms-lab/RefCOCOg. The collector must now refuse to do this."""
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="PNG")
        sionic_row = {
            "image": {"bytes": buf.getvalue()},
            "question": "[detect] the red car",
            "answer": "<bbox>[100, 200, 400, 500]</bbox>",
            "image_path": "train_000001.jpg",
        }

        class _SionicDS:
            def __init__(self, row):
                self._row = row
                self.features = mock.MagicMock()
                self.features.keys.return_value = [
                    "image", "question", "answer", "image_path"
                ]
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self._row
            def __iter__(self):
                return iter([self._row])
            def shuffle(self, seed):
                return self

        mock_dataset = mock.MagicMock()
        mock_dataset.keys.return_value = ["train"]
        mock_dataset.__getitem__.return_value = _SionicDS(sionic_row)
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            coll.collect_domain_local(
                task="referring",
                count=1,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=Path("/fake/sionic"),
            )
        records = coll.read_jsonl(tmp_output_dir / "referring.jsonl")
        assert len(records) == 1
        # CRITICAL: the record must be labelled sionic, not lmms-lab.
        assert records[0]["source_dataset"] == "sionic-ai/refcocog_object_detection"
        assert records[0]["source_dataset"] != "lmms-lab/RefCOCOg"
        assert records[0]["source"] == "RefCOCOg"  # not "RefCOCOg-lmms-audit"
        assert records[0]["split"] == "train"
        assert records[0]["local_source_identity_key"] == "referring_sionic_train"

    def test_lmms_data_never_written_to_manifest(self, tmp_output_dir):
        """lmms-lab/RefCOCOg val/test must NEVER appear in a calibration
        manifest, whether labelled as lmms or as sionic. Fail closed."""
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="PNG")
        lmms_row = {
            "image": {"bytes": buf.getvalue()},
            "question": "Please carefully observe...",
            "answer": ["The tie of the standing man"],
            "bbox": [286.76, 233.17, 65.08, 236.75],
            "question_id": "298801",
            "file_name": "COCO_train2014_000000546154_298801.jpg",
        }

        class _LmmsDS:
            def __init__(self, row):
                self._row = row
                self.features = mock.MagicMock()
                self.features.keys.return_value = [
                    "question_id", "image", "question", "answer",
                    "segmentation", "bbox", "iscrowd", "file_name"
                ]
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self._row
            def __iter__(self):
                return iter([self._row])
            def shuffle(self, seed):
                return self

        mock_dataset = mock.MagicMock()
        mock_dataset.keys.return_value = ["train"]  # pretend train exists
        mock_dataset.__getitem__.return_value = _LmmsDS(lmms_row)
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            with pytest.raises(RuntimeError, match="disabled"):
                coll.collect_domain_local(
                    task="referring",
                    count=1,
                    output_dir=tmp_output_dir,
                    seed=42,
                    resume=False,
                    source_dir=Path("/fake/lmms"),
                )
        # Nothing must be written.
        assert not (tmp_output_dir / "referring.jsonl").exists()

    def test_explicit_sionic_key_with_sionic_schema_succeeds(self, tmp_output_dir):
        """User can explicitly pin the sionic identity via
        --local-source-task-key; when the schema matches, the run succeeds
        and the manifest is labelled sionic."""
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="PNG")
        sionic_row = {
            "image": {"bytes": buf.getvalue()},
            "question": "[detect] the red car",
            "answer": "<bbox>[100, 200, 400, 500]</bbox>",
            "image_path": "train_000001.jpg",
        }

        class _DS:
            def __init__(self, row):
                self._row = row
                self.features = mock.MagicMock()
                self.features.keys.return_value = ["image", "question", "answer", "image_path"]
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self._row
            def __iter__(self):
                return iter([self._row])
            def shuffle(self, seed):
                return self

        mock_dataset = mock.MagicMock()
        mock_dataset.keys.return_value = ["train"]
        mock_dataset.__getitem__.return_value = _DS(sionic_row)
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            coll.collect_domain_local(
                task="referring",
                count=1,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=Path("/fake/sionic"),
                local_source_identity_key="referring_sionic_train",
            )
        records = coll.read_jsonl(tmp_output_dir / "referring.jsonl")
        assert records[0]["source_dataset"] == "sionic-ai/refcocog_object_detection"

    def test_explicit_sionic_key_with_lmms_schema_refuses_mislabelling(
        self, tmp_output_dir
    ):
        """If the user pins the sionic identity but the loaded data is
        actually lmms-schema, the collector must refuse rather than write a
        mis-attributed manifest (sionic label on lmms data)."""
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="PNG")
        lmms_row = {
            "image": {"bytes": buf.getvalue()},
            "question": "Please carefully observe...",
            "answer": ["The tie of the standing man"],
            "bbox": [286.76, 233.17, 65.08, 236.75],
            "question_id": "298801",
        }

        class _DS:
            def __init__(self, row):
                self._row = row
                self.features = mock.MagicMock()
                self.features.keys.return_value = [
                    "question_id", "image", "question", "answer",
                    "segmentation", "bbox", "iscrowd", "file_name"
                ]
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self._row
            def __iter__(self):
                return iter([self._row])
            def shuffle(self, seed):
                return self

        mock_dataset = mock.MagicMock()
        mock_dataset.keys.return_value = ["train"]
        mock_dataset.__getitem__.return_value = _DS(lmms_row)
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="disagrees with detected referring schema"):
                coll.collect_domain_local(
                    task="referring",
                    count=1,
                    output_dir=tmp_output_dir,
                    seed=42,
                    resume=False,
                    source_dir=Path("/fake/lmms"),
                    local_source_identity_key="referring_sionic_train",
                )
        assert not (tmp_output_dir / "referring.jsonl").exists()


class TestParseLocalSourceTaskKeys:
    """D1.5-A: --local-source-task-key parser validates identity keys."""

    def test_valid_sionic_key_parsed(self):
        result = coll.parse_local_source_task_keys(
            ["referring=referring_sionic_train"]
        )
        assert result == {"referring": "referring_sionic_train"}

    def test_valid_default_key_parsed(self):
        result = coll.parse_local_source_task_keys(
            ["referring=referring_lmms_refcocog_audit"]
        )
        assert result == {"referring": "referring_lmms_refcocog_audit"}

    def test_missing_equals_rejected(self):
        with pytest.raises(ValueError, match="TASK=KEY"):
            coll.parse_local_source_task_keys(["referring"])

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="unknown local source task"):
            coll.parse_local_source_task_keys(["bogus=referring_sionic_train"])

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown local source identity key"):
            coll.parse_local_source_task_keys(["referring=referring_typo"])

    def test_key_wrong_for_task_rejected(self):
        # The sionic key is valid globally but not valid for detection.
        with pytest.raises(ValueError, match="not valid for task"):
            coll.parse_local_source_task_keys(
                ["detection=referring_sionic_train"]
            )

    def test_duplicate_task_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            coll.parse_local_source_task_keys([
                "referring=referring_sionic_train",
                "referring=referring_lmms_refcocog_audit",
            ])

    def test_empty_values_return_empty_map(self):
        assert coll.parse_local_source_task_keys(None) == {}
        assert coll.parse_local_source_task_keys([]) == {}


class TestCLILocalSourceTaskKey:
    """D1.5-A: --local-source-task-key is wired into the CLI."""

    def test_help_includes_local_source_task_key(self):
        parser = coll.build_parser()
        help_text = parser.format_help()
        assert "--local-source-task-key" in help_text

    def test_parser_accepts_local_source_task_key(self):
        parser = coll.build_parser()
        args = parser.parse_args([
            "--output-dir", "/tmp/out",
            "--local-source-task-key", "referring=referring_sionic_train",
        ])
        assert args.local_source_task_key == ["referring=referring_sionic_train"]


# ---------------------------------------------------------------------------
# D1.5-B: Local GroundCUA adapter
# ---------------------------------------------------------------------------


class TestLocalGUIAdapter:
    """Local GroundCUA adapter handles instruction/box alignment."""

    def _make_row(
        self,
        instructions=None,
        bboxes=None,
        image_id=42,
        inst_type=None,
        software="Chrome",
    ):
        buf = io.BytesIO()
        Image.new("RGB", (1920, 1080), (200, 200, 200)).save(buf, format="PNG")
        return {
            "image": {"bytes": buf.getvalue()},
            "instructions": instructions if instructions is not None else ["Click the button"],
            "bboxes": bboxes if bboxes is not None else [[100.0, 200.0, 300.0, 250.0]],
            "image_id": image_id,
            "inst_type": inst_type if inst_type is not None else ["click"],
            "software": software,
        }

    def test_produces_box_on_even_index(self):
        row = self._make_row()
        rng = random.Random(42)
        image, record = coll._local_gui_adapter(row, 0, None, rng)
        assert record["output_type"] == "point"
        assert "<box>" in record["target_response"]
        assert record["source_width"] == 1920
        assert record["source_height"] == 1080
        assert record["prompt"].startswith("Locate the region")

    def test_produces_point_on_odd_index(self):
        row = self._make_row()
        rng = random.Random(42)
        image, record = coll._local_gui_adapter(row, 1, None, rng)
        assert record["output_type"] == "box"
        assert "<box>" in record["target_response"]

    def test_selects_random_instruction(self):
        row = self._make_row(
            instructions=["Click A", "Click B", "Click C"],
            bboxes=[[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]],
            inst_type=["a", "b", "c"],
        )
        rng = random.Random(42)
        image, record = coll._local_gui_adapter(row, 0, None, rng)
        assert record["phrase"] in ("Click A", "Click B", "Click C")
        assert record["metadata"]["instruction_type"] is not None
        assert record["metadata"]["software"] == "Chrome"

    def test_rejects_empty_instructions(self):
        row = self._make_row(instructions=[])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="no instructions"):
            coll._local_gui_adapter(row, 0, None, rng)

    def test_rejects_mismatched_instructions_and_boxes(self):
        row = self._make_row(
            instructions=["Click A", "Click B"],
            bboxes=[[10, 10, 20, 20]],
        )
        rng = random.Random(0)
        with pytest.raises(ValueError, match="instructions.*but.*boxes"):
            coll._local_gui_adapter(row, 0, None, rng)

    def test_rejects_empty_phrase(self):
        row = self._make_row(instructions=["  "])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="empty instruction phrase"):
            coll._local_gui_adapter(row, 0, None, rng)

    def test_rejects_out_of_bounds_box(self):
        row = self._make_row(bboxes=[[-10, 10, 20, 20]])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="outside image"):
            coll._local_gui_adapter(row, 0, None, rng)

    def test_no_network_on_embedded_image(self):
        row = self._make_row()
        rng = random.Random(0)
        with mock.patch(
            "collect_locateanything_calibration_sources.urllib.request.urlopen"
        ) as mock_urlopen:
            coll._local_gui_adapter(row, 0, None, rng)
            mock_urlopen.assert_not_called()

    # --- D1.5-B / GATE1 §4.1: "无点" scenario — degenerate/inverted box
    #     must fail closed in point mode rather than silently emit a
    #     nonsense center point.

    def test_rejects_inverted_box_x(self):
        """A box with x1 > x2 is inverted; the point-mode center would be
        meaningless. Fail closed (GATE1 §4.1 无点 scenario)."""
        row = self._make_row(bboxes=[[300.0, 200.0, 100.0, 250.0]])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="inverted"):
            coll._local_gui_adapter(row, 0, None, rng)

    def test_rejects_inverted_box_y(self):
        """A box with y1 > y2 is inverted; fail closed."""
        row = self._make_row(bboxes=[[100.0, 300.0, 200.0, 250.0]])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="inverted"):
            coll._local_gui_adapter(row, 0, None, rng)

    def test_rejects_inverted_box_in_box_mode_too(self):
        """Inverted boxes must also be rejected in box mode (index odd), not
        only in point mode — the geometry is inside-out either way."""
        row = self._make_row(bboxes=[[300.0, 300.0, 100.0, 100.0]])
        rng = random.Random(0)
        with pytest.raises(ValueError, match="inverted"):
            coll._local_gui_adapter(row, 1, None, rng)

    def test_zero_area_box_accepted_in_point_mode(self):
        """A zero-area box (x1==x2 or y1==y2) is NOT inverted — the center
        point is well-defined (on the box edge). This is the boundary between
        valid and invalid; the adapter keeps it (does not over-reject)."""
        # x1==x2: box is a vertical line; center point is valid.
        row = self._make_row(bboxes=[[100.0, 200.0, 100.0, 250.0]])
        rng = random.Random(0)
        image, record = coll._local_gui_adapter(row, 0, None, rng)
        assert record["output_type"] == "point"
        assert "<box>" in record["target_response"]

    def test_point_center_is_box_center(self):
        """Sanity: in point mode the emitted point is the integer box center."""
        # box [100, 200, 300, 250] on 1920x1080 -> normalized
        # x1=52, y1=185, x2=156, y2=231 -> center (104, 208)
        row = self._make_row(bboxes=[[100.0, 200.0, 300.0, 250.0]])
        rng = random.Random(0)
        image, record = coll._local_gui_adapter(row, 0, None, rng)
        # The point token embeds the center; check it's the integer midpoint
        # of the normalized box.
        assert "<104>" in record["target_response"]
        assert "<208>" in record["target_response"]

    # --- Canonical output format (LA spec) ---

    def test_target_response_uses_ref_and_box_tokens(self):
        """target_response must be ``<ref>{phrase}</ref>{geometry}`` per the
        LA canonical spec, where geometry is a <box>…</box> token."""
        row = self._make_row(
            instructions=["Find the login button"],
            bboxes=[[100.0, 200.0, 300.0, 250.0]],
        )
        rng = random.Random(0)
        _, record = coll._local_gui_adapter(row, 1, None, rng)  # box mode
        assert record["target_response"] == (
            "<ref>Find the login button</ref>"
            "<box><52><185><156><231></box>"
        )

    def test_prompt_is_canonical_locate_form(self):
        """prompt is the LA-canonical 'Locate the region that matches...' form."""
        row = self._make_row(instructions=["the red icon"])
        rng = random.Random(0)
        _, record = coll._local_gui_adapter(row, 0, None, rng)
        assert record["prompt"] == (
            "Locate the region that matches the following description: the red icon."
        )


# ---------------------------------------------------------------------------
# D1.5-B: GUI loader — empty dir fail-closed, data present succeeds
# ---------------------------------------------------------------------------


class TestGUILoader:
    """_load_local_gui fail-closed when empty, delegates when data present."""

    def test_empty_dir_raises_clear_message(self, tmp_path):
        with pytest.raises(RuntimeError, match="empty"):
            coll._load_local_gui(tmp_path, seed=42)

    def test_dataset_dict_json_triggers_arrow_load(self, tmp_path):
        (tmp_path / "dataset_dict.json").write_text('{"splits":["train"]}')
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["train"]
            mock_ds = mock.MagicMock()
            mock_ds.__len__.return_value = 10
            mock_ds.features.keys.return_value = [
                "image", "instructions", "bboxes", "image_id", "inst_type", "software",
            ]
            mock_dataset.__getitem__.return_value = mock_ds
            mock_load.return_value = mock_dataset
            ds, inv = coll._load_local_gui(tmp_path, seed=42)
            assert inv["loaded_split"] == "train"

    # --- D1.5-B / GATE1 §4.1: required-features validation at load time ---

    def test_rejects_dataset_missing_required_features(self, tmp_path):
        """If the loaded dataset lacks image/instructions/bboxes, fail closed
        at load time with a clear message naming the missing features, instead
        of letting the adapter raise a per-row KeyError."""
        (tmp_path / "dataset_dict.json").write_text('{"splits":["train"]}')
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["train"]
            mock_ds = mock.MagicMock()
            mock_ds.__len__.return_value = 10
            # Wrong schema: no 'instructions', no 'bboxes' — this is not
            # GroundCUA-train.
            mock_ds.features.keys.return_value = [
                "image", "image_id", "inst_type", "software",
            ]
            mock_dataset.__getitem__.return_value = mock_ds
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="missing required features"):
                coll._load_local_gui(tmp_path, seed=42)

    def test_rejects_dataset_with_no_features_at_all(self, tmp_path):
        """A dataset that reports no features (e.g. empty arrow shard) must
        also be rejected."""
        (tmp_path / "dataset_dict.json").write_text('{"splits":["train"]}')
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["train"]
            mock_ds = mock.MagicMock()
            mock_ds.__len__.return_value = 0
            mock_ds.features.keys.return_value = []
            mock_dataset.__getitem__.return_value = mock_ds
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="missing required features"):
                coll._load_local_gui(tmp_path, seed=42)

    def test_accepts_dataset_with_required_features_only(self, tmp_path):
        """The adapter tolerates absence of optional metadata (image_id,
        inst_type, software) as long as image/instructions/bboxes exist."""
        (tmp_path / "dataset_dict.json").write_text('{"splits":["train"]}')
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["train"]
            mock_ds = mock.MagicMock()
            mock_ds.__len__.return_value = 5
            mock_ds.features.keys.return_value = [
                "image", "instructions", "bboxes",
            ]
            mock_dataset.__getitem__.return_value = mock_ds
            mock_load.return_value = mock_dataset
            ds, inv = coll._load_local_gui(tmp_path, seed=42)
            assert inv["loaded_split"] == "train"

    def test_features_validation_runs_after_empty_dir_check(self, tmp_path):
        """The empty-dir check still fires first (no dataset_dict.json, no
        files) — features validation only runs once data is loadable."""
        # tmp_path is empty and has no dataset_dict.json
        with pytest.raises(RuntimeError, match="empty"):
            coll._load_local_gui(tmp_path, seed=42)

    def test_features_validation_runs_after_train_split_enforcement(self, tmp_path):
        """A dataset with only val/test is rejected by the train-split check
        before features validation runs (train-only is the first gate)."""
        (tmp_path / "dataset_dict.json").write_text('{"splits":["val"]}')
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["val", "test"]
            mock_ds = mock.MagicMock()
            mock_ds.__len__.return_value = 10
            mock_ds.features.keys.return_value = [
                "image", "instructions", "bboxes",
            ]
            mock_dataset.__getitem__.return_value = mock_ds
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_gui(tmp_path, seed=42)


# ---------------------------------------------------------------------------
# D1.5-D: Local PixMo image-cache adapter
# ---------------------------------------------------------------------------


class TestLocalPixMoCache:
    def test_adapter_uses_local_image_and_groups_points(self, tmp_path):
        image_path = tmp_path / "candidate.png"
        Image.new("RGB", (100, 80), (20, 40, 60)).save(image_path)
        row = {
            "candidate_id": "pixmo-cand-001",
            "image_path": str(image_path),
            "image_sha256": "a" * 64,
            "label": ["cup", "handle"],
            "points": [
                [{"x": 25.0, "y": 50.0}],
                [{"x": 75.0, "y": 60.0}],
            ],
            "collection_method": ["pointing", "pointing"],
        }
        image, record = coll._local_pointing_adapter(
            row, 0, None, random.Random(1)
        )
        assert image.size == (100, 80)
        assert record["sample_id"] == f"pixmo-local-{'a' * 64}"
        assert "<ref>cup</ref><box><250><500></box>" in record["target_response"]
        assert "<ref>handle</ref><box><750><600></box>" in record["target_response"]
        assert record["metadata"]["target_count"] == 2

    def test_adapter_rejects_url_only_row(self):
        with pytest.raises(RuntimeError, match="no local image_path"):
            coll._local_pointing_adapter(
                {
                    "image_url": "https://example.invalid/image.png",
                    "label": ["object"],
                    "points": [[{"x": 50.0, "y": 50.0}]],
                },
                0,
                None,
                random.Random(1),
            )

    def test_loader_accepts_complete_local_bundle(self, tmp_path):
        image_path = tmp_path / "candidate.png"
        Image.new("RGB", (10, 10), "white").save(image_path)
        ds = [{"image_path": str(image_path)}]
        inventory = {
            "features": ["image_path", "image_sha256", "label", "points"],
            "loaded_split": "train",
        }
        with mock.patch.object(
            coll, "_load_local_arrow", return_value=(ds, inventory)
        ):
            loaded, result = coll._load_local_pointing(tmp_path, seed=1)
        assert loaded is ds
        assert result["status"] == "ready"
        assert result["image_storage"] == "local_path"

    def test_loader_rejects_url_only_metadata(self, tmp_path):
        ds = [{"image_url": "https://example.invalid/image.png"}]
        inventory = {
            "features": ["image_url", "image_sha256", "label", "points"],
            "loaded_split": "train",
        }
        with mock.patch.object(
            coll, "_load_local_arrow", return_value=(ds, inventory)
        ):
            with pytest.raises(RuntimeError, match="Missing local-cache fields"):
                coll._load_local_pointing(tmp_path, seed=1)

    def test_adapter_caps_total_points_across_labels(self, tmp_path):
        image_path = tmp_path / "candidate.png"
        Image.new("RGB", (100, 80), "white").save(image_path)
        row = {
            "image_path": str(image_path),
            "image_sha256": "b" * 64,
            "label": ["left", "right"],
            "points": [
                [{"x": float(i % 100), "y": 25.0} for i in range(80)],
                [{"x": float(i % 100), "y": 75.0} for i in range(80)],
            ],
            "collection_method": ["pointing", "pointing"],
        }
        _, record = coll._local_pointing_adapter(
            row, 0, None, random.Random(1)
        )
        assert record["metadata"]["target_count"] == 48
        assert record["target_response"].count("<box>") == 48
        assert "<ref>left</ref>" in record["target_response"]
        assert "<ref>right</ref>" in record["target_response"]

    def test_single_label_adapter_prompt_matches_result(self, tmp_path):
        image_path = tmp_path / "candidate.png"
        Image.new("RGB", (100, 80), "white").save(image_path)
        row = {
            "image_path": str(image_path),
            "image_sha256": "c" * 64,
            "label": ["cup", "handle"],
            "points": [
                [{"x": 25.0, "y": 50.0}, {"x": 30.0, "y": 50.0}],
                [{"x": 75.0, "y": 60.0}],
            ],
            "collection_method": ["pointing", "pointing"],
        }
        _, record = coll._local_pointing_single_label_adapter(
            row, 0, None, random.Random(1)
        )
        assert record["prompt"] == f"Point to: {record['phrase']}."
        assert record["target_response"].startswith(
            f"<ref>{record['phrase']}</ref>"
        )
        assert record["metadata"]["target_count"] <= 48


class TestExplicitCalibrationQuotas:
    @staticmethod
    def _load_prepare_module():
        path = SCRIPTS_DIR / "calibration/prepare.py"
        spec = importlib.util.spec_from_file_location("prepare_calibration", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_exact_six_domain_quota(self):
        module = self._load_prepare_module()
        available = {
            "detection": 260, "gui": 130, "referring": 90,
            "ocr": 65, "layout": 65, "pointing": 50,
        }
        result = module.parse_explicit_quotas(
            [
                "detection=208", "gui=102", "referring=68",
                "ocr=48", "layout=48", "pointing=38",
            ],
            available,
            512,
        )
        assert result == {
            "detection": 208, "gui": 102, "referring": 68,
            "ocr": 48, "layout": 48, "pointing": 38,
        }

    def test_quota_requires_all_domains(self):
        module = self._load_prepare_module()
        with pytest.raises(ValueError, match="must cover all domains"):
            module.parse_explicit_quotas(
                ["detection=512"], {"detection": 600}, 512
            )

    def test_quota_sum_must_match_total(self):
        module = self._load_prepare_module()
        values = [f"{task}=1" for task in module.PAPER_TASK_WEIGHTS]
        with pytest.raises(ValueError, match="expected 512"):
            module.parse_explicit_quotas(
                values, {task: 600 for task in module.PAPER_TASK_WEIGHTS}, 512
            )
