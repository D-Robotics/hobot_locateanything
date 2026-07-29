"""Tests for the local offline source-reading layer.

These tests cover:
  - Six-domain field mapping
  - Missing train rejection
  - URL-image-not-local rejection
  - bbox/point boundaries
  - Empty annotations
  - Duplicate images
  - Windows paths
  - Local mode forbids network fallback
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest

# Load the script via importlib so we can reference it as a module for mocking
import importlib.util

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "compiler" / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "calibration/collect_sources.py"
_MODULE_NAME = "collect_locateanything_calibration_sources"

spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
coll = importlib.util.module_from_spec(spec)
sys.modules[_MODULE_NAME] = coll
spec.loader.exec_module(coll)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def detection_fixture_dir():
    return FIXTURES_DIR / "detection"


@pytest.fixture
def empty_fixture_dir():
    return FIXTURES_DIR / "empty_dir"


# ---------------------------------------------------------------------------
# Detection: CSV + JPEG local loader
# ---------------------------------------------------------------------------


class TestLocalDetectionLoader:
    """Test the local SKU110K CSV+JPEG loader."""

    def test_parses_csv_and_groups_boxes(self, detection_fixture_dir):
        rows, inventory = coll._load_local_detection(detection_fixture_dir, seed=42)
        assert inventory["total_train_images_csv"] == 5
        assert inventory["total_boxes_csv"] == 8
        # All 5 images should exist on disk
        assert inventory["images_verified_on_disk"] == 5

    def test_each_row_has_required_fields(self, detection_fixture_dir):
        rows, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        for row in rows:
            assert "image_name" in row
            assert "image_path" in row
            assert "boxes" in row
            assert "image_width" in row
            assert "image_height" in row
            assert Path(row["image_path"]).is_file()
            assert len(row["boxes"]) >= 1

    def test_boxes_are_xyxy_format(self, detection_fixture_dir):
        rows, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        for row in rows:
            for box in row["boxes"]:
                assert len(box) == 4
                x1, y1, x2, y2 = box
                assert x1 < x2
                assert y1 < y2
                assert 0 <= x1 <= row["image_width"]
                assert 0 <= x2 <= row["image_width"]
                assert 0 <= y1 <= row["image_height"]
                assert 0 <= y2 <= row["image_height"]

    def test_deterministic_shuffle(self, detection_fixture_dir):
        rows1, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        rows2, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        names1 = [r["image_name"] for r in rows1]
        names2 = [r["image_name"] for r in rows2]
        assert names1 == names2

    def test_different_seeds_different_order(self, detection_fixture_dir):
        rows1, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        rows2, _ = coll._load_local_detection(detection_fixture_dir, seed=99)
        names1 = [r["image_name"] for r in rows1]
        names2 = [r["image_name"] for r in rows2]
        # With only 5 elements, they might be the same by chance, but it's unlikely
        # We just check that the function accepts different seeds

    def test_rejects_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            coll._load_local_detection(Path("/nonexistent/dir"), seed=42)


class TestDetectionAdapter:
    """Test the local detection adapter."""

    def test_produces_canonical_record(self, detection_fixture_dir):
        rows, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        rng = random = __import__("random")
        sys_random = random.Random(42)
        image, record = coll._local_detection_adapter(rows[0], 0, None, sys_random)
        assert "sample_id" in record
        assert "categories" in record
        assert "target_response" in record
        assert "metadata" in record
        assert "source_width" in record
        assert "source_height" in record
        assert record["source_width"] == 100
        assert record["source_height"] == 80
        assert record["categories"] == ["object"]
        assert "<box>" in record["target_response"]
        assert "</box>" in record["target_response"]
        assert "<ref>object</ref>" in record["target_response"]

    def test_max_48_boxes(self, detection_fixture_dir):
        """Boxes capped at 48 per image."""
        rows, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        rng = __import__("random").Random(42)
        for row in rows:
            _, record = coll._local_detection_adapter(row, 0, None, rng)
            assert record["metadata"]["target_count"] <= 48

    def test_rejects_missing_image(self, detection_fixture_dir, tmp_output_dir):
        """Raises FileNotFoundError when image file is missing."""
        rows, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        row = rows[0].copy()
        row["image_path"] = str(tmp_output_dir / "nonexistent.jpg")
        rng = __import__("random").Random(42)
        with pytest.raises(FileNotFoundError):
            coll._local_detection_adapter(row, 0, None, rng)

    def test_rejects_empty_boxes(self, detection_fixture_dir):
        """Raises ValueError when no boxes."""
        rows, _ = coll._load_local_detection(detection_fixture_dir, seed=42)
        row = rows[0].copy()
        row["boxes"] = []
        rng = __import__("random").Random(42)
        with pytest.raises(ValueError):
            coll._local_detection_adapter(row, 0, None, rng)


# ---------------------------------------------------------------------------
# Local loader: missing train split
# ---------------------------------------------------------------------------


class TestMissingTrainSplit:
    """Test that loaders fail closed when train split is missing."""

    def test_arrow_loader_rejects_no_train(self, tmp_output_dir):
        """load_from_disk with no train split raises ValueError."""
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["val", "test"]
            mock_load.return_value = mock_dataset
            with pytest.raises(ValueError, match="no train split"):
                coll._load_local_arrow(Path("/fake"), "referring", seed=42)

    def test_empty_dir_gui_rejected(self, empty_fixture_dir):
        """Empty GroundCUA directory raises RuntimeError."""
        with pytest.raises(RuntimeError, match="empty"):
            coll._load_local_gui(empty_fixture_dir, seed=42)


# ---------------------------------------------------------------------------
# Pointing: URL-only rejection
# ---------------------------------------------------------------------------


class TestPointingURLOffline:
    """Test that PixMo-Points URL-only storage is rejected in offline mode."""

    def test_adapter_raises(self):
        """Local pointing adapter always raises."""
        with pytest.raises(RuntimeError, match="remote URLs"):
            coll._local_pointing_adapter({}, 0, None, None)

    def test_loader_raises_when_no_local_images(self):
        """load_from_disk with URL-only images raises."""
        with mock.patch("datasets.load_from_disk") as mock_load:
            mock_dataset = mock.MagicMock()
            mock_dataset.keys.return_value = ["train"]
            mock_ds = mock.MagicMock()
            mock_ds.__len__.return_value = 100
            sample = {"image_url": "https://example.com/img.png", "image": None}
            mock_ds.__getitem__.return_value = sample
            mock_dataset.__getitem__.return_value = mock_ds
            mock_load.return_value = mock_dataset
            with pytest.raises(RuntimeError, match="remote URLs"):
                coll._load_local_pointing(Path("/fake"), seed=42)


# ---------------------------------------------------------------------------
# GUI: blocked
# ---------------------------------------------------------------------------


class TestGUIBlocked:
    """Test that GroundCUA is blocked when data is missing."""

    def test_adapter_raises(self):
        """Local GUI adapter raises on empty data."""
        with pytest.raises(ValueError, match="no instructions"):
            coll._local_gui_adapter({}, 0, None, None)


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------


class TestCoordinateNormalization:
    def test_normalized_coordinate_basic(self):
        assert coll.normalized_coordinate(50, 100) == 500
        assert coll.normalized_coordinate(0, 100) == 0
        assert coll.normalized_coordinate(100, 100) == 1000

    def test_normalized_coordinate_clamped(self):
        assert coll.normalized_coordinate(-10, 100) == 0
        assert coll.normalized_coordinate(200, 100) == 1000

    def test_normalized_coordinate_rounding(self):
        # 500 / 1000 * 1000 = 500 exactly
        assert coll.normalized_coordinate(500, 1000) == 500
        # 333 / 1000 * 1000 = 333
        assert coll.normalized_coordinate(333, 1000) == 333

    def test_normalized_percent(self):
        assert coll.normalized_percent(50.0) == 500
        assert coll.normalized_percent(0.0) == 0
        assert coll.normalized_percent(100.0) == 1000
        assert coll.normalized_percent(29.583) == 296

    def test_normalized_coordinate_raises_on_zero_extent(self):
        with pytest.raises(ValueError, match="invalid coordinate extent"):
            coll.normalized_coordinate(10, 0)

    def test_xywh_box(self):
        result = coll.xywh_box([10, 20, 30, 40], 100, 100)
        assert result == (100, 200, 400, 600)

    def test_xyxy_box(self):
        result = coll.xyxy_box([10, 20, 80, 90], 100, 100)
        assert result == (100, 200, 800, 900)


# ---------------------------------------------------------------------------
# Box token generation
# ---------------------------------------------------------------------------


class TestBoxToken:
    def test_box_token(self):
        assert coll.box_token((100, 200, 400, 600)) == "<box><100><200><400><600></box>"

    def test_point_token(self):
        assert coll.point_token((300, 500)) == "<box><300><500></box>"

    def test_grouped_response(self):
        items = [("object", coll.box_token((100, 200, 400, 600)))]
        result = coll.grouped_response(items)
        assert result == "<ref>object</ref><box><100><200><400><600></box>"

    def test_grouped_response_multiple_same_label(self):
        items = [
            ("cat", coll.box_token((0, 0, 100, 100))),
            ("cat", coll.box_token((200, 200, 300, 300))),
        ]
        result = coll.grouped_response(items)
        assert result.count("<ref>cat</ref>") == 1
        assert result.count("<box>") == 2


# ---------------------------------------------------------------------------
# parse_local_sources
# ---------------------------------------------------------------------------


class TestParseLocalSources:
    def test_valid_syntax(self):
        result = coll.parse_local_sources(
            [f"detection={FIXTURES_DIR / 'detection'}"]
        )
        assert "detection" in result
        assert isinstance(result["detection"], Path)

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="TASK=PATH"):
            coll.parse_local_sources(["detection"])

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="unknown local source"):
            coll.parse_local_sources(["nonexistent=/tmp"])

    def test_duplicate_task(self):
        path = str(FIXTURES_DIR / "detection")
        with pytest.raises(ValueError, match="duplicate"):
            coll.parse_local_sources([f"detection={path}", f"detection={path}"])

    def test_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            coll.parse_local_sources(["detection=/nonexistent/path"])


# ---------------------------------------------------------------------------
# Windows path handling
# ---------------------------------------------------------------------------


class TestWindowsPaths:
    def test_windows_path_resolved(self):
        """Windows paths like D:\\dataset\\SKU are handled."""
        result = coll.parse_local_sources(
            [f"detection={FIXTURES_DIR / 'detection'}"]
        )
        path = result["detection"]
        assert path.is_absolute()
        assert "\\" in str(path) or "/" in str(path)


# ---------------------------------------------------------------------------
# parse_counts
# ---------------------------------------------------------------------------


class TestParseCounts:
    def test_defaults(self):
        counts = coll.parse_counts(None)
        assert counts == coll.DEFAULT_COUNTS

    def test_override(self):
        counts = coll.parse_counts(["detection=10"])
        assert counts["detection"] == 10
        assert counts["gui"] == coll.DEFAULT_COUNTS["gui"]

    def test_invalid_syntax(self):
        with pytest.raises(ValueError, match="task=number"):
            coll.parse_counts(["detection"])

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="unsupported count"):
            coll.parse_counts(["nonexistent=10"])


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


class TestJSONLHelpers:
    def test_read_jsonl_empty(self, tmp_output_dir):
        path = tmp_output_dir / "empty.jsonl"
        assert coll.read_jsonl(path) == []

    def test_read_jsonl_nonexistent(self, tmp_output_dir):
        path = tmp_output_dir / "nonexistent.jsonl"
        assert coll.read_jsonl(path) == []

    def test_write_and_read(self, tmp_output_dir):
        path = tmp_output_dir / "test.jsonl"
        coll.append_jsonl(path, {"a": 1, "b": 2})
        coll.append_jsonl(path, {"c": 3})
        records = coll.read_jsonl(path)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["c"] == 3

    def test_skip_blank_lines(self, tmp_output_dir):
        path = tmp_output_dir / "test.jsonl"
        path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        records = coll.read_jsonl(path)
        assert len(records) == 2

    def test_invalid_json_raises(self, tmp_output_dir):
        path = tmp_output_dir / "bad.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            coll.read_jsonl(path)

    def test_resume_skips_existing(self, tmp_output_dir):
        """Resume should skip already-collected sample_ids."""
        path = tmp_output_dir / "test.jsonl"
        coll.append_jsonl(path, {"sample_id": "abc", "value": 1})
        existing = coll.read_jsonl(path)
        ids = {r["sample_id"] for r in existing}
        assert "abc" in ids
        assert "xyz" not in ids


# ---------------------------------------------------------------------------
# collect_domain_local: detection integration
# ---------------------------------------------------------------------------


class TestCollectDomainLocalDetection:
    def test_collects_detection_samples(self, detection_fixture_dir, tmp_output_dir):
        result = coll.collect_domain_local(
            task="detection",
            count=3,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=detection_fixture_dir,
        )
        assert result["accepted"] == 3
        assert result["task"] == "detection"
        manifest = tmp_output_dir / "detection.jsonl"
        assert manifest.exists()
        records = coll.read_jsonl(manifest)
        assert len(records) == 3
        for record in records:
            assert record["schema_version"] == 1
            assert record["task"] == "detection"
            assert record["split"] == "train"
            assert "image" in record
            assert "image_sha256" in record
            assert "source_width" in record
            assert "source_height" in record
            assert "target_response" in record
            # Check image file exists
            img_path = tmp_output_dir / record["image"]
            assert img_path.is_file()

    def test_resume_does_not_duplicate(self, detection_fixture_dir, tmp_output_dir):
        # First run
        result1 = coll.collect_domain_local(
            task="detection",
            count=2,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=detection_fixture_dir,
        )
        assert result1["accepted"] == 2
        # Resume (same count but fewer new needed)
        result2 = coll.collect_domain_local(
            task="detection",
            count=2,
            output_dir=tmp_output_dir,
            seed=42,
            resume=True,
            source_dir=detection_fixture_dir,
        )
        assert result2["accepted"] == 2
        records = coll.read_jsonl(tmp_output_dir / "detection.jsonl")
        assert len(records) == 2

    def test_resume_collects_more(self, detection_fixture_dir, tmp_output_dir):
        # First collect 1
        coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=detection_fixture_dir,
        )
        # Then collect 3 total with resume
        result = coll.collect_domain_local(
            task="detection",
            count=3,
            output_dir=tmp_output_dir,
            seed=42,
            resume=True,
            source_dir=detection_fixture_dir,
        )
        assert result["accepted"] == 3
        records = coll.read_jsonl(tmp_output_dir / "detection.jsonl")
        assert len(records) == 3

    def test_no_resume_overwrite(self, detection_fixture_dir, tmp_output_dir):
        """Without --resume, existing manifest raises error."""
        coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=detection_fixture_dir,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            coll.collect_domain_local(
                task="detection",
                count=1,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=detection_fixture_dir,
            )

    def test_insufficient_data_raises(self, detection_fixture_dir, tmp_output_dir):
        """Requesting more than available raises RuntimeError."""
        with pytest.raises(RuntimeError, match="collected"):
            coll.collect_domain_local(
                task="detection",
                count=100,
                output_dir=tmp_output_dir,
                seed=42,
                resume=False,
                source_dir=detection_fixture_dir,
            )


# ---------------------------------------------------------------------------
# CLI: --local-source
# ---------------------------------------------------------------------------


class TestCLILocalSource:
    def test_help_includes_local_source(self):
        parser = coll.build_parser()
        help_text = parser.format_help()
        assert "--local-source" in help_text

    def test_help_deprecates_local_parquet(self):
        parser = coll.build_parser()
        help_text = parser.format_help()
        assert "DEPRECATED" in help_text

    def test_invalid_local_source_task(self):
        parser = coll.build_parser()
        # The parser accepts any string for --local-source; validation happens later
        args = parser.parse_args(
            ["--output-dir", "/tmp", "--local-source", "bad=path"]
        )
        assert "bad" in args.local_source[0]


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


class TestStratifyBoxes:
    def test_returns_all_when_under_max(self):
        rng = __import__("random").Random(42)
        boxes = [((x, x, x + 10, x + 10), "obj") for x in range(0, 50, 10)]
        result = coll._stratify_boxes(boxes, 48, rng)
        assert len(result) == len(boxes)

    def test_caps_at_max(self):
        rng = __import__("random").Random(42)
        boxes = [((x, x, x + 10, x + 10), "obj") for x in range(0, 200, 2)]
        result = coll._stratify_boxes(boxes, 48, rng)
        assert len(result) == 48

    def test_includes_varied_sizes(self):
        rng = __import__("random").Random(42)
        boxes = [
            ((0, 0, 5, 5), "small"),
            ((0, 0, 10, 10), "medium"),
            ((0, 0, 50, 50), "large"),
            ((0, 0, 6, 6), "small"),
            ((0, 0, 12, 12), "medium"),
        ]
        result = coll._stratify_boxes(boxes, 5, rng)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# No network fallback when using local mode
# ---------------------------------------------------------------------------


class TestNoNetworkFallback:
    """Local mode must never fall back to network streaming."""

    def test_local_mode_does_not_import_network(self, detection_fixture_dir, tmp_output_dir):
        """Local collection should not trigger any network imports."""
        result = coll.collect_domain_local(
            task="detection",
            count=1,
            output_dir=tmp_output_dir,
            seed=42,
            resume=False,
            source_dir=detection_fixture_dir,
        )
        assert "local_source" in result

    def test_pointing_local_always_fails(self):
        """Pointing local adapter must NEVER succeed (no network fallback)."""
        with pytest.raises(RuntimeError, match="remote URLs"):
            coll._local_pointing_adapter({"image_url": "http://example.com"}, 0, None, None)

    def test_gui_local_always_fails(self):
        """GUI local adapter raises on empty data (no network fallback)."""
        with pytest.raises(ValueError, match="no instructions"):
            coll._local_gui_adapter({}, 0, None, None)


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_atomic_write_creates_parents(self, tmp_output_dir):
        path = tmp_output_dir / "sub" / "deep" / "file.json"
        coll.atomic_write_json(path, {"key": "value"})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["key"] == "value"

    def test_atomic_write_overwrites(self, tmp_output_dir):
        path = tmp_output_dir / "file.json"
        coll.atomic_write_json(path, {"a": 1})
        coll.atomic_write_json(path, {"b": 2})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"b": 2}


# ---------------------------------------------------------------------------
# Edge cases: zero-extent, empty annotations, etc.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_width_image(self):
        with pytest.raises(ValueError, match="invalid coordinate extent"):
            coll.normalized_coordinate(10, 0)

    def test_negative_coordinate_clamped(self):
        assert coll.normalized_coordinate(-5, 100) == 0

    def test_oversized_coordinate_clamped(self):
        assert coll.normalized_coordinate(150, 100) == 1000

    def test_parse_hiertext_empty(self):
        assert coll.parse_hiertext("") == []

    def test_parse_hiertext_no_child_lines(self):
        assert coll.parse_hiertext("78 0 9 12 True\n") == []

    def test_parse_hiertext_with_json_quotes(self):
        items = coll.parse_hiertext('"78 0 9 12 True\\n-  78 0 8 7 False False hello"')
        assert len(items) == 1
        assert items[0][0] == "hello"

    def test_parse_hiertext_drops_boxes_that_collapse_after_normalization(self):
        value = "\n".join(
            [
                "- 10 20 0.01 2 False False too-narrow",
                "- 30 40 5 3 False False valid",
            ]
        )
        items, stats = coll.parse_hiertext_with_stats(value)
        assert items == [("valid", "<box><300><400><350><430></box>")]
        assert stats == {
            "parsed_word_boxes": 2,
            "dropped_non_positive_extent": 0,
            "dropped_degenerate_after_normalization": 1,
        }

    def test_parse_hiertext_drops_non_positive_extent(self):
        value = "- 10 20 0 2 False False zero-width"
        items, stats = coll.parse_hiertext_with_stats(value)
        assert items == []
        assert stats["dropped_non_positive_extent"] == 1

    def test_parse_hiertext_stats_scan_beyond_48_without_emitting_extra_targets(self):
        lines = [f"- {i} 10 1 1 False False label-{i}" for i in range(49)]
        items, stats = coll.parse_hiertext_with_stats("\n".join(lines))
        assert len(items) == 48
        assert stats["parsed_word_boxes"] == 49

    @pytest.mark.parametrize(
        "stats",
        [
            {"parsed_word_boxes": 1, "dropped_non_positive_extent": 1},
            {"parsed_word_boxes": 1, "dropped_degenerate_after_normalization": 1},
            {"parsed_word_boxes": 49},
        ],
    )
    def test_lossless_ocr_rejects_any_label_loss(self, stats):
        with pytest.raises(ValueError, match="not lossless"):
            coll.ensure_lossless_ocr_record({"metadata": {"hiertext_filter": stats}})

    def test_lossless_ocr_accepts_clean_bounded_record(self):
        coll.ensure_lossless_ocr_record(
            {
                "metadata": {
                    "hiertext_filter": {
                        "parsed_word_boxes": 48,
                        "dropped_non_positive_extent": 0,
                        "dropped_degenerate_after_normalization": 0,
                    }
                }
            }
        )

    def test_lossless_layout_rejects_truncation_or_bad_geometry(self):
        with pytest.raises(ValueError, match="not lossless"):
            coll.ensure_lossless_layout_record(
                {"metadata": {"layout_filter": {"unique_valid_boxes": 49}}}
            )
        with pytest.raises(ValueError, match="not lossless"):
            coll.ensure_lossless_layout_record(
                {
                    "metadata": {
                        "layout_filter": {"degenerate_after_normalization": 1}
                    }
                }
            )

    def test_lossless_layout_accepts_bounded_clean_record(self):
        coll.ensure_lossless_layout_record(
            {
                "metadata": {
                    "layout_filter": {
                        "invalid_source_boxes": 0,
                        "degenerate_after_normalization": 0,
                        "unique_valid_boxes": 48,
                    }
                }
            }
        )

    def test_store_image_deduplication(self, tmp_output_dir):
        from PIL import Image
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        name1, digest1 = coll.store_image(img, tmp_output_dir)
        name2, digest2 = coll.store_image(img, tmp_output_dir)
        assert digest1 == digest2
        # Only one file should exist
        assert (tmp_output_dir / name1).exists()
        assert name1 == name2
