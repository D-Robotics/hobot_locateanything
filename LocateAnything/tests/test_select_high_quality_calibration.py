import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "calibration/select_quality.py"
spec = importlib.util.spec_from_file_location("select_high_quality_calibration", SCRIPT)
selector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(selector)


def row(task, response, **extra):
    value = {
        "task": task,
        "phrase": "button",
        "target_response": response,
        "metadata": {"target_count": 1},
    }
    value.update(extra)
    return value


def test_prompt_is_canonical_and_preserves_source_prompt():
    item = row("gui", "<ref>button</ref><box><10><20><30><40></box>", prompt="old")
    assert selector.response_valid(item) == (True, "")
    assert selector.canonical_prompt(item) == (
        "Locate the region that matches the following description: button."
    )


def test_detection_preserves_multicategory_prompt_and_refs():
    item = row(
        "detection",
        "<ref>person</ref><box><10><20><30><40></box>"
        "<ref>motorcycle</ref><box><50><60><90><100></box>",
        categories=["person", "motorcycle"],
        metadata={"target_count": 2},
    )
    assert selector.response_valid(item) == (True, "")
    assert selector.canonical_prompt(item) == (
        "Locate all the instances that matches the following description: "
        "person</c>motorcycle."
    )


def test_detection_rejects_category_ref_mismatch():
    item = row(
        "detection",
        "<ref>person</ref><box><10><20><30><40></box>",
        categories=["person", "motorcycle"],
    )
    assert selector.response_valid(item)[0] is False


def test_phrase_terminal_punctuation_is_normalized():
    item = row("gui", "<ref>button.</ref><box><10><20><30><40></box>", phrase="button.")
    assert selector.response_valid(item) == (True, "")
    assert selector.canonical_prompt(item).endswith("button.")
    assert not selector.canonical_prompt(item).endswith("button..")


def test_pointing_requires_single_phrase_alignment():
    good = row("pointing", "<ref>button</ref><box><10><20></box>")
    bad = row("pointing", "<ref>icon</ref><box><10><20></box>")
    assert selector.response_valid(good) == (True, "")
    assert selector.response_valid(bad)[0] is False


def test_ocr_rejects_any_loss_or_truncation():
    base = {"metadata": {"target_count": 1, "hiertext_filter": {"parsed_word_boxes": 1}}}
    assert selector.response_valid(
        row("ocr", "<ref>CDC</ref><box><10><20><30><40></box>", **base)
    ) == (True, "")
    for stats in (
        {"dropped_non_positive_extent": 1},
        {"dropped_degenerate_after_normalization": 1},
        {"parsed_word_boxes": 49},
    ):
        item = row(
            "ocr",
            "<ref>CDC</ref><box><10><20><30><40></box>",
            metadata={"target_count": 1, "hiertext_filter": {"parsed_word_boxes": 1, **stats}},
        )
        assert selector.response_valid(item)[0] is False


def test_layout_rejects_invalid_geometry_and_label_truncation():
    response = "<ref>text</ref><box><10><20><30><40></box>"
    categories = ["text"]
    assert selector.response_valid(
        row("layout", response, categories=categories, metadata={"target_count": 1, "layout_filter": {"unique_valid_boxes": 1}})
    ) == (True, "")
    assert selector.canonical_prompt(
        row("layout", response, categories=categories)
    ) == "Detect all the objects in the image that belong to the category set: text."
    for stats in (
        {"invalid_source_boxes": 1},
        {"degenerate_after_normalization": 1},
        {"unique_valid_boxes": 49},
    ):
        item = row(
            "layout", response, categories=categories,
            metadata={"target_count": 1, "layout_filter": {"unique_valid_boxes": 1, **stats}},
        )
        assert selector.response_valid(item)[0] is False


def test_degenerate_box_is_rejected_for_every_domain():
    response = "<ref>button</ref><box><10><20><10><40></box>"
    assert selector.response_valid(row("gui", response))[0] is False


def test_selection_drops_near_duplicate():
    rows = [
        row("gui", "<ref>button</ref><box><10><20><30><40></box>", sample_id=str(i))
        for i in range(2)
    ]
    qualities = {f"gui:{i}:{i}": {"dhash": 0, "aspect_ratio": 1.0} for i in range(2)}
    for i, value in enumerate(rows):
        value["_candidate_id"] = f"gui:{i}:{i}"
    chosen, drops = selector.select_diverse(rows, qualities, 2, "seed")
    assert len(chosen) == 1
    assert drops == 1


def test_selection_is_deterministic():
    rows = [
        row("gui", "<ref>button</ref><box><10><20><30><40></box>", sample_id=str(i))
        for i in range(3)
    ]
    qualities = {
        f"gui:{i}:{i}": {"dhash": sum(1 << bit for bit in range(i * 8, i * 8 + 8)), "aspect_ratio": 1.0}
        for i in range(3)
    }
    for i, value in enumerate(rows):
        value["_candidate_id"] = f"gui:{i}:{i}"
    chosen, _ = selector.select_diverse(rows, qualities, 2, "seed")
    assert len(chosen) == 2
    chosen_again, _ = selector.select_diverse(rows, qualities, 2, "seed")
    assert [item["sample_id"] for item in chosen] == [item["sample_id"] for item in chosen_again]


def test_exclude_manifest_loads_unique_image_sha256(tmp_path):
    digest = "a" * 64
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text(
        '{"image_sha256":"' + digest + '"}\n'
        '{"image_sha256":"' + digest + '"}\n',
        encoding="utf-8",
    )
    assert selector.excluded_image_sha256([manifest]) == {digest}


def test_exclude_manifest_rejects_invalid_sha(tmp_path):
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text('{"image_sha256":"short"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid image_sha256"):
        selector.excluded_image_sha256([manifest])
