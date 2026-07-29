import importlib.util
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "deploy" / "run_locateanything.py"
SPEC = importlib.util.spec_from_file_location("run_locateanything_box_test", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime)


TRANSFORM = {
    "source_size": [640, 640],
    "target_size": [672, 672],
    "resized_size": [672, 672],
    "scale_xy": [1.05, 1.05],
    "padding_ltrb": [0, 0, 0, 0],
}


def test_box_wrapper_requires_and_returns_task_command():
    assert runtime.unwrap_box_command("/box /detect cat") == ("/detect cat", True)
    assert runtime.unwrap_box_command("/detect cat") == ("/detect cat", False)
    try:
        runtime.unwrap_box_command("/box detect cat")
    except ValueError as error:
        assert "/box must wrap a task command" in str(error)
    else:
        raise AssertionError("invalid /box command was accepted")


def test_single_request_defaults_to_hybrid(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_locateanything.py", "image.jpg", "/detect cat"],
    )
    assert runtime.parse_args().generation_mode == "hybrid"


def test_interactive_runtime_defaults_to_hybrid(monkeypatch):
    interactive_script = SCRIPT.with_name("run_locateanything_interactive.py")
    interactive_spec = importlib.util.spec_from_file_location(
        "run_locateanything_interactive_default_test", interactive_script
    )
    interactive = importlib.util.module_from_spec(interactive_spec)
    assert interactive_spec.loader is not None
    monkeypatch.setitem(sys.modules, "run_locateanything", runtime)
    interactive_spec.loader.exec_module(interactive)
    monkeypatch.setattr(sys, "argv", ["LocateAnything"])
    assert interactive.parse_args().generation_mode == "hybrid"


def test_layout_command_uses_official_document_layout_template():
    prompt, task = runtime.normalize_prompt("/layout title,table,figure")
    assert task == "layout_grounding"
    assert prompt == (
        "Detect all the objects in the image that belong to the category set: "
        "title</c>table</c>figure."
    )


def test_box_and_point_coordinates_are_parsed_independently():
    text = (
        "<ref>cat</ref><box><100><200><300><400></box>"
        "<ref>nose</ref><box><500><600></box>"
    )
    boxes = runtime.parse_detections(text, TRANSFORM)
    points = runtime.parse_points(text, TRANSFORM)
    assert boxes == [{
        "label": "cat",
        "bbox_profile_1000": [100, 200, 300, 400],
        "bbox_xyxy": [64.0, 128.0, 192.0, 256.0],
    }]
    assert points == [{
        "label": "nose",
        "point_profile_1000": [500, 600],
        "point_xy": [320.0, 384.0],
    }]


def test_inverted_and_zero_area_boxes_are_rejected():
    text = (
        "<ref>bad</ref><box><900><900><100><100></box>"
        "<box><100><100><100><200></box>"
        "<box><100><100><200><100></box>"
    )
    assert runtime.parse_detections(text, TRANSFORM) == []


def test_class_aware_nms_removes_only_same_label_near_duplicates():
    detections = [
        {"label": "motorcycle", "bbox_xyxy": [0.0, 120.0, 152.0, 226.0]},
        {"label": "Motorcycle", "bbox_xyxy": [0.0, 122.0, 154.0, 227.0]},
        {"label": "motorcycle", "bbox_xyxy": [80.0, 125.0, 535.0, 360.0]},
        {"label": "person", "bbox_xyxy": [0.0, 120.0, 152.0, 226.0]},
    ]

    kept, suppressed = runtime.class_aware_nms(detections, 0.90)

    assert kept == [detections[0], detections[2], detections[3]]
    assert len(suppressed) == 1
    assert suppressed[0]["suppressed_by"] == 1
    assert suppressed[0]["nms_iou"] >= 0.90


def test_detection_postprocess_can_be_disabled_and_skips_other_tasks():
    detections = [
        {"label": "cat", "bbox_xyxy": [10.0, 10.0, 100.0, 100.0]},
        {"label": "cat", "bbox_xyxy": [10.0, 10.0, 100.0, 100.0]},
    ]

    disabled, disabled_suppressed = runtime.postprocess_detections(
        detections, "object_detection", enabled=False
    )
    layout, layout_suppressed = runtime.postprocess_detections(
        detections, "layout_grounding"
    )

    assert disabled == detections
    assert disabled_suppressed == []
    assert layout == detections
    assert layout_suppressed == []


def test_annotated_image_contains_drawn_pixels(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output" / "annotated.png"
    Image.new("RGB", (640, 640), (255, 255, 255)).save(source)
    runtime.save_annotated_image(
        source,
        [{"label": "cat", "bbox_xyxy": [64.0, 128.0, 192.0, 256.0]}],
        [{"label": "nose", "point_xy": [320.0, 384.0]}],
        output,
    )
    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (640, 640)
        assert image.getpixel((64, 128)) != (255, 255, 255)
        assert image.getpixel((320, 384)) != (255, 255, 255)


def test_default_filename_is_unique_png_in_requested_directory(tmp_path):
    first = runtime.annotated_output_path(Path("cat image.jpg"), "object_detection", tmp_path)
    second = runtime.annotated_output_path(Path("cat image.jpg"), "object_detection", tmp_path)
    assert first.parent == tmp_path.resolve()
    assert first.suffix == ".png"
    assert first.name.startswith("cat_image_object_detection_")
    assert first != second


def test_bottom_edge_box_caption_stays_inside_tiny_image(tmp_path):
    source = tmp_path / "tiny.jpg"
    output = tmp_path / "tiny_box.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(source)
    runtime.save_annotated_image(
        source,
        [{"label": "bottom", "bbox_xyxy": [0.0, 62.0, 64.0, 64.0]}],
        [],
        output,
    )
    assert output.is_file()


def test_landscape_letterbox_coordinates_draw_on_original_image(tmp_path):
    source = tmp_path / "landscape.jpg"
    output = tmp_path / "landscape_box.png"
    Image.new("RGB", (640, 360), (255, 255, 255)).save(source)
    _, transform = runtime.prepare_image(source)
    assert transform["padding_ltrb"] == [0, 147, 0, 147]
    boxes = runtime.parse_detections(
        "<ref>target</ref><box><250><359><750><641></box>", transform
    )
    assert len(boxes) == 1
    assert abs(boxes[0]["bbox_xyxy"][0] - 160.0) < 1.0
    assert abs(boxes[0]["bbox_xyxy"][1] - 90.0) < 1.0
    runtime.save_annotated_image(source, boxes, [], output)
    with Image.open(output) as image:
        assert image.getpixel((160, 90)) != (255, 255, 255)


def test_portrait_letterbox_coordinates_draw_on_original_image(tmp_path):
    source = tmp_path / "portrait.jpg"
    output = tmp_path / "portrait_box.png"
    Image.new("RGB", (360, 640), (255, 255, 255)).save(source)
    _, transform = runtime.prepare_image(source)
    assert transform["padding_ltrb"] == [147, 0, 147, 0]
    boxes = runtime.parse_detections(
        "<ref>target</ref><box><359><250><641><750></box>", transform
    )
    assert len(boxes) == 1
    assert abs(boxes[0]["bbox_xyxy"][0] - 90.0) < 1.0
    assert abs(boxes[0]["bbox_xyxy"][1] - 160.0) < 1.0
    runtime.save_annotated_image(source, boxes, [], output)
    with Image.open(output) as image:
        assert image.getpixel((90, 160)) != (255, 255, 255)


def test_landscape_padding_only_box_is_rejected(tmp_path):
    source = tmp_path / "landscape.jpg"
    Image.new("RGB", (640, 360), (255, 255, 255)).save(source)
    _, transform = runtime.prepare_image(source)
    boxes = runtime.parse_detections(
        "<ref>padding</ref><box><100><10><200><100></box>", transform
    )
    assert boxes == []


def test_portrait_padding_only_box_is_rejected(tmp_path):
    source = tmp_path / "portrait.jpg"
    Image.new("RGB", (360, 640), (255, 255, 255)).save(source)
    _, transform = runtime.prepare_image(source)
    boxes = runtime.parse_detections(
        "<ref>padding</ref><box><10><100><100><200></box>", transform
    )
    assert boxes == []
