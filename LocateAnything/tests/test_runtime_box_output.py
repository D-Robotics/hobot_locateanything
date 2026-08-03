import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "deploy" / "run_locateanything.py"
SPEC = importlib.util.spec_from_file_location("run_locateanything_box_test", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


TRANSFORM = {
    "source_size": [640, 640],
    "target_size": [672, 672],
    "resized_size": [672, 672],
    "scale_xy": [1.05, 1.05],
    "padding_ltrb": [0, 0, 0, 0],
}


def test_prediction_outputs_use_one_self_contained_run_directory(tmp_path):
    paths = runtime.create_prediction_paths(
        tmp_path,
        Path("fixtures/orange tray.jpg"),
        now=datetime(2026, 8, 3, 12, 34, 56),
        unique_id="abc123",
    )

    expected_root = (
        tmp_path
        / "artifacts"
        / "runs"
        / "predict"
        / "20260803_123456_orange_tray_abc123"
    ).resolve()
    assert paths.root == expected_root
    assert paths.prediction == expected_root / "prediction.json"
    assert paths.annotated_image == expected_root / "annotated.png"
    assert paths.timings == expected_root / "timings.json"
    assert paths.runtime_log == expected_root / "logs" / "runtime.log"
    assert paths.runtime_log.parent.is_dir()


def test_explicit_prediction_directory_preserves_the_same_output_layout(tmp_path):
    output_dir = tmp_path / "demo"
    paths = runtime.create_prediction_paths(
        tmp_path,
        Path("image.jpg"),
        output_dir=output_dir,
    )

    assert paths.root == output_dir.resolve()
    assert {path.relative_to(paths.root).as_posix() for path in (
        paths.prediction,
        paths.annotated_image,
        paths.timings,
        paths.runtime_log,
    )} == {
        "prediction.json",
        "annotated.png",
        "timings.json",
        "logs/runtime.log",
    }


def write_runtime_config(tmp_path, **updates):
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    config.update(updates)
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def load_interactive_module(monkeypatch):
    interactive_script = SCRIPT.with_name("run_locateanything_interactive.py")
    interactive_spec = importlib.util.spec_from_file_location(
        "run_locateanything_interactive_default_test", interactive_script
    )
    interactive = importlib.util.module_from_spec(interactive_spec)
    assert interactive_spec.loader is not None
    monkeypatch.setitem(sys.modules, "run_locateanything", runtime)
    monkeypatch.setitem(sys.modules, interactive_spec.name, interactive)
    interactive_spec.loader.exec_module(interactive)
    return interactive


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


def test_runtime_config_is_the_source_for_cli_defaults(tmp_path):
    config = write_runtime_config(
        tmp_path,
        default_max_new_tokens=77,
        default_nms_iou=0.75,
    )
    args = runtime.parse_args(["-c", str(config), "image.jpg", "/detect cat"])
    assert args.runtime.source == config.resolve()
    assert args.generation_mode == "hybrid"
    assert args.max_new_tokens == 77
    assert args.nms_iou == 0.75
    assert args.runtime.language_graph_set == "standard"
    assert args.runtime.specification()["language_graph_set"] == "standard"


@pytest.mark.parametrize("profile", ["standard", "fused_decode"])
def test_runtime_config_accepts_supported_execution_profiles(tmp_path, profile):
    config = write_runtime_config(tmp_path, language_graph_set=profile)
    loaded = runtime.load_runtime_config(config)
    assert loaded.language_graph_set == profile
    assert runtime.language_runner_command(loaded)[-2:] == ["--graph-set", profile]


def test_runtime_config_rejects_unknown_graph_set(tmp_path):
    config = write_runtime_config(tmp_path, language_graph_set="partial")
    with pytest.raises(ValueError, match="language_graph_set"):
        runtime.load_runtime_config(config)


def test_runtime_config_requires_graph_set(tmp_path):
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    del config["language_graph_set"]
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="language_graph_set"):
        runtime.load_runtime_config(path)


def test_runtime_config_environment_and_artifact_overrides(monkeypatch, tmp_path):
    config = write_runtime_config(tmp_path)
    artifacts = tmp_path / "release-artifacts"
    monkeypatch.setenv("LA_RUNTIME_CONFIG", str(config))
    monkeypatch.setenv("LA_RELEASE_ROOT", str(artifacts))
    loaded = runtime.load_runtime_config()
    assert loaded.source == config.resolve()
    assert loaded.vision_model == (artifacts / "LocateAnything-3B_vision.hbm").resolve()
    assert loaded.language_model == (artifacts / "LocateAnything-3B_language.hbm").resolve()


def test_published_runtime_config_is_discovered_next_to_deploy(tmp_path):
    release = tmp_path / "release"
    runtime_dir = release / "deploy"
    config_dir = release / "config"
    runtime_dir.mkdir(parents=True)
    config_dir.mkdir()
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    path = config_dir / "locateanything_3b_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = runtime.load_runtime_config(path, runtime_dir=ROOT / "deploy")
    assert loaded.source == path.resolve()
    assert loaded.layout_root == release.resolve()
    assert loaded.vision_runner == (runtime_dir / "build/vision_hbm_runner").resolve()
    assert loaded.language_runner == (runtime_dir / "build/language_hbm_runner").resolve()
    assert loaded.tokenizer_dir == (runtime_dir / "tokenizer").resolve()


def test_relative_runtime_environment_paths_use_release_root(monkeypatch, tmp_path):
    release = tmp_path / "release"
    runtime_dir = release / "deploy"
    config_dir = release / "config"
    runtime_dir.mkdir(parents=True)
    config_dir.mkdir()
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    path = config_dir / "locateanything_3b_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LA_VISION_MODEL", "overrides/vision.hbm")
    monkeypatch.setenv("LA_LANGUAGE_RUNNER", "overrides/language_hbm_runner")
    loaded = runtime.load_runtime_config(path, runtime_dir=ROOT / "deploy")
    assert loaded.vision_model == (release / "overrides/vision.hbm").resolve()
    assert loaded.language_runner == (release / "overrides/language_hbm_runner").resolve()


def test_published_release_layout_contains_complete_runtime_payload(tmp_path):
    release = tmp_path / "release"
    artifacts = release / "artifacts"
    tokenizer = release / "tokenizer"
    build = release / "deploy/build"
    config_dir = release / "config"
    for directory in (artifacts, tokenizer, build, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    config["model_dir"] = str(artifacts)
    config["vocabulary_path"] = str(tokenizer)
    config_path = config_dir / "locateanything_3b_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for path in (
        artifacts / "LocateAnything-3B_vision.hbm",
        artifacts / "LocateAnything-3B_language.hbm",
        artifacts / "LocateAnything-3B_embed_tokens.bin",
        tokenizer / "tokenizer.json",
        build / "vision_hbm_runner",
        build / "language_hbm_runner",
    ):
        path.write_bytes(b"fixture")
    loaded = runtime.load_runtime_config(config_path, runtime_dir=ROOT / "deploy")
    runtime.require_runtime_paths(loaded)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_width", 448),
        ("cache_len", 2048),
        ("pbd_query_len", 5),
        ("ar_query_len", 2),
        ("default_generation_mode", "slow"),
        ("decode_bpu_core", [0]),
    ],
)
def test_runtime_config_rejects_non_release_graph_contract(tmp_path, field, value):
    config = write_runtime_config(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        runtime.load_runtime_config(config)


def test_runtime_config_rejects_missing_required_field(tmp_path):
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    del config["cache_len"]
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="cache_len"):
        runtime.load_runtime_config(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_config_rejects_non_finite_startup_timeout(tmp_path, value):
    config = write_runtime_config(
        tmp_path, runner_startup_timeout_seconds=value
    )
    with pytest.raises(ValueError, match="non-finite|finite and positive"):
        runtime.load_runtime_config(config)


def test_missing_environment_runtime_config_fails_closed(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("LA_RUNTIME_CONFIG", str(missing))
    with pytest.raises(FileNotFoundError, match="missing.json"):
        runtime.load_runtime_config()


def test_runtime_payload_validation_requires_tokenizer_json(monkeypatch, tmp_path):
    config = write_runtime_config(tmp_path)
    files = {
        "LA_VISION_RUNNER": tmp_path / "vision_hbm_runner",
        "LA_LANGUAGE_RUNNER": tmp_path / "language_hbm_runner",
        "LA_VISION_MODEL": tmp_path / "vision.hbm",
        "LA_LANGUAGE_MODEL": tmp_path / "language.hbm",
        "LA_EMBEDDINGS": tmp_path / "embed.bin",
    }
    for name, path in files.items():
        path.write_bytes(b"fixture")
        monkeypatch.setenv(name, str(path))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    monkeypatch.setenv("LA_TOKENIZER_DIR", str(tokenizer))
    loaded = runtime.load_runtime_config(config)
    with pytest.raises(FileNotFoundError, match="Tokenizer JSON"):
        runtime.require_runtime_paths(loaded)
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    runtime.require_runtime_paths(loaded)


def test_runtime_config_overrides_stale_shell_l2m(monkeypatch):
    monkeypatch.setenv("HB_DNN_USER_DEFINED_L2M_SIZES", "1:1:1:1")
    loaded = runtime.load_runtime_config()
    env = runtime.build_runtime_environment(loaded)
    assert env["HB_DNN_USER_DEFINED_L2M_SIZES"] == "6:6:6:6"


def test_s600_resource_status_fixture_parses_all_cores_and_temperature():
    fixture = (ROOT / "tests/fixtures/hrut_somstatus_s600.txt").read_text(encoding="utf-8")
    bpu, temperature = runtime.parse_s600_resource_status(fixture)
    assert bpu == [17.0, 22.0, 0.0, 5.0]
    assert temperature == 44.5
    _, legacy_temperature = runtime.parse_s600_resource_status(
        "pvt_bpu_sensor: 46.25 (C)\n"
    )
    assert legacy_temperature == 46.25


def test_interactive_runtime_defaults_to_hybrid(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["LocateAnything"])
    assert interactive.parse_args().generation_mode == "hybrid"
    dashboard = interactive.ResourceDashboard(enabled=False, interval_seconds=1.0)
    assert dashboard.interval_seconds == 1.0
    assert not dashboard._active.is_set()
    assert "IDLE" in dashboard._render()
    with dashboard.request_scope():
        assert dashboard._active.is_set()
    assert not dashboard._active.is_set()
    assert "LAST REQUEST" in dashboard._render()
    samples = []
    dashboard._sample = lambda: samples.append("sample")
    dashboard._draw = dashboard._stop.set
    dashboard._run()
    assert samples == []
    active_dashboard = interactive.ResourceDashboard(enabled=False, interval_seconds=1.0)
    active_samples = []
    active_dashboard._active.set()
    active_dashboard._sample = lambda: active_samples.append("sample")
    active_dashboard._draw = active_dashboard._stop.set
    active_dashboard._run()
    assert active_samples == ["sample"]


def test_interactive_result_summary_uses_clear_performance_terms(monkeypatch, capsys):
    interactive = load_interactive_module(monkeypatch)

    interactive.print_result(
        [{
            "label": "cat",
            "bbox_profile_1000": [100, 200, 300, 400],
            "bbox_xyxy": [64.0, 128.0, 192.0, 256.0],
        }],
        [],
        vit_ms=25.0,
        vit_infer_ms=20.0,
        vision_cached=False,
        prefill_tokens=100,
        prefill_ms=50.0,
        decode_tokens=10,
        decode_ms=200.0,
        total_ms=300.0,
    )

    output = capsys.readouterr().out
    assert "Performance" in output
    assert "Vision" in output
    assert "Prefill" in output and "2000.000 tokens/s" in output
    assert "Decode" in output and "20.000 ms/token" in output
    assert "End-to-end" in output
    assert "Predictions" in output and "Boxes   1" in output
    assert "normalized=[100, 200, 300, 400]" in output


def test_hbm_server_startup_timeout_terminates_child(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    children = []
    real_popen = interactive.subprocess.Popen

    def capture_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(interactive.subprocess, "Popen", capture_process)
    command = [sys.executable, "-u", "-c", "import time; time.sleep(30)"]
    with pytest.raises(TimeoutError, match="ready"):
        interactive.HbmServer(command, os.environ.copy(), "fixture", 0.1)
    assert len(children) == 1
    assert children[0].poll() is not None


def test_hbm_server_ready_and_clean_quit(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys\n"
            "print('LAHBM/1\\tREADY\\tfixture', flush=True)\n"
            "for line in sys.stdin:\n"
            "    if line.strip() == 'LAHBM/1\\tQUIT':\n"
            "        break\n"
        ),
    ]
    server = interactive.HbmServer(command, os.environ.copy(), "fixture", 2.0)
    server.close()
    assert server.process.poll() is not None


def test_stream_callback_failure_drains_request_before_next_request(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys\n"
            "print('LAHBM/1\\tREADY\\tfixture', flush=True)\n"
            "for line in sys.stdin:\n"
            "    fields = line.strip().split('\\t')\n"
            "    if line.strip() == 'LAHBM/1\\tQUIT':\n"
            "        break\n"
            "    request_id = fields[2]\n"
            "    print(f'LAHBM/1\\tTOKEN\\t{request_id}\\t101', flush=True)\n"
            "    print(f'LAHBM/1\\tTOKEN\\t{request_id}\\t102', flush=True)\n"
            "    print(f'LAHBM/1\\tRESULT\\t{request_id}\\tok', flush=True)\n"
        ),
    ]
    server = interactive.HbmServer(command, os.environ.copy(), "fixture", 2.0)
    callbacks = []

    def broken_callback(token):
        callbacks.append(token)
        raise OSError("terminal disconnected")

    with pytest.raises(OSError, match="terminal disconnected"):
        server.request(["LAHBM/1", "RUN", "first"], on_token=broken_callback)
    result, _ = server.request(["LAHBM/1", "RUN", "second"])
    server.close()

    assert callbacks == [101]
    assert result.startswith("LAHBM/1\tRESULT\tsecond\t")


def test_terminal_request_id_mismatch_terminates_server(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys, time\n"
            "print('LAHBM/1\\tREADY\\tfixture', flush=True)\n"
            "sys.stdin.readline()\n"
            "print('LAHBM/1\\tRESULT\\twrong\\tok', flush=True)\n"
            "time.sleep(30)\n"
        ),
    ]
    server = interactive.HbmServer(command, os.environ.copy(), "fixture", 2.0)
    with pytest.raises(RuntimeError, match="invalid terminal frame"):
        server.request(["LAHBM/1", "RUN", "expected"])
    assert server.process.poll() is not None


def test_dashboard_construction_failure_closes_both_hbm_servers(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    closed: list[str] = []
    commands: dict[str, list[str]] = {}

    class FakeServer:
        def __init__(self, command, env, name, startup_timeout_seconds):
            self.name = name
            commands[name] = command

        def close(self):
            closed.append(self.name)

    runtime_config = SimpleNamespace(
        vision_runner=Path("vision_hbm_runner"),
        language_runner=Path("language_hbm_runner"),
        tokenizer_dir=Path("tokenizer"),
        vision_model=Path("vision.hbm"),
        language_model=Path("language.hbm"),
        embeddings=Path("embed.bin"),
        language_graph_set="fused_decode",
        runner_startup_timeout_seconds=120.0,
        telemetry_interval_seconds=1.0,
        l2m_sizes="6:6:6:6",
    )
    args = SimpleNamespace(
        max_new_tokens=32,
        nms_iou=0.9,
        prompt=None,
        image=None,
        output_dir=None,
        no_dashboard=False,
        generation_mode="hybrid",
        runtime=runtime_config,
    )
    monkeypatch.setattr(interactive, "parse_args", lambda: args)
    monkeypatch.setattr(interactive, "require_runtime_paths", lambda _: None)
    monkeypatch.setattr(interactive, "load_tokenizer", lambda _: object())
    monkeypatch.setattr(interactive, "HbmServer", FakeServer)
    monkeypatch.setattr(
        interactive,
        "ResourceDashboard",
        lambda **_: (_ for _ in ()).throw(RuntimeError("dashboard failed")),
    )

    with pytest.raises(RuntimeError, match="dashboard failed"):
        interactive.main()

    assert closed == ["language", "vision"]
    assert commands["language"][-3:] == ["--graph-set", "fused_decode", "--server"]


def test_cleanup_attempts_both_runners_when_dashboard_stop_fails(monkeypatch, capsys):
    interactive = load_interactive_module(monkeypatch)
    closed: list[str] = []

    class BrokenDashboard:
        def stop(self):
            raise OSError("terminal disconnected")

    class FakeServer:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    interactive.close_runtime_resources(
        BrokenDashboard(), FakeServer("language"), FakeServer("vision")
    )

    assert closed == ["language", "vision"]
    assert "failed to close dashboard" in capsys.readouterr().err


def test_cleanup_finishes_when_warning_stream_is_broken(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    closed: list[str] = []

    class BrokenResource:
        def stop(self):
            raise OSError("terminal disconnected")

    class FakeServer:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    class BrokenStderr:
        def write(self, _value):
            raise OSError("stderr disconnected")

        def flush(self):
            raise OSError("stderr disconnected")

    monkeypatch.setattr(interactive.sys, "stderr", BrokenStderr())
    interactive.close_runtime_resources(
        BrokenResource(), FakeServer("language"), FakeServer("vision")
    )

    assert closed == ["language", "vision"]


def test_dashboard_start_keeps_prompt_below_startup_output(monkeypatch, capsys):
    interactive = load_interactive_module(monkeypatch)
    dashboard = interactive.ResourceDashboard(enabled=False, interval_seconds=1.0)
    dashboard.enabled = True
    dashboard._rows = 32
    dashboard._run = lambda: None

    dashboard.start()
    dashboard.stop()

    terminal_output = capsys.readouterr().out
    assert "\033[1;31r\033[31;1H" in terminal_output


def test_cleanup_masks_repeated_sigint_and_restores_handler(monkeypatch):
    interactive = load_interactive_module(monkeypatch)
    events = []
    handlers = []
    original_handler = object()

    monkeypatch.setattr(interactive.signal, "getsignal", lambda _: original_handler)
    monkeypatch.setattr(
        interactive.signal,
        "signal",
        lambda signal_number, handler: handlers.append((signal_number, handler)),
    )

    class Resource:
        def __init__(self, name, method):
            self.name = name
            setattr(self, method, self._close)

        def _close(self):
            events.append(self.name)

    interactive.close_runtime_resources(
        Resource("dashboard", "stop"),
        Resource("language", "close"),
        Resource("vision", "close"),
    )

    assert events == ["dashboard", "language", "vision"]
    assert handlers == [
        (interactive.signal.SIGINT, interactive.signal.SIG_IGN),
        (interactive.signal.SIGINT, original_handler),
    ]


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
