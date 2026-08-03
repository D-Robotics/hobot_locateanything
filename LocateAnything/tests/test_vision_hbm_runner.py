import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_board_runner_uses_real_672_input_contract():
    source = (ROOT / "deploy/example/vision_hbm_runner.cpp").read_text(encoding="utf-8")
    assert "{1, 2304, 588}" in source
    assert "{1, 576, 2048}" in source
    assert "BuildDummyVisionPatchTensor" not in source
    assert "--model" in source and "--input" in source and "--output" in source


def test_board_runner_supports_one_load_multi_inference_protocol():
    source = (ROOT / "deploy/example/vision_hbm_runner.cpp").read_text(encoding="utf-8")
    assert '"--server"' in source
    assert '"LAHBM/1\\tREADY\\tvisual\\n"' in source
    assert '"LAHBM/1\\tRESULT\\t%s\\t%.3f\\t%zu\\n"' in source
    assert 'request == "LAHBM/1\\tQUIT"' in source
    assert "RunOne(&session" in source
    assert 'path + ".tmp"' in source


def test_board_shell_runner_has_only_model_input_output_parameters():
    script = (ROOT / "deploy/run_vision_hbm.sh").read_text(encoding="utf-8")
    assert "--model" in script
    assert "--input" in script
    assert "--output" in script
    assert "HB_DNN_USER_DEFINED_L2M_SIZES" in script
    assert "448x448" not in script


def test_cmake_builds_board_runner():
    cmake = (ROOT / "deploy/CMakeLists.txt").read_text(encoding="utf-8")
    assert "add_executable(vision_hbm_runner" in cmake
    assert "target_link_libraries(vision_hbm_runner PRIVATE locateanything_runtime)" in cmake
    assert "add_executable(language_graph_set_test example/language_graph_set_test.cpp)" in cmake
    assert "vision_dummy_test" not in cmake
    assert "prefill_verify_v3" not in cmake
    assert "add_executable(prefill_verify" not in cmake
    assert "add_executable(layout_probe" not in cmake


def test_release_runtime_code_does_not_encode_historical_build_directories():
    runtime = (ROOT / "deploy/run_locateanything.py").read_text(encoding="utf-8")
    assert "_820" not in runtime
    assert "nash-p_w4" not in runtime


def test_runtime_config_matches_release_contract():
    config = json.loads((ROOT / "deploy/runtime_config.json").read_text(encoding="utf-8"))
    assert (config["image_width"], config["image_height"]) == (672, 672)
    assert config["vocab_size"] == 152681
    assert config["embed_dim"] == 2048
    assert config["patch_size"] == 14
    assert config["visual_tokens"] == 576
    assert config["prefill_chunk"] == 1024
    assert config["cache_len"] == 4096
    assert config["pbd_query_len"] == 6
    assert config["ar_query_len"] == 1
    assert config["default_generation_mode"] == "hybrid"
    assert config["default_max_new_tokens"] == 2048
    assert config["default_nms_iou"] == 0.9
    assert config["l2m_sizes"] == "6:6:6:6"
    assert config["telemetry_interval_ms"] == 1000
    assert config["runner_startup_timeout_seconds"] == 120
    assert config["language_graph_set"] == "standard"
    assert config["model_dir"] == "artifacts/releases/current/"
    assert config["vit_model_file"] == "LocateAnything-3B_vision.hbm"
    assert config["llm_model_file"] == "LocateAnything-3B_language.hbm"


def test_runtime_fails_closed_when_output_cache_invalidation_fails():
    source = (ROOT / "deploy/src/hbm_session.cpp").read_text(encoding="utf-8")
    assert "err = hbUCPMemFlush" in source
    assert 'Result::Err(err, "hbUCPMemFlush INVALIDATE output idx="' in source
