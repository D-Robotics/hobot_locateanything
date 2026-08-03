import io
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
import validate_fused_pbd_s600_ab as ab  # noqa: E402


@pytest.mark.parametrize("profile", ["standard", "fused_decode"])
def test_resident_server_passes_graph_set(monkeypatch, tmp_path, profile):
    commands = []

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("LAHBM/1\tREADY\tlanguage\n")

    def fake_popen(command, **kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    server = ab.ResidentLanguageServer(
        Path("runner"),
        Path("model.hbm"),
        Path("embed.bin"),
        profile,
        tmp_path / "server.log",
        1.0,
        1.0,
    )
    server.start()

    assert commands[0][-3:] == ["--graph-set", profile, "--server"]


def generation(
    stop_reason="im_end",
    token_ids=(1, 2, 3),
    structured="<box><1><2><3><4></box>",
):
    return ab.GenerationOutput(stop_reason, token_ids, structured)


def make_run(tag: str, fused: bool) -> ab.ModelRun:
    outputs = (generation(),) * ab.RUN_COUNT
    metrics = tuple(
        ab.RequestMetrics(
            request_id=f"{tag}-{index:02d}",
            stop_reason="im_end",
            response_size=3,
            prefill_tokens=615,
            prefill_ms=(100.0, 12.0, 14.0, 16.0, 18.0)[index - 1]
            if tag == "old"
            else (90.0, 10.0, 12.0, 14.0, 16.0)[index - 1],
            decode_ms=(200.0, 80.0, 100.0, 120.0, 140.0)[index - 1]
            if tag == "old"
            else (180.0, 40.0, 50.0, 60.0, 70.0)[index - 1],
        )
        for index in range(1, ab.RUN_COUNT + 1)
    )
    graphs = ab.EXPECTED_FUSED_GRAPHS if fused else ab.BASE_GRAPHS
    profile_log = ab.FUSED_DECODE_LOG if fused else ab.STANDARD_LOG
    return ab.ModelRun(
        tag=tag,
        model_path=Path(f"/{tag}.hbm"),
        graph_names=graphs,
        generations=outputs,
        metrics=metrics,
        log_path=Path(f"/{tag}.log"),
        log_lines=(f"[ok] loaded graphs: {' '.join(graphs)}",)
        + (profile_log,) * ab.RUN_COUNT,
    )


def test_parses_generation_graphs_and_result_frame():
    output = ab.parse_generation_text(
        "stop_reason=im_end\ntoken_ids=10,20,30\nstructured=<ref>x</ref><box><1><2><3><4></box>\n"
    )
    assert output == ab.GenerationOutput(
        "im_end", (10, 20, 30), "<ref>x</ref><box><1><2><3><4></box>"
    )
    graphs = ab.parse_graph_names(
        [f"[ok] loaded graphs: {' '.join(ab.EXPECTED_FUSED_GRAPHS)}"]
    )
    assert graphs == ab.EXPECTED_FUSED_GRAPHS
    result = ab.parse_result_line(
        "LAHBM/1\tRESULT\tnew-01\tim_end\t3\t615\t150.250\t80.125",
        "new-01",
    )
    assert result.response_size == 3
    assert result.prefill_ms == pytest.approx(150.25)
    assert result.decode_ms == pytest.approx(80.125)


def test_strips_ansi_prefix_before_profile_matching():
    raw = "\x1b[31m\x1b[1m" + ab.STANDARD_LOG + "\x1b[0m"
    assert ab.ANSI_ESCAPE_RE.sub("", raw) == ab.STANDARD_LOG


def test_acceptance_discards_warmup_and_reports_speedup():
    report = ab.validate_ab(make_run("old", False), make_run("new", True))
    assert report["passed"] is True
    assert report["output_comparison"]["exact_token_rounds"] == ab.RUN_COUNT
    assert report["old"]["prefill_median_ms"] == pytest.approx(15.0)
    assert report["new"]["prefill_median_ms"] == pytest.approx(13.0)
    assert report["old"]["decode_median_ms"] == pytest.approx(110.0)
    assert report["new"]["decode_median_ms"] == pytest.approx(55.0)
    assert report["speedup_old_over_new"]["decode"] == pytest.approx(2.0)


def test_rejects_incomplete_new_graph_family():
    new = make_run("new", True)
    new = replace(new, graph_names=new.graph_names[:-1])
    with pytest.raises(ab.ValidationError, match="graph set mismatch"):
        ab.validate_ab(make_run("old", False), new)


def test_rejects_new_standard_fallback():
    new = make_run("new", True)
    new = replace(new, log_lines=new.log_lines + (ab.STANDARD_LOG,))
    with pytest.raises(ab.ValidationError, match="standard graph set"):
        ab.validate_ab(make_run("old", False), new)


def test_accepts_token_difference_when_box_iou_passes():
    old = make_run("old", False)
    new = make_run("new", True)
    old_output = generation(
        token_ids=(1, 2, 3),
        structured="<ref>cat</ref><box><495><212><736><516></box><|im_end|>",
    )
    new_output = generation(
        token_ids=(1, 2, 4),
        structured="<ref>cat</ref><box><495><212><736><525></box><|im_end|>",
    )
    old = replace(old, generations=(old_output,) * ab.RUN_COUNT)
    new = replace(new, generations=(new_output,) * ab.RUN_COUNT)
    report = ab.validate_ab(old, new)
    assert report["output_comparison"]["exact_token_rounds"] == 0
    assert report["output_comparison"]["rounds"][0]["box_ious"][0] == pytest.approx(
        304.0 / 313.0
    )


def test_matches_multiple_boxes_independently_of_output_order():
    old = generation(
        structured=(
            "<ref>item</ref><box><0><0><100><100></box>"
            "<box><200><200><300><300></box><|im_end|>"
        )
    )
    new = generation(
        structured=(
            "<ref>item</ref><box><200><200><300><300></box>"
            "<box><0><0><100><100></box><|im_end|>"
        )
    )
    comparison = ab.compare_localization_outputs(old, new, 0.90, 10.0)
    assert comparison["box_ious"] == [1.0, 1.0]


def test_compares_two_coordinate_box_as_point():
    old = generation(structured="<ref>button</ref><box><100><200></box><|im_end|>")
    new = generation(structured="<ref>button</ref><box><106><208></box><|im_end|>")
    comparison = ab.compare_localization_outputs(old, new, 0.90, 10.0)
    assert comparison["point_distances"] == [10.0]


@pytest.mark.parametrize(
    "changed, message",
    [
        (
            generation(structured="<ref>dog</ref><box><1><2><3><4></box>"),
            "labels or localization output structure",
        ),
        (
            generation(structured="<box><1><2><30><40></box>"),
            "box IoU below",
        ),
        (generation(stop_reason="max_new_tokens"), "stop_reason"),
    ],
)
def test_rejects_semantic_output_mismatch_or_incomplete_generation(changed, message):
    new = make_run("new", True)
    new = replace(new, generations=(changed,) * ab.RUN_COUNT)
    with pytest.raises(ab.ValidationError, match=message):
        ab.validate_ab(make_run("old", False), new)


def test_rejects_result_output_token_count_mismatch():
    new = make_run("new", True)
    bad_metric = replace(new.metrics[0], response_size=2)
    new = replace(new, metrics=(bad_metric,) + new.metrics[1:])
    with pytest.raises(ab.ValidationError, match="token count mismatch"):
        ab.validate_ab(make_run("old", False), new)
