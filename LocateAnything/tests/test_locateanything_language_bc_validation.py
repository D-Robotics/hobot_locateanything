import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "compiler" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "compiler"))

LANGUAGE_SCRIPT = SCRIPT_DIR / "common/language.py"
LANGUAGE_SPEC = importlib.util.spec_from_file_location("la_language_bc_validation", LANGUAGE_SCRIPT)
LANGUAGE = importlib.util.module_from_spec(LANGUAGE_SPEC)
assert LANGUAGE_SPEC.loader is not None
sys.modules[LANGUAGE_SPEC.name] = LANGUAGE
LANGUAGE_SPEC.loader.exec_module(LANGUAGE)

REPLAY_SCRIPT = ROOT / "compiler" / "leap_llm" / "apis" / "calibration" / "locateanything_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location(
    "leap_llm.apis.calibration.locateanything_replay", REPLAY_SCRIPT
)
REPLAY = importlib.util.module_from_spec(REPLAY_SPEC)
assert REPLAY_SPEC.loader is not None
sys.modules[REPLAY_SPEC.name] = REPLAY
REPLAY_SPEC.loader.exec_module(REPLAY)

PIPELINE_SCRIPT = SCRIPT_DIR / "validate/compare_pipeline.py"
PIPELINE_SPEC = importlib.util.spec_from_file_location("la_pipeline_language_bc", PIPELINE_SCRIPT)
PIPELINE = importlib.util.module_from_spec(PIPELINE_SPEC)
assert PIPELINE_SPEC.loader is not None
sys.modules[PIPELINE_SPEC.name] = PIPELINE
PIPELINE_SPEC.loader.exec_module(PIPELINE)


def _write_bundle(base: Path, converted: bool = False) -> dict[str, Path]:
    marker = "_convert" if converted else ""
    paths = {
        graph: Path(f"{base}.{graph}{marker}.bc")
        for graph in LANGUAGE.LANGUAGE_BC_GRAPHS
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))
    return paths


def test_language_eager_candidate_uses_w8_head_and_dynamic_attention_cache_a8():
    assert LANGUAGE.DECODER_WEIGHT_BITS == 8
    assert LANGUAGE.LM_HEAD_WEIGHT_BITS == 8
    assert LANGUAGE.EMULATION_VERSION == 4
    assert LANGUAGE.DYNAMIC_A8_PATTERNS == (
        r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
        r"layers\.\d+\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
        r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq",
    )


def test_language_attention_eager_rope_preserves_batch_and_kv_heads():
    source = (
        ROOT
        / "compiler/leap_llm/models/locateanything/blocks/text_attention_leap.py"
    ).read_text(encoding="utf-8")
    forward = source.split("    def forward(\n", 1)[1].split(
        "\n\nclass Qwen2_5_VLVisionAttention", 1
    )[0]

    assert "query_states = query_states.reshape(-1, q_len" not in forward
    assert "key_states = key_states.reshape(-1, q_len" not in forward
    assert "apply_rope_torch_1d(\n            query_states, key_states, cos, sin" in forward


def _descriptor(name: str, shape: tuple[int, ...], dtype: np.dtype):
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(shape=shape, np_dtype=np.dtype(dtype)),
    )


class FakeFunction:
    def __init__(self, graph: str, output_name: str, q_len: int, vocab: int) -> None:
        self.graph = graph
        self.output_name = output_name
        self.q_len = q_len
        self.vocab = vocab
        self.feeds: list[dict[str, np.ndarray]] = []

    def feed(self, *, inputs):
        self.feeds.append(inputs)
        marker = {"prefill": 1.0, "decode": 2.0, "decode_ar": 3.0}[self.graph]
        logits = np.full((1, self.q_len, self.vocab), marker, dtype=np.float16)
        return {self.output_name: logits}


def _artifact(graph: str, q_len: int, *, layers: int, cache_len: int, vocab: int):
    inputs = [
        _descriptor(f"{graph}_arg0", (1, q_len, 4), np.float16),
        _descriptor(f"{graph}_arg1", (1, 1, q_len), np.int32),
        _descriptor(f"{graph}_arg2", (1, q_len, cache_len), np.float16),
    ]
    inputs.extend(
        _descriptor(f"{graph}_arg{index + 3}", (1, cache_len, 1, 2), np.float32)
        for index in range(2 * layers)
    )
    output = _descriptor(f"{graph}_result0", (1, q_len, vocab), np.float16)
    function = FakeFunction(graph, output.name, q_len, vocab)
    artifact = LANGUAGE.LanguageBCArtifact(
        graph=graph,
        path=Path(f"{graph}.bc"),
        function=function,
        inputs=inputs,
        outputs=[output],
    )
    return artifact, function


class FakeLanguageModel(torch.nn.Module):
    def __init__(self, layers: int, vocab: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            num_key_value_heads=1,
            num_attention_heads=2,
            hidden_size=4,
            text_mask_token_id=31,
        )
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.vocab = vocab
        self.calls = 0

    def forward(self, inputs_embeds, _positions, _mask, *_caches):
        self.calls += 1
        batch, q_len, _ = inputs_embeds.shape
        logits = torch.arange(
            batch * q_len * self.vocab,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ).reshape(batch, q_len, self.vocab)
        keys = [
            torch.full((batch, q_len, 1, 2), 10.0 + layer, dtype=inputs_embeds.dtype)
            for layer in range(self.config.num_hidden_layers)
        ]
        values = [
            torch.full((batch, q_len, 1, 2), 20.0 + layer, dtype=inputs_embeds.dtype)
            for layer in range(self.config.num_hidden_layers)
        ]
        return logits, keys, values


def test_resolve_language_bc_bundle_from_directory_or_any_member(tmp_path):
    exported = _write_bundle(tmp_path / "language")
    converted = _write_bundle(tmp_path / "language", converted=True)

    assert LANGUAGE.resolve_language_bc_paths(exported["decode"], converted=False) == {
        "decode": exported["decode"]
    }
    assert LANGUAGE.resolve_language_bc_paths(converted["decode_ar"], converted=True) == {
        "decode_ar": converted["decode_ar"]
    }
    assert LANGUAGE.resolve_language_bc_paths(tmp_path, converted=False) == exported

    assert PIPELINE._language_bc_prefill_provider_path(
        {"decode": converted["decode"]}, converted=True
    ) == converted["prefill"]
    assert PIPELINE._language_bc_prefill_provider_path(
        {"decode": exported["decode"]}, converted=False
    ) is None


def test_language_bc_runner_maps_flattened_inputs_and_uses_float_prefill_cache(
    monkeypatch,
):
    monkeypatch.setattr(LANGUAGE, "CHUNK_SIZE", 4)
    monkeypatch.setattr(LANGUAGE, "CACHE_LEN", 8)
    monkeypatch.setattr(LANGUAGE, "PBD_QUERY_LEN", 2)
    monkeypatch.setattr(LANGUAGE, "AR_QUERY_LEN", 1)
    monkeypatch.setattr(LANGUAGE, "VOCAB_SIZE", 5)

    artifacts = {}
    functions = {}
    for graph, q_len in (("prefill", 4), ("decode", 2), ("decode_ar", 1)):
        artifacts[graph], functions[graph] = _artifact(
            graph, q_len, layers=2, cache_len=8, vocab=5
        )

    model = FakeLanguageModel(layers=2, vocab=5).eval()
    runner = LANGUAGE.LanguageBCRunner(
        model, torch.eye(4), "cpu", artifacts
    )
    payload = {
        "prompt_input_ids": torch.tensor([2, 3]),
        "prompt_attention_mask": torch.ones(2, dtype=torch.int64),
        "projected_visual_features": torch.empty((0, 4)),
        "prediction_token_ids": {
            "hybrid": torch.tensor([4, 5]),
            "slow": torch.tensor([6]),
        },
        "target_token_ids": torch.tensor([7]),
    }

    result = runner.run(payload)
    runner.close()

    assert model.calls == 1
    assert list(result.outputs) == ["prefill_logits", "pbd_logits", "ar_logits"]
    assert result.outputs["prefill_logits"].shape == (1, 5)
    assert result.outputs["pbd_logits"].shape == (1, 2, 5)
    assert result.outputs["ar_logits"].shape == (1, 1, 5)
    assert result.execution["pbd_q6"]["cache_source"] == "float_prefill"
    assert result.execution["ar_q1"]["cache_source"] == "float_prefill"

    prefill_feed = functions["prefill"].feeds[0]
    assert prefill_feed["prefill_arg2"].shape == (1, 4, 8)
    assert prefill_feed["prefill_arg1"].dtype == np.int32
    for index in range(3, 7):
        assert not np.any(prefill_feed[f"prefill_arg{index}"])

    decode_feed = functions["decode"].feeds[0]
    assert decode_feed["decode_arg3"].dtype == np.float32
    np.testing.assert_array_equal(decode_feed["decode_arg3"][:, -2:], 10.0)
    np.testing.assert_array_equal(decode_feed["decode_arg4"][:, -2:], 11.0)
    np.testing.assert_array_equal(decode_feed["decode_arg5"][:, -2:], 20.0)
    np.testing.assert_array_equal(decode_feed["decode_arg6"][:, -2:], 21.0)
    assert not np.any(decode_feed["decode_arg3"][:, :-2])

    prefill_artifact, prefill_function = _artifact(
        "prefill", 4, layers=2, cache_len=8, vocab=5
    )
    prefill_model = FakeLanguageModel(layers=2, vocab=5).eval()
    prefill_runner = LANGUAGE.LanguageBCRunner(
        prefill_model, torch.eye(4), "cpu", {"prefill": prefill_artifact}
    )
    prefill_result = prefill_runner.run(payload)
    prefill_runner.close()
    assert list(prefill_result.outputs) == ["prefill_logits"]
    assert prefill_model.calls == 0
    assert len(prefill_function.feeds) == 1


def test_language_bc_runner_uses_compiled_int8_prefill_cache(monkeypatch):
    monkeypatch.setattr(LANGUAGE, "CHUNK_SIZE", 4)
    monkeypatch.setattr(LANGUAGE, "CACHE_LEN", 8)
    monkeypatch.setattr(LANGUAGE, "PBD_QUERY_LEN", 2)
    monkeypatch.setattr(LANGUAGE, "VOCAB_SIZE", 5)

    def inputs(graph: str, q_len: int):
        descriptors = [
            _descriptor(f"{graph}_arg0", (1, q_len, 4), np.float16),
            _descriptor(f"{graph}_arg1", (1, 1, q_len), np.int32),
            _descriptor(f"{graph}_arg2", (1, q_len, 8), np.float16),
        ]
        descriptors.extend(
            _descriptor(f"{graph}_arg{index + 3}", (1, 8, 1, 2), np.int8)
            for index in range(4)
        )
        return descriptors

    class MultiOutputFunction:
        def __init__(self, values):
            self.values = values
            self.feeds = []

        def feed(self, *, inputs):
            self.feeds.append(inputs)
            return {name: value.copy() for name, value in self.values.items()}

    prefill_outputs = [
        _descriptor("prefill_result0", (1, 1, 5), np.float16),
        *[
            _descriptor(f"prefill_result{index + 1}", (1, 4, 1, 2), np.int8)
            for index in range(4)
        ],
    ]
    prefill_values = {
        prefill_outputs[0].name: np.ones((1, 1, 5), dtype=np.float16),
        **{
            descriptor.name: np.full(
                descriptor.type.shape, 10 + index, dtype=np.int8
            )
            for index, descriptor in enumerate(prefill_outputs[1:])
        },
    }
    prefill_function = MultiOutputFunction(prefill_values)
    prefill = LANGUAGE.LanguageBCArtifact(
        graph="prefill",
        path=Path("language.prefill_convert.bc"),
        function=prefill_function,
        inputs=inputs("prefill", 4),
        outputs=prefill_outputs,
    )

    decode_output = _descriptor("decode_result0", (1, 2, 5), np.float16)
    decode_function = MultiOutputFunction(
        {decode_output.name: np.full((1, 2, 5), 2.0, dtype=np.float16)}
    )
    decode = LANGUAGE.LanguageBCArtifact(
        graph="decode",
        path=Path("language.decode_convert.bc"),
        function=decode_function,
        inputs=inputs("decode", 2),
        outputs=[decode_output],
    )

    model = FakeLanguageModel(layers=2, vocab=5).eval()
    runner = LANGUAGE.LanguageBCRunner(
        model,
        torch.eye(4),
        "cpu",
        {"decode": decode},
        cache_provider=prefill,
    )
    payload = {
        "prompt_input_ids": torch.tensor([2, 3]),
        "prompt_attention_mask": torch.ones(2, dtype=torch.int64),
        "projected_visual_features": torch.empty((0, 4)),
        "prediction_token_ids": {"hybrid": torch.tensor([4, 5])},
        "target_token_ids": torch.tensor([7]),
    }

    result = runner.run(payload)
    runner.close()

    assert model.calls == 0
    assert result.execution["pbd_q6"]["cache_source"] == "compiled_prefill"
    assert result.timings["compiled_prefill_seconds"] >= 0
    decode_feed = decode_function.feeds[0]
    for index in range(4):
        cache = decode_feed[f"decode_arg{index + 3}"]
        assert cache.dtype == np.int8
        assert not np.any(cache[:, :-2])
        np.testing.assert_array_equal(cache[:, -2:], 10 + index)


def test_pbd_metrics_include_each_query_position():
    reference_logits = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
    candidate_logits = reference_logits.clone()
    cache = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
    eager, _ = LANGUAGE._compare_outputs(
        "pbd_q6",
        (reference_logits, [cache], [cache]),
        (candidate_logits, [cache], [cache]),
        2,
    )
    assert [row["module"] for row in eager["rows"][:3]] == [
        "logits",
        "logits.token_0",
        "logits.token_1",
    ]

    bc = PIPELINE._language_bc_metric_rows(
        {"pbd_logits": reference_logits.numpy()},
        {"pbd_logits": candidate_logits.numpy()},
        candidate_type="LanguageExportedBC",
    )
    assert [row["module"] for row in bc] == [
        "pbd_q6/logits",
        "pbd_q6/logits.token_0",
        "pbd_q6/logits.token_1",
    ]


def test_compact_bc_logits_accepts_prefill_last_row_output(monkeypatch):
    monkeypatch.setattr(LANGUAGE, "VOCAB_SIZE", 5)
    last_row = np.arange(5, dtype=np.float16).reshape(1, 1, 5)

    compact = LANGUAGE._compact_bc_logits("prefill", last_row, active_len=617)

    assert compact.shape == (1, 5)
    np.testing.assert_array_equal(compact, last_row[:, 0])


def test_logits_comparison_preserves_topk_margin_and_signed_choice_gap():
    reference = torch.tensor([[0.0, 4.0, 3.0, 1.0]])
    candidate = torch.tensor([[0.0, 3.5, 3.7, 1.0]])

    result = LANGUAGE.logits_comparison(reference, candidate, top_k=2)
    decision = result["decisions"][0]

    assert result["top1_agreement"] == 0.0
    assert result["top1_flip_rate"] == 1.0
    assert result["topk_overlap"] == 1.0
    assert result["reference_top1_margin"] == pytest.approx(1.0)
    assert result["candidate_top1_margin"] == pytest.approx(0.2)
    assert decision["position"] == [0]
    assert decision["reference_topk"][0]["token_id"] == 1
    assert decision["candidate_topk"][0]["token_id"] == 2
    assert decision["reference_choice_gap"] == pytest.approx(1.0)
    assert decision["candidate_choice_gap"] == pytest.approx(-0.2)
    assert decision["reference_top1_rank_in_candidate"] == 2
    assert decision["candidate_top1_rank_in_reference"] == 2


def test_clone_cache_tensors_isolates_reference_and_candidate_calls():
    source = [torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)]
    reference = LANGUAGE._clone_cache_tensors(source)
    candidate = LANGUAGE._clone_cache_tensors(source)

    reference[0].add_(100)
    candidate[0].sub_(100)

    torch.testing.assert_close(
        source[0], torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)
    )
    assert source[0].data_ptr() != reference[0].data_ptr()
    assert source[0].data_ptr() != candidate[0].data_ptr()
    assert reference[0].data_ptr() != candidate[0].data_ptr()


def test_pipeline_routes_language_exported_bc_to_language_collection(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    model_path = tmp_path / "language.prefill.bc"
    model_path.write_bytes(b"bc")
    calls = []
    monkeypatch.setattr(
        PIPELINE,
        "run_language_bc_collection",
        lambda args: calls.append((args.phase, args.mode, args.level)) or 17,
    )

    result = PIPELINE.main(
        [
            "--mode", "exported-bc",
            "--phase", "language",
            "--level", "small",
            "--input_dir", str(input_dir),
            "--model_path", str(model_path),
        ]
    )

    assert result == 17
    assert calls == [("language", "exported-bc", "small")]


def test_pipeline_rejects_detailed_language_bc(tmp_path):
    root = PIPELINE.parser()
    args = root.parse_args(
        [
            "--mode", "exported-bc",
            "--phase", "language",
            "--level", "medium",
            "--input_dir", str(tmp_path),
            "--model_path", str(tmp_path),
        ]
    )
    with pytest.raises(SystemExit):
        PIPELINE.validate_args(root, args)


def test_language_bc_collection_writes_selected_graph_npz_and_metrics(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_path = input_dir / "sample.pt"
    input_path.write_bytes(b"payload")
    record = {
        "id": "sample",
        "path": str(input_path.resolve()),
        "sha256": PIPELINE.sha256(input_path),
    }
    PIPELINE.atomic_json(
        output_dir / "inputs.json",
        {
            "schema_version": 1,
            "source": str(input_dir.resolve()),
            "count": 1,
            "inputs": [record],
        },
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="ascii")
    reference_path = output_dir / "float" / "outputs" / "sample.npz"
    reference = {
        "prefill_logits": np.ones((1, 5), dtype=np.float32),
        "pbd_logits": np.full((1, 2, 5), 2.0, dtype=np.float32),
    }
    PIPELINE.atomic_named_npz(reference_path, reference)
    PIPELINE.atomic_json(
        output_dir / "float" / "samples" / "sample.json",
        {
            "status": "completed",
            "id": "sample",
            "phase": "language",
            "capture_level": "final",
            "input_sha256": record["sha256"],
            "output_sha256": PIPELINE.sha256(reference_path),
        },
    )
    PIPELINE.atomic_json(
        output_dir / "float" / "stage.json",
        {
            "status": "completed",
            "phase": "language",
            "level": "small",
            "model": str(checkpoint.resolve()),
            "model_sha256": PIPELINE.path_sha256(checkpoint),
            "input_set_sha256": PIPELINE.input_set_sha256(output_dir / "inputs.json"),
        },
    )

    bc_path = tmp_path / "language.decode.bc"
    bc_path.write_bytes(b"decode-bc")
    args = SimpleNamespace(
        mode="exported-bc",
        level="small",
        phase="language",
        nums=None,
        output_dir=output_dir,
        input_dir=input_dir,
        model_path=bc_path,
    )

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def describe_artifacts(self):
            return {"decode": {"graph": "decode"}}

        def run(self, _payload):
            return SimpleNamespace(
                outputs={"pbd_logits": reference["pbd_logits"].copy()},
                execution={"pbd_q6": {"cache_source": "float_prefill"}},
                timings={"pbd_bc_seconds": 0.01},
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(PIPELINE, "prepare_input_index", lambda *_: [record])
    monkeypatch.setattr(PIPELINE, "detect_float_device", lambda: "cpu")
    monkeypatch.setattr(
        PIPELINE, "create_language_model", lambda *_: (object(), object(), object())
    )
    monkeypatch.setattr(
        PIPELINE, "load_language_bc_artifacts", lambda *_args, **_kwargs: {"decode": object()}
    )
    monkeypatch.setattr(PIPELINE, "LanguageBCRunner", FakeRunner)
    monkeypatch.setattr(PIPELINE, "load_language_payload", lambda *_: {})

    assert PIPELINE.run_language_bc_collection(args) == 0
    stage = PIPELINE.read_json(output_dir / "exported_bc" / "stage.json")
    sample = PIPELINE.read_json(output_dir / "exported_bc" / "samples" / "sample.json")
    candidate = PIPELINE.load_named_npz(output_dir / "exported_bc" / "outputs" / "sample.npz")
    assert stage["status"] == "completed"
    assert list(stage["artifact_files"]) == ["decode"]
    assert sample["executed_stages"] == ["pbd_logits"]
    assert sample["intermediate"]["exported_bc"][0]["module"] == "pbd_q6/logits"
    assert list(candidate) == ["pbd_logits"]
