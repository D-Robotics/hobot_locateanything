from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "validate/hbm_sanity.py"
SPEC = importlib.util.spec_from_file_location("hbm_sanity", SCRIPT)
sanity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sanity
SPEC.loader.exec_module(sanity)


def tensor(name: str, shape: tuple[int, ...], dtype: str = "float16"):
    return sanity.TensorDescriptor(name, shape, dtype)


def release_graphs(profile=None):
    profile = profile or sanity.ExpectedProfile()
    cache = (1, profile.cache_length, profile.cache_groups, profile.head_dim)
    graphs = {
        "visual": sanity.GraphDescriptor(
            "visual",
            (tensor("pixels", (1, profile.patch_tokens, profile.patch_vector)),),
            (tensor("features", (1, profile.visual_tokens, profile.hidden_size)),),
        )
    }
    query_lengths = {
        **sanity.LANGUAGE_GRAPH_QUERIES,
        "prefill": profile.prefill_query,
        "decode": profile.pbd_query,
        "decode_ar": profile.ar_query,
    }
    for name, q_len in query_lengths.items():
        inputs = [
            tensor("embeddings", (1, q_len, profile.hidden_size)),
            tensor("position_ids", (1, 1, q_len), "int32"),
            tensor("attention_mask", (1, q_len, profile.cache_length)),
        ]
        inputs.extend(tensor(f"cache_{i}", cache, "float32") for i in range(profile.cache_tensor_count))
        outputs = [tensor("logits", (1, q_len, profile.vocab_size))]
        outputs.extend(
            tensor(f"update_{i}", (1, q_len, profile.cache_groups, profile.head_dim), "float32")
            for i in range(profile.cache_tensor_count)
        )
        graphs[name] = sanity.GraphDescriptor(name, tuple(inputs), tuple(outputs))
    return graphs


def test_release_descriptor_contract_passes_and_records_derived_geometry():
    summary = sanity.validate_descriptor_contract(release_graphs(), sanity.ExpectedProfile())
    assert summary["derived"] == {
        "patch_vector": 588,
        "patch_tokens": 2304,
        "visual_tokens": 576,
        "cache_tensor_count": 72,
    }
    assert summary["language_graphs"]["decode"]["query_length"] == 6
    assert summary["language_graphs"]["decode_ar"]["query_length"] == 1
    assert summary["language_graphs"]["decode_pbd_q12"]["query_length"] == 12
    assert summary["language_graphs"]["decode_ar_q5"]["query_length"] == 5


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("old_vision", "visual input"),
        ("old_prefill", "prefill embeddings"),
        ("old_cache", "prefill attention mask"),
        ("wrong_pbd", "decode embeddings"),
        ("missing_ar", r"required graph\(s\) missing"),
    ),
)
def test_old_or_incomplete_profiles_fail_closed(mutation, message):
    graphs = release_graphs()
    if mutation == "old_vision":
        graph = graphs["visual"]
        graphs["visual"] = sanity.GraphDescriptor(
            graph.name, (tensor("pixels", (1, 1024, 588)),), graph.outputs
        )
    elif mutation == "old_prefill":
        graph = graphs["prefill"]
        inputs = list(graph.inputs)
        inputs[0] = tensor("embeddings", (1, 256, 2048))
        graphs["prefill"] = sanity.GraphDescriptor(graph.name, tuple(inputs), graph.outputs)
    elif mutation == "old_cache":
        graph = graphs["prefill"]
        inputs = list(graph.inputs)
        inputs[2] = tensor("attention_mask", (1, 1024, 1024))
        graphs["prefill"] = sanity.GraphDescriptor(graph.name, tuple(inputs), graph.outputs)
    elif mutation == "wrong_pbd":
        graph = graphs["decode"]
        inputs = list(graph.inputs)
        inputs[0] = tensor("embeddings", (1, 1, 2048))
        graphs["decode"] = sanity.GraphDescriptor(graph.name, tuple(inputs), graph.outputs)
    else:
        del graphs["decode_ar"]

    with pytest.raises(ValueError, match=message):
        sanity.validate_descriptor_contract(graphs, sanity.ExpectedProfile())


def test_cache_updates_must_match_each_graph_query_length():
    graphs = release_graphs()
    graph = graphs["decode"]
    outputs = list(graph.outputs)
    outputs[-1] = tensor("bad_update", (1, 1, 2, 128), "float32")
    graphs["decode"] = sanity.GraphDescriptor(graph.name, graph.inputs, tuple(outputs))
    with pytest.raises(ValueError, match="cache update shapes"):
        sanity.validate_descriptor_contract(graphs, sanity.ExpectedProfile())


def test_embed_file_size_is_derived_from_profile(tmp_path):
    profile = sanity.ExpectedProfile(vocab_size=5, hidden_size=4)
    embed = tmp_path / "embed.bin"
    embed.write_bytes(b"\0" * (5 * 4 * 2))
    result = sanity.validate_embed_file(embed, profile, scan_finite=False)
    assert result["bytes"] == 40
    assert result["shape"] == [5, 4]

    embed.write_bytes(b"\0" * 38)
    with pytest.raises(ValueError, match="embedding bytes"):
        sanity.validate_embed_file(embed, profile, scan_finite=False)


def test_cli_defaults_are_the_v5_release_profile():
    parser = sanity.build_parser()
    args = parser.parse_args([
        "--vision-hbm", "vision.hbm",
        "--language-hbm", "language.hbm",
        "--embed-bin", "embed.bin",
    ])
    profile = sanity._profile_from_args(args)
    assert args.mode == "descriptor-only"
    assert (profile.image_size, profile.prefill_query, profile.cache_length) == (672, 1024, 4096)
    assert (profile.pbd_query, profile.ar_query) == (6, 1)
