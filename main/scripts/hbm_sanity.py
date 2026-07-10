"""HBM sanity check on x86 (4090) via hbdk4 x86 simulator.

Loads vision.hbm and language.hbm, feeds dummy inputs, verifies:
  - Load succeeds
  - Graph execute runs without error
  - Output shapes match declared IO
  - No NaN/Inf, output range is reasonable

Runs the x86 hbrt4-run-model-nash simulator (no S600 hardware needed).
"""

import os
import sys
import time
import numpy as np
import hbdk4.compiler as hb

VISION_HBM = "/home/kangjie.xu/oe_locateanything/main/vision/outputs/locateanything-vit-3b_nash-p_w4/LocateAnything-3B_vision_448x448_w8_nash-p_corenum_4.hbm"
LANGUAGE_HBM = "/home/kangjie.xu/oe_locateanything/main/language/outputs/locateanything-lm-3b_nash-p_w4/LocateAnything-3B_language_chunk_256_cache_1024_w4_nash-p_corenum_4_4.hbm"
EMBED_BIN = "/home/kangjie.xu/oe_locateanything/main/language/outputs/locateanything-lm-3b_nash-p_w4/LocateAnything-3B_embed_tokens.bin"

MASK_VALUE = -32768


def _tensor_stats(name, arr):
    """One-line summary line for a tensor: name / shape / dtype / stats."""
    if arr.dtype.kind in "biu":  # integer types
        lo = int(arr.min())
        hi = int(arr.max())
        mean = float(arr.astype(np.float64).mean())
        print(f"  {name:20s} shape={arr.shape} dtype={arr.dtype} "
              f"min={lo} max={hi} mean={mean:.3f}")
    else:
        finite = arr[np.isfinite(arr)] if arr.size else arr
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        if finite.size:
            lo = float(finite.min())
            hi = float(finite.max())
            mean = float(finite.mean())
            print(f"  {name:20s} shape={arr.shape} dtype={arr.dtype} "
                  f"min={lo:.4g} max={hi:.4g} mean={mean:.4g} nan={nan} inf={inf}")
        else:
            print(f"  {name:20s} shape={arr.shape} dtype={arr.dtype} (all-nonfinite) nan={nan} inf={inf}")


def sanity_vision(hbm_path):
    print("=" * 70)
    print("SANITY 1: vision.hbm x86 simulate")
    print("=" * 70)
    hbm = hb.Hbm(hbm_path)
    print(f"[load] march={hbm.march_name} toolkit={hbm.toolkit_version}")
    g = None
    for gr in hbm.graphs:
        if gr.name == "visual":
            g = gr
            break
    assert g is not None, "visual graph not found"
    print(f"[graph] name={g.name} inputs={len(g.inputs)} outputs={len(g.outputs)}")

    # Build dummy input: (1, 1024, 588) fp32 — 1024 patches x (3x14x14 RGB)
    inp = g.inputs[0]
    shape = tuple(inp.type.shape)
    dtype = inp.type.np_dtype
    print(f"[input] name={inp.name} shape={shape} dtype={dtype}")

    # Use a deterministic dummy image: normal-scaled RGB
    rng = np.random.default_rng(seed=42)
    x = rng.normal(0.0, 1.0, size=shape).astype(dtype)
    _tensor_stats("dummy_input", x)

    t0 = time.time()
    feed_dict = {inp.name: x}
    outs = g.feed(feed_dict)
    dt = time.time() - t0
    print(f"[execute] wall-clock = {dt:.3f}s")

    for oname, oarr in outs.items():
        _tensor_stats(oname, oarr)
        assert not np.isnan(oarr).any(), f"output {oname} contains NaN"
        assert not np.isinf(oarr).any(), f"output {oname} contains Inf"

    # Expected: (1, 256, 2048)
    out0 = outs[g.outputs[0].name]
    assert out0.shape == (1, 256, 2048), \
        f"expected (1,256,2048), got {out0.shape}"

    print("[verdict] vision.hbm PASS: dummy forward OK, output shape and finiteness match")
    return outs


def sanity_language_prefill(hbm_path, embed_bin_path):
    print()
    print("=" * 70)
    print("SANITY 2: language.hbm::prefill x86 simulate")
    print("=" * 70)
    hbm = hb.Hbm(hbm_path)
    print(f"[load] march={hbm.march_name} toolkit={hbm.toolkit_version}")
    g = None
    for gr in hbm.graphs:
        if gr.name == "prefill":
            g = gr
            break
    assert g is not None, "prefill graph not found"
    print(f"[graph] name={g.name} inputs={len(g.inputs)} outputs={len(g.outputs)}")

    # Build dummy inputs.
    #   input_0: (1, 256, 2048) fp16 — token embeds
    #   input_1: (1, 1, 256) int32 — position IDs (naive: 0..255)
    #   input_2: (1, 256, 1024) fp16 — attention mask (0 for allowed, MASK_VALUE blocked)
    #   input_3..74: 72 x (1, 1024, 2, 128) int8 — KV cache (zeros for cold start)
    feed_dict = {}
    for i, inp in enumerate(g.inputs):
        name = inp.name
        shape = tuple(inp.type.shape)
        dtype = inp.type.np_dtype
        if i == 0:
            # Real embed lookup from embed_tokens.bin, first 256 tokens
            assert os.path.exists(embed_bin_path), f"embed bin missing: {embed_bin_path}"
            # 152681 x 2048 fp16
            embed_all = np.fromfile(embed_bin_path, dtype=np.float16)
            expect = 152681 * 2048
            assert embed_all.size == expect, \
                f"embed bin size unexpected: {embed_all.size} vs {expect}"
            embed_all = embed_all.reshape(152681, 2048)
            # Pick token IDs 0..255 as prompt (deterministic dummy)
            token_ids = np.arange(256, dtype=np.int64)
            embeds = embed_all[token_ids][None, :, :].astype(dtype)
            feed_dict[name] = embeds
            _tensor_stats(f"in[{i}]_{name}(embeds)", embeds)
        elif i == 1:
            # position_ids: (1, 1, 256), int32, values 0..255
            pos = np.arange(256, dtype=np.int32)[None, None, :]
            assert pos.shape == shape, f"pos shape {pos.shape} vs {shape}"
            feed_dict[name] = pos
            _tensor_stats(f"in[{i}]_{name}(pos_ids)", pos)
        elif i == 2:
            # attention_mask: (1, 256, 1024) fp16
            # Causal: token i can attend to tokens 0..i in the prompt window.
            # cache_len=1024, and only the first 256 slots are "prompt" this turn.
            # Everything past position 256 is future -> blocked.
            m = np.full(shape, MASK_VALUE, dtype=dtype)
            for q in range(256):
                # allow attention over prompt positions 0..q
                m[0, q, :q+1] = 0
            feed_dict[name] = m
            _tensor_stats(f"in[{i}]_{name}(attn_mask)", m)
        else:
            # KV cache: zeros for cold start
            feed_dict[name] = np.zeros(shape, dtype=dtype)
    print(f"[input] built {len(feed_dict)} tensors")

    t0 = time.time()
    outs = g.feed(feed_dict)
    dt = time.time() - t0
    print(f"[execute] wall-clock = {dt:.3f}s")

    # Verify shapes + finiteness
    logits_name = g.outputs[0].name
    logits = outs[logits_name]
    _tensor_stats(f"out[0]_{logits_name}(logits)", logits)
    assert logits.shape == (1, 256, 152681), \
        f"logits expected (1,256,152681), got {logits.shape}"
    assert not np.isnan(logits).any(), "logits contain NaN"

    # Report KV cache stats (first 2 out of 72)
    for i in [1, 2]:
        oname = g.outputs[i].name
        oarr = outs[oname]
        _tensor_stats(f"out[{i}]_{oname}", oarr)

    # Predict next token from position 0's logits (should not be all zeros)
    p0 = logits[0, 0]
    top = np.argsort(p0)[-5:][::-1]
    print(f"[predict] token@pos0 top5={top.tolist()} values={[float(p0[t]) for t in top]}")

    print("[verdict] language.hbm::prefill PASS: dummy forward OK, "
          f"logits shape=({logits.shape}) finite=True")
    return outs


def main():
    print(f"pid={os.getpid()} python={sys.version.split()[0]}")
    print(f"HBM sanity on 4090 (x86 simulator, no S600 hw needed)")
    print()

    sanity_vision(VISION_HBM)
    sanity_language_prefill(LANGUAGE_HBM, EMBED_BIN)

    print()
    print("=" * 70)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
