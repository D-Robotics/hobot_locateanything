# LocateAnything Source Review

This review records the source facts that constrain the S600 port. The source
was inspected under `Eagle/Embodied`; the downloaded checkpoint configuration
and remote-code files remain authoritative when repository and checkpoint code
differ.

## Model Contract

| Component | Checkpoint value |
|---|---|
| Language decoder | Qwen2.5/Qwen2, 36 layers, hidden 2048, MLP 11008 |
| Attention | 16 query heads, 2 KV heads, head dim 128 |
| RoPE | 1D, theta 1,000,000 |
| Vocabulary | 152,681, tied input/output embeddings |
| PBD | block size 6, text-mask token 151676 |
| Vision | MoonViT, 27 layers, hidden 1152, 16 heads, patch 14 |
| Vision merge | 2x2 patch merge, then 4608 -> 2048 -> 2048 projector |
| Image token | 151665 |
| Coordinate tokens | 151677 through 152677 |

## Upstream Paths

- `eaglevl/model/locany`: primarily training-oriented implementation.
- `eaglevl/utils/locany`: hybrid PBD inference implementation used by the
  demos.
- `eaglevl/model/moon_vit/modeling_vit.py`: native MoonViT definition.
- Checkpoint remote code: closest to `eaglevl/utils/locany`, but not byte-identical
  in `modeling_locateanything.py`, `modeling_qwen2.py`, and processing code.

## Generation Semantics

1. The processor inserts an image placeholder and replaces it with a variable
   number of MoonViT visual embeddings.
2. MTP/PBD prepares six positions: the real trailing token plus five text-mask
   tokens.
3. The six-token block shifts its position IDs with
   `position_ids[-6:] -= 1`.
4. Hybrid mode falls back to autoregressive decoding when a box pattern is
   malformed and switches back to PBD after `</box>`.
5. Output structure uses `<ref>...</ref><box>x1 y1 x2 y2</box>` and the 1001
   coordinate-token IDs.

## Deployment Consequences

- The Qwen2.5-VL reference path validates the common OELLM compiler, HBM,
  embedding-table, and HBRT execution chain. LocateAnything extends the Host
  runtime with its tokenizer, image-token insertion, 1D position IDs, PBD
  mask, six-token sampler, and box parser.
- Hybrid decoding needs both q=6 PBD and q=1 AR graphs. Fix #011 exports them
  as `decode` and `decode_ar` in the same Language HBM.
- The current fixed 448x448 Vision graph accepts `(1,1024,588)` and emits 256
  visual tokens. An upstream dynamic-resolution example emitted 925 tokens.
  The Host prompt selects the token count associated with the compiled Vision
  profile.
- The same hidden-domain transform must be folded into the Language stack,
  embedding table, and MoonViT projector. Shape compatibility alone is not
  sufficient.
- Qwen Fix #009/#010 establishes the compiler and hidden-domain strategy.
  LocateAnything then applies the same validation method to its own HBM,
  PyTorch reference, PBD flow, and S600 grounding output.
