"""LocateAnything-3B special token id map.

Ground truth precedence:
  1. `config.json` at the checkpoint root
  2. `configuration_locateanything.py` defaults (only for keys not in config.json)
  3. NEVER hardcode: the checkpoint may override these ids in a later release

All ids below are read once from config.json at LocateAnythingApi.__init__
and passed to the compile pipeline; the constants here are only the
canonical *fallback* / documentation of what each token means.

See report chapter 5.4 for the full table with source references.
"""

# Vision context marker: replaced in `modeling_locateanything.py:230`
# NB. `configuration_locateanything.py:58` defaults 151667 but the LocateAnything-3B
# checkpoint config.json overrides to 151665. Only trust config.json at runtime.
IMAGE_TOKEN_INDEX_DEFAULT = 151665

# Box / ref / coord anchors used by `generate_utils.handle_pattern`
BOX_START_TOKEN_ID = 151668       # <box>
BOX_END_TOKEN_ID = 151669         # </box>
REF_START_TOKEN_ID = 151672       # <ref>
REF_END_TOKEN_ID = 151673         # </ref>

# 1001-token coord range: <0> ... <1000> inclusive, id 151677..152677
COORD_START_TOKEN_ID = 151677     # <0>
COORD_END_TOKEN_ID = 152677       # <1000>
COORD_TOKEN_COUNT = COORD_END_TOKEN_ID - COORD_START_TOKEN_ID + 1  # 1001

# PBD / MTP scaffolding
TEXT_MASK_TOKEN_ID = 151676       # <text_mask>  — AR/MTP sentinel
NULL_TOKEN_ID = 152678            # <null>       — empty-box sentinel used by handle_pattern
SWITCH_TOKEN_ID = 152679          # <switch>     — reserved training label, not checked in modeling

# `none` is not a new special token — it is Qwen's original subword id for "none",
# reused for the <box>none</box> empty-box convention.
NONE_TOKEN_ID = 4064


def build_id_map_from_config(cfg_dict: dict) -> dict:
    """Return {name: id} pulled from a loaded config.json dict.

    Falls back to the constants above when a key is missing in the config.
    """
    text_cfg = cfg_dict.get("text_config", {})
    return {
        "image_token_index": cfg_dict.get("image_token_index", IMAGE_TOKEN_INDEX_DEFAULT),
        "box_start_token_id": cfg_dict.get("box_start_token_id", BOX_START_TOKEN_ID),
        "box_end_token_id": cfg_dict.get("box_end_token_id", BOX_END_TOKEN_ID),
        "ref_start_token_id": cfg_dict.get("ref_start_token_id", REF_START_TOKEN_ID),
        "ref_end_token_id": cfg_dict.get("ref_end_token_id", REF_END_TOKEN_ID),
        "coord_start_token_id": cfg_dict.get("coord_start_token_id", COORD_START_TOKEN_ID),
        "coord_end_token_id": cfg_dict.get("coord_end_token_id", COORD_END_TOKEN_ID),
        "text_mask_token_id": text_cfg.get("text_mask_token_id", TEXT_MASK_TOKEN_ID),
        "null_token_id": text_cfg.get("null_token_id", NULL_TOKEN_ID),
        "switch_token_id": text_cfg.get("switch_token_id", SWITCH_TOKEN_ID),
        "none_token_id": cfg_dict.get("none_token_id", NONE_TOKEN_ID),
    }
