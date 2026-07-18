import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hbdk4.compiler import link
from qwen_vl_utils.vision_process import process_vision_info
from transformers import AutoProcessor


LEAP_ROOT = "/home/kangjie.xu/oe_locateanything/toolchain/leap_llm"
MODEL = "/home/kangjie.xu/oellm_clean/output/ckpt"
PROCESSOR_MODEL = "/home/kangjie.xu/oe_locateanything/main/language/baseline_weights/Qwen2.5-VL-3B-Instruct"
CALIBRATION = (
    LEAP_ROOT
    + "/apis/calibration/calibration_data/mmstar/conversation.json"
)
ROTATION = (
    "/home/kangjie.xu/oellm_clean/output/"
    "qwen2_5_vl_fix009_vision_official_domain/official_hidden_rotation_exact.pt"
)
OFFICIAL_EMBED = "/tmp/official_embed.bin"
OUTPUT = Path(
    "/home/kangjie.xu/oellm_clean/output/"
    "qwen2_5_vl_fix010_language_official_domain"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

LANGUAGE_HBM = OUTPUT / (
    "Qwen2.5-VL-3B-Instruct_language_chunk_256_cache_1024_"
    "w4_nash-p_corenum_4_4.hbm"
)
EMBED = OUTPUT / "Qwen2.5-VL-3B-Instruct_embed_tokens.bin"

CHUNK_SIZE = 256
CACHE_LEN = 1024
DECODE_LEN = 1
CORE_NUM = 4

os.chdir(LEAP_ROOT)
sys.path.insert(0, "/tmp")

from leap_llm.apis.calibration.data_loader import load_message_data
from leap_llm.apis.model.qwen2_5_vl import (
    ensure_visual_dimensions,
    gen_inputs_embeds,
    get_causal_mask,
    get_rope_index,
    init_prefill_kv_cachev,
    padding_input_ids,
    padding_mask,
    remove_repeat,
)
from leap_llm.models.qwen2_5_vl.model import Qwen2_5_VL
from qwen25_language_official_rotation import (
    rotate_qwen25_language_to_official_domain,
    rotate_vision_output_to_official_domain,
)


def compare_embed(candidate_path):
    if not os.path.exists(OFFICIAL_EMBED):
        print("[embed] official reference not found; skip comparison", flush=True)
        return
    vocab_size = 151936
    hidden_size = 2048
    official = np.memmap(
        OFFICIAL_EMBED,
        dtype=np.float16,
        mode="r",
        shape=(vocab_size, hidden_size),
    )
    candidate = np.memmap(
        candidate_path,
        dtype=np.float16,
        mode="r",
        shape=(vocab_size, hidden_size),
    )
    ids = np.random.default_rng(20260717).choice(vocab_size, 4096, replace=False)
    left = np.array(candidate[ids], dtype=np.float32).reshape(-1)
    right = np.array(official[ids], dtype=np.float32).reshape(-1)
    cosine = float(
        np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    )
    print(
        f"[embed] Fix010 vs official cosine={cosine:.9f} "
        f"max_diff={np.max(np.abs(left - right)):.7g} "
        f"rmse={np.sqrt(np.mean((left - right) ** 2)):.7g}",
        flush=True,
    )


print("[FIX #010 official_language_domain] [1] load model", flush=True)
wrapper = Qwen2_5_VL.build(
    MODEL,
    chunk_size=CHUNK_SIZE,
    cache_len=CACHE_LEN,
    decode_seq_len=DECODE_LEN,
    w_bits=4,
    input_model_format="hf",
    image_width=448,
    image_height=448,
)
model = wrapper.model
text_model = wrapper.get_text_model()
vision_model = wrapper.get_visual_model()
rotation = torch.load(ROTATION, weights_only=True).float()

orthogonal_error = rotate_qwen25_language_to_official_domain(
    text_model,
    rotation,
    device="cuda:0",
)
rotate_vision_output_to_official_domain(
    vision_model,
    rotation,
    device="cuda:0",
)
print(
    f"[FIX #010] language and calibration Vision rotated to official domain; "
    f"orthogonal_max_error={orthogonal_error:.9g}",
    flush=True,
)

embed = text_model.embed_tokens.weight.detach().to(torch.float16).cpu().numpy()
embed.tofile(EMBED)
print(f"[2] saved rotated embed {EMBED} bytes={EMBED.stat().st_size}", flush=True)
compare_embed(str(EMBED))

device = "cuda:0"
dtype = torch.float32
text_model.to(device=device, dtype=dtype)
vision_model.to(device=device, dtype=dtype)
model.compile_mode(False)
processor = AutoProcessor.from_pretrained(PROCESSOR_MODEL, use_fast=True)
messages_list = list(
    load_message_data(CALIBRATION, model_type="qwen2_5-vl-3b")
)
print(f"[3] calibration samples={len(messages_list)}", flush=True)

config = model.get_config()
image_grid = torch.tensor([[1, 32, 32]], device=device)
window_index, _ = vision_model.get_window_index(image_grid)
num_vision_tokens = 256
max_prefill_tokens = config.text_config.max_prefill_text_token + num_vision_tokens
num_layers = config.text_config.num_hidden_layers
head_dim = config.text_config.hidden_size // config.text_config.num_attention_heads
num_kv_heads = config.text_config.num_key_value_heads
max_lm_tokens = 4096

for index, messages in enumerate(messages_list):
    if isinstance(messages, dict):
        messages = [messages]
    text = processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    messages = ensure_visual_dimensions(messages, 448, 448)
    for item in messages:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        for part in item["content"]:
            image = part.get("image") if isinstance(part, dict) else None
            if isinstance(image, str) and image.startswith("leap_llm/"):
                part["image"] = LEAP_ROOT + "/" + image[len("leap_llm/") :]
    images, videos = process_vision_info([messages])
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    attention_mask = padding_mask(
        inputs["attention_mask"], max_prefill_tokens, left=True
    )
    padded_input_ids = padding_input_ids(
        input_ids, max_prefill_tokens, left=True
    )
    position_ids, _ = get_rope_index(
        config,
        padded_input_ids,
        image_grid_thw,
        attention_mask,
    )
    position_ids = position_ids.squeeze().unsqueeze(0)

    with torch.no_grad():
        input_embeddings = text_model.embed_tokens(padded_input_ids)
        pixel_values = remove_repeat(pixel_values)
        image_embeddings = vision_model.forward(pixel_values.unsqueeze(0))
        input_embeddings = gen_inputs_embeds(
            config,
            padded_input_ids,
            input_embeddings,
            image_embeddings,
            window_index,
        )
        cache_keys, cache_values = init_prefill_kv_cachev(
            input_embeddings.shape[0],
            num_layers,
            num_kv_heads,
            head_dim,
            attention_mask,
            max_lm_tokens,
        )
        caches = [cache.to(device) for cache in cache_keys + cache_values]
        causal_mask = get_causal_mask(
            attention_mask,
            max_lm_tokens,
            -32768,
        ).squeeze(0).to(device)
        text_model.forward(
            inputs_embeds=input_embeddings,
            position_ids=position_ids.to(device),
            attention_mask=causal_mask,
            caches=caches,
        )
    if (index + 1) % 10 == 0:
        print(f"[calib] {index + 1}", flush=True)

print("[4] calibration done", flush=True)
model.compile_mode(True)
model.to("cpu", dtype=torch.float16)
del rotation, processor, messages_list
gc.collect()
torch.cuda.empty_cache()

config = model.get_config()
stage_specs = [
    (
        "prefill",
        text_model.get_leap_input_types_text_model(
            num_layers,
            CHUNK_SIZE,
            CACHE_LEN,
        ),
    ),
    (
        "decode",
        text_model.get_leap_input_types_decode_model(
            num_layers,
            DECODE_LEN,
            CACHE_LEN,
        ),
    ),
]

converted_modules = []
for stage_name, input_types in stage_specs:
    print(f"[5] export {stage_name} BC", flush=True)
    bc_path = OUTPUT / f"language.{stage_name}.bc"
    bc = text_model.export_module(
        input_types,
        stage_name,
        str(bc_path),
        high_precision_qpp=True,
    )
    print(f"[6] convert {stage_name}", flush=True)
    convert_path = OUTPUT / f"language.{stage_name}_convert.bc"
    converted = model.convert_mlir(
        bc,
        str(convert_path),
        enable_vpu=True,
        march="nash-p",
        dynamic_quant=True,
    )
    converted.functions[0].remove_io_op(["Dequantize", "Quantize"])
    converted_modules.append((stage_name, converted))

hbos = []
for stage_name, converted in converted_modules:
    print(f"[7] compile {stage_name} HBO", flush=True)
    print(
        "[PARAMS] march=nash-p opt=2 jobs=16 core_num=4 "
        "input_no_padding=True output_no_padding=True enable_hpc=True "
        "max_l2m_size=25165824",
        flush=True,
    )
    hbo = model.compile_hbo(
        converted,
        str(OUTPUT / f"language.{stage_name}.hbo"),
        march="nash-p",
        opt=2,
        jobs=16,
        progress_bar=True,
        input_no_padding=True,
        output_no_padding=True,
        enable_hpc=True,
        core_num=CORE_NUM,
        max_l2m_size=25165824,
    )
    hbos.append(hbo)

print("[8] link Language HBM", flush=True)
link(hbos, str(LANGUAGE_HBM))
print(
    f"[DONE] language={LANGUAGE_HBM} bytes={LANGUAGE_HBM.stat().st_size} "
    f"embed={EMBED} bytes={EMBED.stat().st_size}",
    flush=True,
)
