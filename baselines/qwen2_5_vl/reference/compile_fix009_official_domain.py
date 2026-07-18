import gc
import os
from pathlib import Path

import torch
from qwen_vl_utils.vision_process import process_vision_info
from transformers import AutoProcessor

os.chdir("/home/kangjie.xu/oe_locateanything/toolchain/leap_llm")

from leap_llm.apis.calibration.data_loader import load_message_data
from leap_llm.apis.model.qwen2_5_vl import ensure_visual_dimensions, remove_repeat
from leap_llm.models.qwen2_5_vl.model import Qwen2_5_VL


MODEL = "/home/kangjie.xu/oe_locateanything/main/language/baseline_weights/Qwen2.5-VL-3B-Instruct"
CALIBRATION = "/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/apis/calibration/calibration_data/mmstar/conversation.json"
ROTATION = "/tmp/official_hidden_rotation_exact.pt"
OUTPUT = Path(
    "/home/kangjie.xu/oellm_clean/output/qwen2_5_vl_fix009_vision_official_domain"
)
OUTPUT.mkdir(parents=True, exist_ok=True)
HBM = OUTPUT / "Qwen2.5-VL-3B-Instruct_vision_448x448_w8_nash-p_corenum_4.hbm"

print("[FIX #009 official_domain] [1] loading model", flush=True)
qwen = Qwen2_5_VL.build(
    MODEL,
    chunk_size=256,
    cache_len=1024,
    input_model_format="llmc",
    image_width=448,
    image_height=448,
)
vision = qwen.get_visual_model()

rotation = torch.load(ROTATION, weights_only=True).float()
assert rotation.shape == (2048, 2048)
identity = torch.eye(2048)
orthogonal_error = (rotation.T @ rotation - identity).abs().max().item()
assert orthogonal_error < 1e-5, orthogonal_error

final_projection = vision.merger.mlp.proj1
with torch.no_grad():
    final_projection.weight.copy_(rotation.T @ final_projection.weight.float())
    final_projection.bias.copy_(final_projection.bias.float() @ rotation)
print(
    f"[FIX #009] folded official hidden rotation into merger.proj1; "
    f"orthogonal_max_error={orthogonal_error:.9g}",
    flush=True,
)

vision.to(device="cuda:0", dtype=torch.float32)
qwen.model.compile_mode(False)
processor = AutoProcessor.from_pretrained(MODEL, use_fast=True)
messages = list(load_message_data(CALIBRATION, model_type="qwen2_5-vl-3b"))
print("[2] calibration samples", len(messages), flush=True)

for index, message in enumerate(messages):
    if isinstance(message, dict):
        message = [message]
    text = processor.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    message = ensure_visual_dimensions(message, 448, 448)
    for item in message:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        for part in item["content"]:
            image = part.get("image") if isinstance(part, dict) else None
            if isinstance(image, str) and image.startswith("leap_llm/"):
                part["image"] = (
                    "/home/kangjie.xu/oe_locateanything/toolchain/leap_llm/"
                    + image[len("leap_llm/") :]
                )
    images, videos = process_vision_info([message])
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    if "pixel_values" not in inputs:
        continue
    pixel_values = remove_repeat(inputs["pixel_values"]).to("cuda:0")
    with torch.no_grad():
        vision.forward(pixel_values.unsqueeze(0))
    if (index + 1) % 10 == 0:
        print("[calib]", index + 1, flush=True)

print("[3] calibration done", flush=True)
qwen.model.compile_mode(True)
qwen.model.to("cpu", dtype=torch.float16)
del rotation, identity
gc.collect()
torch.cuda.empty_cache()

print("[4] export BC", flush=True)
input_types = vision.get_leap_input_types()
bc = vision.export_module(
    input_types,
    "visual",
    str(OUTPUT / "vision.visual.bc"),
    high_precision_qpp=True,
)

print("[5] convert", flush=True)
mlir = qwen.model.convert_mlir(
    bc,
    str(OUTPUT / "vision.visual_convert.bc"),
    enable_vpu=True,
    march="nash-p",
    dynamic_quant=True,
)
mlir.functions[0].remove_io_op(["Dequantize", "Quantize"])

print("[6] compile HBO", flush=True)
print(
    "[PARAMS] march=nash-p opt=2 jobs=16 core_num=4 "
    "input_no_padding=True output_no_padding=True enable_hpc=True "
    "max_l2m_size=25165824",
    flush=True,
)
hbo = qwen.model.compile_hbo(
    mlir,
    str(OUTPUT / "vision.visual.hbo"),
    march="nash-p",
    opt=2,
    jobs=16,
    progress_bar=True,
    input_no_padding=True,
    output_no_padding=True,
    enable_hpc=True,
    core_num=4,
    max_l2m_size=25165824,
)

print("[7] link HBM", flush=True)
qwen.model.link_models([hbo], str(HBM))
print("[DONE]", HBM, HBM.stat().st_size, flush=True)
