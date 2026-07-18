import os
import sys

import torch


LEAP_ROOT = "/home/kangjie.xu/oe_locateanything/toolchain/leap_llm"
MODEL = "/home/kangjie.xu/oellm_clean/output/ckpt"
ROTATION = (
    "/home/kangjie.xu/oellm_clean/output/"
    "qwen2_5_vl_fix009_vision_official_domain/official_hidden_rotation_exact.pt"
)

os.chdir(LEAP_ROOT)
sys.path.insert(0, "/tmp")

from leap_llm.models.qwen2_5_vl.model import Qwen2_5_VL
from qwen25_language_official_rotation import (
    rotate_qwen25_language_to_official_domain,
)


def cosine(left, right):
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    return float(
        torch.dot(left, right) / (torch.linalg.norm(left) * torch.linalg.norm(right))
    )


device = "cuda:0"
rotation = torch.load(ROTATION, weights_only=True).float()
wrapper = Qwen2_5_VL.build(
    MODEL,
    chunk_size=8,
    cache_len=16,
    input_model_format="hf",
    image_width=448,
    image_height=448,
)
model = wrapper.model
text_model = wrapper.get_text_model().to(device=device, dtype=torch.float32)
model.compile_mode(False)

token_ids = torch.tensor([[1, 42, 1024, 151]], dtype=torch.long, device=device)
position_ids = torch.arange(token_ids.shape[1], device=device).view(1, 1, -1)
original_inputs = text_model.embed_tokens(token_ids)

with torch.no_grad():
    original_logits, original_keys, original_values = text_model.forward(
        inputs_embeds=original_inputs,
        position_ids=position_ids,
        attention_mask=None,
        caches=None,
    )

orthogonal_error = rotate_qwen25_language_to_official_domain(
    text_model,
    rotation,
    device=device,
)
rotated_inputs = text_model.embed_tokens(token_ids)
expected_inputs = original_inputs @ rotation.to(device)

with torch.no_grad():
    rotated_logits, rotated_keys, rotated_values = text_model.forward(
        inputs_embeds=rotated_inputs,
        position_ids=position_ids,
        attention_mask=None,
        caches=None,
    )

print("orthogonal_max_error", orthogonal_error)
print("input_cosine", cosine(rotated_inputs, expected_inputs))
print("input_max_diff", float((rotated_inputs - expected_inputs).abs().max()))
print("logits_cosine", cosine(rotated_logits, original_logits))
print("logits_max_diff", float((rotated_logits - original_logits).abs().max()))
print(
    "logits_rmse",
    float(torch.sqrt(torch.mean((rotated_logits - original_logits).float().square()))),
)

key_cosines = [cosine(left, right) for left, right in zip(rotated_keys, original_keys)]
value_cosines = [
    cosine(left, right) for left, right in zip(rotated_values, original_values)
]
print("kv_key_cosine_min", min(key_cosines))
print("kv_value_cosine_min", min(value_cosines))
print("argmax_equal", bool(torch.equal(rotated_logits.argmax(-1), original_logits.argmax(-1))))
