"""Hidden-domain transforms shared by LocateAnything vision and language.

The built-in 2048-dimensional transform was recovered from the public
Qwen2.5-VL S600 reference artifacts and validated by the Qwen Fix #009/#010
experiments. It is a signed, normalized Sylvester Hadamard matrix. Applying
the transform is an offline weight rewrite; no rotation operator is added to
the runtime graph.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch


REFERENCE_HIDDEN_SIZE = 2048
REFERENCE_ROTATION_SHA256 = (
    "19c57d8f5cd400e00c2744dbb4bd898da98fc8ad30ef97060fa0ba4cf5ed10e5"
)
_NEGATIVE_ROW_MASK_HEX = (
    "09d897a1920585a63ea705b269a7af2064cfc245cbaaaebedd9af3750ca78523"
    "cc09aedcd0a3810178043cad1ef568a44c600d62e27e8c8ad6fe4b0cf5a0cab"
    "15d65f5943ecf0260d86a5fc3c2c6e9d43ec04323825da52b098656d2dbf59"
    "8a43c73eddc9af17d30c061c17bd060634e85c34febcd2acaf9bdd31dcbbfba5"
    "cf49d4aedba51b0e6fc007f9574b68ccdc376c9477a5030c3cfb34cb3234306"
    "2b8179de91321db40e5915a8ea4225409db41083d6025b422daf35a0adcd92ee"
    "89200f3abfaf051aa9926505dc13d45a4dc7058f5d8dbade324033fc610fd57f"
    "90e451396f4a13e9ddff130905d71580961efd1899070bd1a6b85416dc5d6a0"
    "51bb6"
)


def _sylvester_hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two: {size}")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def build_reference_hidden_rotation(hidden_size: int = REFERENCE_HIDDEN_SIZE) -> torch.Tensor:
    if hidden_size != REFERENCE_HIDDEN_SIZE:
        raise ValueError(
            "The built-in S600 reference rotation is defined only for hidden_size=2048; "
            "provide --hidden_rotation_path for another size."
        )

    packed = bytes.fromhex(_NEGATIVE_ROW_MASK_HEX)
    signs = torch.ones(hidden_size, dtype=torch.float32)
    for index in range(hidden_size):
        if packed[index // 8] & (1 << (index % 8)):
            signs[index] = -1.0

    rotation = signs[:, None] * _sylvester_hadamard(hidden_size)
    rotation.mul_(1.0 / math.sqrt(hidden_size))
    digest = hashlib.sha256(rotation.contiguous().numpy().tobytes()).hexdigest()
    if digest != REFERENCE_ROTATION_SHA256:
        raise RuntimeError(f"Reference rotation checksum mismatch: {digest}")
    return rotation


def load_hidden_rotation(
    path: str | None,
    hidden_size: int,
) -> tuple[torch.Tensor, str]:
    if path:
        rotation = torch.load(Path(path), map_location="cpu", weights_only=True)
        source = str(Path(path).resolve())
    else:
        rotation = build_reference_hidden_rotation(hidden_size)
        source = "built-in qwen2.5-vl S600 reference Hadamard"

    rotation = rotation.detach().float().contiguous()
    if tuple(rotation.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"Rotation shape {tuple(rotation.shape)} does not match hidden size {hidden_size}"
        )
    return rotation, source


def _replace_parameter(parameter: torch.Tensor, value: torch.Tensor) -> None:
    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _fold_norm_into_inputs(norm, projections, rotation, device) -> None:
    gamma = norm.weight.detach().float().to(device)
    for projection in projections:
        weight = projection.weight.detach().float().to(device)
        transformed = (weight * gamma.unsqueeze(0)) @ rotation
        _replace_parameter(projection.weight, transformed)
    norm.weight.fill_(1.0)


def _rotate_projection_output(projection, rotation, device) -> None:
    weight = projection.weight.detach().float().to(device)
    _replace_parameter(projection.weight, rotation.T @ weight)
    if projection.bias is not None:
        bias = projection.bias.detach().float().to(device)
        _replace_parameter(projection.bias, bias @ rotation)


@torch.no_grad()
def rotate_language_to_hidden_domain(text_model, rotation, device="cuda:0") -> float:
    rotation = rotation.detach().float().to(device)
    hidden_size = text_model.config.hidden_size
    if tuple(rotation.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"Rotation shape {tuple(rotation.shape)} does not match hidden size {hidden_size}"
        )

    identity = torch.eye(hidden_size, device=device)
    orthogonal_error = (rotation.T @ rotation - identity).abs().max().item()
    if orthogonal_error >= 1e-5:
        raise ValueError(f"Rotation is not orthogonal enough: {orthogonal_error}")

    embedding = text_model.embed_tokens.weight.detach().float().to(device)
    _replace_parameter(text_model.embed_tokens.weight, embedding @ rotation)

    for layer in text_model.layers:
        attention = layer.self_attn
        _fold_norm_into_inputs(
            layer.input_layernorm,
            [attention.q_proj, attention.k_proj, attention.v_proj],
            rotation,
            device,
        )
        _rotate_projection_output(attention.o_proj, rotation, device)

        mlp = layer.mlp
        _fold_norm_into_inputs(
            layer.post_attention_layernorm,
            [mlp.gate_proj, mlp.up_proj],
            rotation,
            device,
        )
        _rotate_projection_output(mlp.down_proj, rotation, device)

    _fold_norm_into_inputs(text_model.norm, [text_model.lm_head], rotation, device)
    return orthogonal_error


@torch.no_grad()
def rotate_vision_output_to_hidden_domain(vision_model, rotation, device="cuda:0") -> None:
    rotation = rotation.detach().float().to(device)
    final_projection = vision_model.merger.mlp1[3]
    _rotate_projection_output(final_projection, rotation, device)
