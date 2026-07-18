import torch


def _replace_parameter(parameter, value):
    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _fold_norm_into_inputs(norm, projections, rotation, device):
    gamma = norm.weight.detach().float().to(device)
    for projection in projections:
        weight = projection.weight.detach().float().to(device)
        transformed = (weight * gamma.unsqueeze(0)) @ rotation
        _replace_parameter(projection.weight, transformed)
    norm.weight.fill_(1.0)


def _rotate_projection_output(projection, rotation, device):
    weight = projection.weight.detach().float().to(device)
    _replace_parameter(projection.weight, rotation.T @ weight)
    if projection.bias is not None:
        bias = projection.bias.detach().float().to(device)
        _replace_parameter(projection.bias, bias @ rotation)


@torch.no_grad()
def rotate_qwen25_language_to_official_domain(text_model, rotation, device="cuda:0"):
    rotation = rotation.detach().float().to(device)
    hidden_size = text_model.config.hidden_size
    if tuple(rotation.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"rotation shape {tuple(rotation.shape)} does not match hidden size {hidden_size}"
        )

    identity = torch.eye(hidden_size, device=device)
    orthogonal_error = (rotation.T @ rotation - identity).abs().max().item()
    if orthogonal_error >= 1e-5:
        raise ValueError(f"rotation is not orthogonal enough: {orthogonal_error}")

    embedding_weight = text_model.embed_tokens.weight.detach().float().to(device)
    _replace_parameter(text_model.embed_tokens.weight, embedding_weight @ rotation)

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

    _fold_norm_into_inputs(
        text_model.norm,
        [text_model.lm_head],
        rotation,
        device,
    )

    return orthogonal_error


@torch.no_grad()
def rotate_vision_output_to_official_domain(vision_model, rotation, device="cuda:0"):
    rotation = rotation.detach().float().to(device)
    _rotate_projection_output(vision_model.merger.mlp.proj1, rotation, device)

