_model_builders = {}


def register_model(name, marches=None):
    def decorator(func):
        _model_builders[name] = {"builder": func, "marches": marches or []}
        return func

    return decorator


def get_supported_models():
    return list(_model_builders.keys())


def get_marches_with_model(model_name: str) -> list[str]:
    return _model_builders.get(model_name, {}).get("marches", [])


def get_supported_marches():
    return list(set(march for model_name in _model_builders for march in _model_builders[model_name]["marches"]))


def create_model_api(model_name, args):
    model_info = _model_builders.get(model_name)
    if not model_info:
        print(f"Model '{model_name}' is not supported yet.")
        return None

    supported_marches = get_supported_marches()
    if args.march not in supported_marches:
        print(f"March {args.march} is not supported for model {model_name}.")
        print(f"Supported marches are: {', '.join(supported_marches)}")
        return None

    builder = model_info["builder"]
    return builder(args)


@register_model("deepseek-qwen-1_5b", ["nash-e", "nash-m", "nash-p"])
def _build_deepseek_qwen_1_5b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    preserve_precision = False
    mask_value = -512
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        preserve_precision=preserve_precision,
        model_type="deepseek-qwen-1_5b",
        w_bits=args.w_bits,
        mask_value=mask_value,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
    )


@register_model("deepseek-qwen-7b", ["nash-e", "nash-m"])
def _build_deepseek_qwen_7b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    preserve_precision = True
    dtype = "float16"
    mask_value = -512
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        dtype=dtype,
        preserve_precision=preserve_precision,
        model_type="deepseek-qwen-7b",
        mask_value=mask_value,
    )


@register_model("qwen2_5-1_5b", ["nash-e", "nash-m"])
def _build_qwen2_5_1_5b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    preserve_precision = True
    mask_value = -32767
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        preserve_precision=preserve_precision,
        model_type="qwen2_5-1_5b",
        mask_value=mask_value,
    )


@register_model("qwen2_5-7b", ["nash-e", "nash-m"])
def _build_qwen2_5_7b(args):
    from leap_llm.apis.model.deepseek import DeepSeekApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    dtype = "float16"
    preserve_precision = True
    mask_value = -16384
    return DeepSeekApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        dtype=dtype,
        preserve_precision=preserve_precision,
        model_type="qwen2_5-7b",
        mask_value=mask_value,
    )


@register_model("internvl2-2b", ["nash-p"])
def _build_internvl2_2b(args):
    from leap_llm.apis.model.internvl_2b import Internvl2bApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Internvl2bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        vlm_model_type="internvl2-2b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
    )


@register_model("internvl2_5-2b", ["nash-p"])
def _build_internvl2_5_2b(args):
    from leap_llm.apis.model.internvl_2b import Internvl2bApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Internvl2bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        vlm_model_type="internvl2_5-2b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
    )


@register_model("internvl2-1b", ["nash-p"])
def _build_internvl2_1b(args):
    from leap_llm.apis.model.internvl_1b import Internvl1bApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Internvl1bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        vlm_model_type="internvl2-1b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
    )


@register_model("internvl2_5-1b", ["nash-p"])
def _build_internvl2_5_1b(args):
    from leap_llm.apis.model.internvl_1b import Internvl1bApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Internvl1bApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_image_path=args.calib_image_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        vlm_model_type="internvl2_5-1b",
        w_bits=args.w_bits,
        weight_scales_file=args.weight_scales_file,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        kept_tokens_file=args.kept_tokens_file,
        march=args.march,
    )


@register_model("internlm2-1_8b", ["nash-e", "nash-m"])
def _build_internlm2_18b(args):
    from leap_llm.apis.model.internlm2 import Internlm2Api

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Internlm2Api(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        dtype="float32",
        preserve_precision=False,
        model_type="internlm2-1_8b",
    )


@register_model("qwen2_5-omni-3b", ["nash-e", "nash-m"])
def _build_qwen2_5_omni_3b(args):
    from leap_llm.apis.model.qwen2_5_omni import Qwen2_5OmniApi

    if args.chunk_size != 256 or args.cache_len != 2048:
        print(
            f"Warning: {args.model_name} model only supports chunk_size=256 and "
            f"cache_len=2048."
            "Setting chunk_size to 256 and cache_len to 2048."
        )
        args.chunk_size = 256
        args.cache_len = 2048

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Qwen2_5OmniApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        dtype="float32",
        preserve_precision=True,
        model_type="qwen2_5_omni_3b",
    )


@register_model("qwen2_5-vl-3b", ["nash-p"])
def _build_qwen2_5_vl_3b(args):
    from leap_llm.apis.model.qwen2_5_vl import Qwen2_5VlApi

    # Convert device to list if it's a single device string (for backward compatibility)
    devices = args.device if isinstance(args.device, list) else [args.device]

    mask_value = -32768
    return Qwen2_5VlApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_tsv_path=args.calib_tsv_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        image_width=args.image_width,
        image_height=args.image_height,
        devices=devices,
        model_type="qwen2_5-vl-3b",
        dtype="float32",
        w_bits=args.w_bits,
        mask_value=mask_value,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        decode_seq_len=args.decode_seq_len,
        input_model_format=args.input_model_format,
        march=args.march,
    )


@register_model("qwen2_5-vl-7b", ["nash-p"])
def _build_qwen2_5_vl_7b(args):
    from leap_llm.apis.model.qwen2_5_vl import Qwen2_5VlApi

    # Convert device to list if it's a single device string (for backward compatibility)
    devices = args.device if isinstance(args.device, list) else [args.device]

    mask_value = -32768
    return Qwen2_5VlApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_tsv_path=args.calib_tsv_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        image_width=args.image_width,
        image_height=args.image_height,
        devices=devices,
        model_type="qwen2_5-vl-7b",
        dtype="float32",
        w_bits=args.w_bits,
        mask_value=mask_value,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        decode_seq_len=args.decode_seq_len,
        input_model_format=args.input_model_format,
        march=args.march,
    )


@register_model("locateanything-lm-3b", ["nash-p"])
def _build_locateanything_lm_3b(args):
    from leap_llm.apis.model.locateanything_language import (
        LocateAnythingLanguageApi,
    )

    device = args.device[0] if isinstance(args.device, list) else args.device

    return LocateAnythingLanguageApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        decode_seq_len=args.decode_seq_len,
        device=device,
        w_bits=args.w_bits,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
    )


@register_model("locateanything-vit-3b", ["nash-p"])
def _build_locateanything_vit_3b(args):
    from leap_llm.apis.model.locateanything_vision import (
        LocateAnythingVisionApi,
    )

    device = args.device[0] if isinstance(args.device, list) else args.device

    return LocateAnythingVisionApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        image_width=args.image_width,
        image_height=args.image_height,
        device=device,
        w_bits=args.w_bits,
        vit_core_num=args.vit_core_num,
        march=args.march,
    )


@register_model("locateanything-3b", ["nash-p"])
def _build_locateanything_3b(args):
    from leap_llm.apis.model.locateanything import LocateAnythingApi

    # Convert device to list if it's a single device string.
    devices = args.device if isinstance(args.device, list) else [args.device]

    return LocateAnythingApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_tsv_path=args.calib_tsv_path,
        calib_message_path=args.calib_json_path,
        calib_image_path=args.calib_image_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        image_width=args.image_width,
        image_height=args.image_height,
        decode_seq_len=args.decode_seq_len,
        block_size=args.decode_seq_len,   # PBD: decode_seq_len IS the block size
        causal_attn=False,                # PBD default
        devices=devices,
        model_type="locateanything-3b",
        dtype="float16",
        w_bits=args.w_bits,
        mask_value=-32768,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        input_model_format=args.input_model_format,
        march=args.march,
    )


@register_model("qwen3", ["nash-p"])
def _build_qwen3(args):
    from leap_llm.apis.model.qwen3 import Qwen3Api

    # For qwen3, support multi-GPU by passing devices list
    devices = args.device if isinstance(args.device, list) else [args.device]

    return Qwen3Api(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        devices=devices,
        model_type="qwen3",
        dtype="float32",
        w_bits=args.w_bits,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
        speculative_algorithm=getattr(args, "speculative_algorithm", None),
        speculative_draft_model_path=getattr(args, "speculative_draft_model_path", None),
        speculative_num_steps=getattr(args, "speculative_num_steps", 7),
        speculative_eagle_topk=getattr(args, "speculative_eagle_topk", 8),
        speculative_num_draft_tokens=getattr(args, "speculative_num_draft_tokens", 32),
        decode_seq_len=getattr(args, "decode_seq_len", None),
    )


@register_model("internvl3_5-1b", ["nash-p"])
def _build_internvl3_5(args):
    from leap_llm.apis.model.internvl3_5 import InternVL3_5Api

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return InternVL3_5Api(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        calib_image_path=args.calib_image_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        device=device,
        dtype="float32",
        w_bits=args.w_bits,
        model_type="internvl3_5-1b",
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        march=args.march,
    )


@register_model("pi0", ["nash-p"])
def _build_pi0(args):
    from leap_llm.apis.model.pi0 import Pi0Api

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Pi0Api(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        calib_image_path=args.calib_image_path,
        calib_action_data_path=args.calib_action_data_path,
        device=device,
        vision_tokens_num=args.vision_tokens_num,
        dtype="float16",
        model_type="pi0",
    )


@register_model("pi05", ["nash-p"])
def _build_pi05(args):
    from leap_llm.apis.model.pi05 import Pi05Api

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Pi05Api(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        calib_image_path=args.calib_image_path,
        calib_action_data_path=args.calib_action_data_path,
        device=device,
        vision_tokens_num=args.vision_tokens_num,
        action_horizon=args.action_horizon,
        dtype="float16",
        model_type="pi05",
    )


@register_model("smolvla", ["nash-p"])
def _build_smolvla(args):
    from leap_llm.apis.model.smolvla import SmolVLAApi

    device = args.device[0] if isinstance(args.device, list) else args.device

    return SmolVLAApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        calib_image_path=args.calib_image_path,
        calib_action_data_path=args.calib_action_data_path,
        policy_config_path=args.config_path,
        device=device,
        vision_tokens_num=args.vision_tokens_num,
        dtype="float16",
        model_type="smolvla",
    )


@register_model("spirit_v1_5", ["nash-p"])
def _build_spirit_v1_5(args):
    from leap_llm.apis.model.spirit_v1_5 import SpiritV1_5Api

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return SpiritV1_5Api(
        input_model_path=args.input_model_path,
        config_path=args.config_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        calib_image_path=args.calib_image_path,
        calib_action_data_path=args.calib_action_data_path,
        device=device,
        dtype="float16",
        model_type="spirit_v1_5",
        chunk_size=args.chunk_size,
    )


@register_model("whisper", ["nash-p"])
def _build_whisper(args):
    from leap_llm.apis.model.whisper import WhisperApi

    device = args.device[0] if isinstance(args.device, list) else args.device

    return WhisperApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_audio_path=args.calib_audio_path,
        device=device,
        dtype="float16",
        model_type="whisper",
    )


@register_model("qwen3-vl-2b", ["nash-p"])
def _build_qwen3_vl_2b(args):
    from leap_llm.apis.model.qwen3_vl import Qwen3VlApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Qwen3VlApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_tsv_path=args.calib_tsv_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
        model_type="qwen3-vl-2b",
        march=args.march,
        dtype="float32",
        w_bits=args.w_bits,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        input_model_format=args.input_model_format,
    )


@register_model("qwen3-vl-4b", ["nash-p"])
def _build_qwen3_vl_4b(args):
    from leap_llm.apis.model.qwen3_vl import Qwen3VlApi

    # For non-multi-GPU models, use first device from list
    device = args.device[0] if isinstance(args.device, list) else args.device

    return Qwen3VlApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_tsv_path=args.calib_tsv_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
        model_type="qwen3-vl-4b",
        march=args.march,
        dtype="float32",
        w_bits=args.w_bits,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        input_model_format=args.input_model_format,
    )


@register_model("qwen3-vl-8b", ["nash-p"])
def _build_qwen3_vl_8b(args):
    from leap_llm.apis.model.qwen3_vl import Qwen3VlApi

    # 8b needs multiple cuda devices
    device = args.device if isinstance(args.device, list) else args.device

    return Qwen3VlApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_tsv_path=args.calib_tsv_path,
        calib_message_path=args.calib_json_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
        model_type="qwen3-vl-8b",
        march=args.march,
        dtype="float32",
        w_bits=args.w_bits,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        input_model_format=args.input_model_format,
    )


@register_model("gemma-4-e2b-it", ["nash-p"])
def _build_gemma4_e2b(args):
    from leap_llm.apis.model.gemma4_e2b import Gemma4E2BApi

    device = args.device[0] if isinstance(args.device, list) else args.device

    return Gemma4E2BApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        calib_text_path=args.calib_text_path,
        chunk_size=args.chunk_size,
        cache_len=args.cache_len,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
        model_type="gemma-4-e2b-it",
        march=args.march,
        dtype="float32",
        w_bits=args.w_bits,
        vit_core_num=args.vit_core_num,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        input_model_format=args.input_model_format,
    )
