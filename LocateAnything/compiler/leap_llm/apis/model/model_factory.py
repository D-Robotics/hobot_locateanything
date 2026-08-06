"""Model registry for LocateAnything Vision and Language compilation."""

_model_builders = {}


def register_model(name, marches=None):
    def decorator(func):
        _model_builders[name] = {"builder": func, "marches": marches or []}
        return func

    return decorator


def get_supported_models():
    return list(_model_builders)


def get_marches_with_model(model_name: str) -> list[str]:
    return _model_builders.get(model_name, {}).get("marches", [])


def get_supported_marches():
    return sorted(
        {
            march
            for model_info in _model_builders.values()
            for march in model_info["marches"]
        }
    )

def create_model_api(model_name, args):
    model_info = _model_builders.get(model_name)
    if model_info is None:
        print(f"Model '{model_name}' is not supported.")
        return None

    if args.march not in model_info["marches"]:
        supported = ", ".join(model_info["marches"])
        print(f"March {args.march} is not supported for {model_name}: {supported}")
        return None

    return model_info["builder"](args)


def _primary_device(args):
    return args.device[0] if isinstance(args.device, list) else args.device


@register_model("locateanything-lm-3b", ["nash-p"])
def _build_locateanything_lm_3b(args):
    from leap_llm.apis.model.locateanything_language import LocateAnythingLanguageApi

    return LocateAnythingLanguageApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        chunk_size=args.chunk_size,
        batch_size=1,
        cache_len=args.cache_len,
        decode_seq_len=args.decode_seq_len,
        device=_primary_device(args),
        w_bits=args.w_bits,
        lm_head_w_bits=args.lm_head_w_bits,
        prefill_core_num=args.prefill_core_num,
        decode_core_num=args.decode_core_num,
        ar_core_num=args.ar_core_num,
        march=args.march,
        hidden_rotation_path=args.hidden_rotation_path,
        apply_hidden_rotation=not args.disable_hidden_rotation,
        export_only=args.export_only,
        calibration_scale_manifest=args.calibration_scale_manifest,
        sampling_backend=args.sampling_backend,
        sampling_temperature=args.sampling_temperature,
        sampling_top_p=args.sampling_top_p,
        sampling_repetition_penalty=args.sampling_repetition_penalty,
    )


@register_model("locateanything-vit-3b", ["nash-p"])
def _build_locateanything_vit_3b(args):
    from leap_llm.apis.model.locateanything_vision import LocateAnythingVisionApi

    return LocateAnythingVisionApi(
        input_model_path=args.input_model_path,
        output_model_path=args.output_model_path,
        image_width=args.image_width,
        image_height=args.image_height,
        device=_primary_device(args),
        w_bits=args.w_bits,
        vit_core_num=args.vit_core_num,
        march=args.march,
        hidden_rotation_path=args.hidden_rotation_path,
        apply_hidden_rotation=not args.disable_hidden_rotation,
        export_only=args.export_only,
        calibration_scale_manifest=args.calibration_scale_manifest,
    )
