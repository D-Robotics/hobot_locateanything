"""SmolVLA leap_llm models (lazy imports to avoid loading torch at package import)."""

__all__ = ["SmolVLMVision", "SmolVLMPrefix", "SmolVLMActionExpert"]


def __getattr__(name):
    if name == "SmolVLMVision":
        from leap_llm.models.smolvla.model_vision import SmolVLMVision

        return SmolVLMVision
    if name == "SmolVLMPrefix":
        from leap_llm.models.smolvla.model_vlm import SmolVLMPrefix

        return SmolVLMPrefix
    if name == "SmolVLMActionExpert":
        from leap_llm.models.smolvla.model_action_expert import SmolVLMActionExpert

        return SmolVLMActionExpert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
