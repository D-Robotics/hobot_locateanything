from llm_compression.utils.logger import get_logger


logger = get_logger(__name__)


def _is_flash_attention_available() -> bool:
    try:
        from llm_compression.models.horizon_modules import HzFlashAttention  # noqa: F401
    except (ImportError, AttributeError):
        return False

    return True


class AttentionManager:
    _atten_type = "eager"
    _flash_block_size = 1024

    @classmethod
    def set(cls, config) -> None:
        flash_attention_config = getattr(config, "flash_attention", None)
        if flash_attention_config is None:
            cls._atten_type = "eager"
            return

        enable = getattr(flash_attention_config, "enable", None)
        if enable:
            if _is_flash_attention_available():
                cls._atten_type = "flash"
                block_size = getattr(flash_attention_config, "block_size", None)
                if block_size is not None:
                    cls._flash_block_size = int(block_size)
                else:
                    raise ValueError("Flash attention is enabled but no block_size is specified.")
                logger.info(f"Flash attention enabled with block size {cls._flash_block_size}.")
            else:
                logger.warning("Flash attention requested but unavailable, fallback to eager attention.")
                cls._atten_type = "eager"
        else:
            cls._atten_type = "eager"

    @classmethod
    def is_flash_attn(cls) -> bool:
        return cls._atten_type == "flash"

    @classmethod
    def get_flash_block_size(cls) -> int:
        return cls._flash_block_size


__all__ = [
    "AttentionManager",
]
