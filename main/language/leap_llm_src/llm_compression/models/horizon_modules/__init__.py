from importlib import import_module
from importlib.util import find_spec

_OPTIONAL_EXPORTS = {
    "HzFlashAttention": "flash_attention",
}

__all__ = [
    symbol_name for symbol_name, module_name in _OPTIONAL_EXPORTS.items() if find_spec(f"{__name__}.{module_name}")
]


def __getattr__(name: str):
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _OPTIONAL_EXPORTS[name]
    if find_spec(f"{__name__}.{module_name}") is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    try:
        symbol = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    globals()[name] = symbol
    return symbol


def __dir__():
    return sorted(set(globals()) | set(_OPTIONAL_EXPORTS))
