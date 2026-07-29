"""PyTorch reference emulation for LocateAnything Vision quantization."""

from __future__ import annotations

import re
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


EMULATION_VERSION = 9
SCHEME = (
    "W8 per-output-channel weights + per-row dynamic A8 Linear inputs + "
    "per-row dynamic S8 QK/V + mean-centered dynamic S8 WV values"
)
LIMITATION = (
    "Ideal PyTorch QDQ with exact WV value centering; HBDK reciprocal-LUT and "
    "target accumulation rounding are not modeled"
)


def tensor_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "status": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    left = reference.detach().float()
    right = candidate.detach().float()
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        return {"status": "nonfinite", "shape": list(left.shape)}
    delta = right - left
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    tiny = torch.finfo(torch.float32).tiny
    top1 = None
    if left.ndim >= 2 and left.shape[-1] > 1:
        top1 = float((left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item())
    return {
        "status": "compared",
        "shape": list(left.shape),
        "cosine": float(
            (torch.dot(left_flat, right_flat) / torch.clamp(left_norm * right_norm, min=tiny)).item()
        ),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta.reshape(-1)) / torch.clamp(left_norm, min=tiny)).item()
        ),
        "mae": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs": float(delta.abs().max().item()),
        "top1_agreement": top1,
        "exact_equal": bool(torch.equal(left, right)),
        "reference_mean": float(left.mean().item()),
        "candidate_mean": float(right.mean().item()),
        "reference_std": float(left.std(unbiased=False).item()),
        "candidate_std": float(right.std(unbiased=False).item()),
    }


def dynamic_activation_qdq(value: Any) -> tuple[Any, Any, Any]:
    """Emulate blockSize=-1 as one symmetric A8 scale per last-dimension row."""
    import torch

    absmax = value.abs().amax(dim=-1, keepdim=True)
    scale = absmax * value.new_tensor(1.0 / 127.0) + value.new_tensor(2.0**-16)
    quantized = torch.round(value / scale).clamp(-127, 127)
    return quantized * scale, quantized, scale


def dynamic_nonnegative_u8_qdq(value: Any) -> tuple[Any, Any, Any]:
    """Use all 256 unsigned codes for one nonnegative last-dimension row."""
    import torch

    minimum = float(value.detach().min().item())
    if minimum < -1e-7:
        raise ValueError(
            f"nonnegative U8 quantization received minimum {minimum:.8g}"
        )
    maximum = value.amax(dim=-1, keepdim=True)
    scale = maximum * value.new_tensor(1.0 / 255.0) + value.new_tensor(2.0**-16)
    quantized = torch.round(value / scale).clamp(0, 255)
    return quantized * scale, quantized, scale


def centered_value_wv_qdq(attention: Any, value_transposed: Any) -> dict[str, Any]:
    """Emulate dynamic S8 WV after removing each V channel's token mean."""
    import torch

    value_mean = value_transposed.mean(dim=-1, keepdim=True)
    centered_value = value_transposed - value_mean
    quant_attention, attention_int, attention_scale = dynamic_activation_qdq(attention)
    quant_value, value_int, value_scale = dynamic_activation_qdq(centered_value)
    main = torch.matmul(quant_attention, quant_value.transpose(-1, -2))
    attention_sum = attention.sum(dim=-1, keepdim=True, dtype=attention.dtype)
    mean_term = attention_sum * value_mean.transpose(-1, -2)
    return {
        "output": main + mean_term,
        "value_mean": value_mean,
        "centered_value": centered_value,
        "quant_attention": quant_attention,
        "attention_int": attention_int,
        "attention_scale": attention_scale,
        "quant_value": quant_value,
        "value_int": value_int,
        "value_scale": value_scale,
        "attention_sum": attention_sum,
        "mean_term": mean_term,
    }


def weight_qdq(weight: Any, bits: int = 8, scales: Any | None = None) -> tuple[Any, Any, Any]:
    import torch

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    if scales is None:
        scale = weight.abs().amax(dim=-1, keepdim=True) / qmax
    else:
        scale = scales.to(device=weight.device, dtype=weight.dtype)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(weight / scale).clamp(qmin, qmax)
    return quantized * scale, quantized, scale


def static_activation_qdq(value: Any, absmax: float, bits: int) -> tuple[Any, Any, Any]:
    import torch

    qmax = 2 ** (bits - 1) - 1
    scale = value.new_tensor(absmax / qmax)
    quantized = torch.round(value / scale).clamp(-qmax, qmax)
    return quantized * scale, quantized, scale


def is_attention_quantizer(module_name: str) -> bool:
    return ".qk_matmul." in module_name or ".wv_matmul." in module_name


def activation_qdq(
    value: Any,
    module_name: str,
    absmax: float,
    bits: int,
) -> tuple[Any, Any, Any, str]:
    """Use the proposed dynamic A8 path only for MoonViT QK/WV matmuls."""
    if is_attention_quantizer(module_name):
        dequantized, quantized, scale = dynamic_activation_qdq(value)
        return dequantized, quantized, scale, "dynamic_attention_quantizer"
    dequantized, quantized, scale = static_activation_qdq(value, absmax, bits)
    return dequantized, quantized, scale, "static_quantizer"


def quantization_statistics(
    reference: Any,
    candidate: Any,
    quantized: Any,
    qmax: int,
) -> dict[str, Any]:
    comparison = tensor_comparison(reference, candidate)
    comparison.update(
        {
            "saturation_rate": float((quantized.abs() >= qmax).float().mean().item()),
            "reference_zero_rate": float((reference == 0).float().mean().item()),
            "candidate_zero_rate": float((candidate == 0).float().mean().item()),
        }
    )
    return comparison


@dataclass(frozen=True)
class FloatRescueRule:
    """Select one quantized operation for a controlled Float intervention."""

    module_pattern: str
    stages: tuple[str, ...] = ("*",)
    kinds: tuple[str, ...] = ("linear", "matmul", "static")
    mode: str = "float"

    def __post_init__(self) -> None:
        if not self.module_pattern:
            raise ValueError("Float rescue module_pattern must not be empty")
        re.compile(self.module_pattern)
        if not self.stages:
            raise ValueError("Float rescue stages must not be empty")
        if not self.kinds:
            raise ValueError("Float rescue kinds must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_pattern": self.module_pattern,
            "stages": list(self.stages),
            "kinds": list(self.kinds),
            "mode": self.mode,
        }


class FloatRescuePolicy:
    """Resolve exact, auditable Float interventions without replacing modules."""

    MODES = {
        "linear": {"quantized", "float", "float_weight", "float_activation"},
        "matmul": {"quantized", "float"},
        "fake_matmul": {"quantized", "float", "centered_value_s8"},
        "static": {"quantized", "float", "nonnegative_u8"},
    }

    def __init__(self, rules: list[FloatRescueRule], *, name: str) -> None:
        if not name.strip():
            raise ValueError("Float rescue policy name must not be empty")
        self.name = name.strip()
        self.rules = list(rules)
        self._patterns = [re.compile(rule.module_pattern) for rule in self.rules]
        self.inventory_matches: list[dict[str, Any]] = []
        self.runtime_matches: dict[str, int] = {}
        for rule in self.rules:
            unknown_kinds = sorted(set(rule.kinds) - set(self.MODES))
            if unknown_kinds:
                raise ValueError(f"unsupported Float rescue kinds: {unknown_kinds}")
            invalid = sorted(
                kind for kind in rule.kinds if rule.mode not in self.MODES[kind]
            )
            if invalid:
                raise ValueError(
                    f"Float rescue mode {rule.mode!r} is invalid for {invalid}"
                )

    def bind(self, inventory: list[tuple[str, str]]) -> None:
        self.inventory_matches = []
        self.runtime_matches = {}
        for index, (rule, pattern) in enumerate(zip(self.rules, self._patterns)):
            matches = [
                {"module": module_name, "kind": kind}
                for module_name, kind in inventory
                if kind in rule.kinds and pattern.fullmatch(module_name)
            ]
            if not matches:
                raise ValueError(
                    f"Float rescue rule {index} matched no quantized modules: "
                    f"{rule.module_pattern}"
                )
            self.inventory_matches.append({
                "rule": index,
                "mode": rule.mode,
                "matches": matches,
            })

    def resolve(self, stage: str, module_name: str, kind: str) -> str:
        matches = [
            (index, rule)
            for index, (rule, pattern) in enumerate(zip(self.rules, self._patterns))
            if ("*" in rule.stages or stage in rule.stages)
            and kind in rule.kinds
            and pattern.fullmatch(module_name)
        ]
        if not matches:
            return "quantized"
        modes = {rule.mode for _index, rule in matches}
        if len(modes) != 1:
            indices = [index for index, _rule in matches]
            raise RuntimeError(
                f"conflicting Float rescue rules {indices} for {stage}/{module_name}/{kind}"
            )
        mode = modes.pop()
        key = f"{stage}/{module_name}/{kind}/{mode}"
        self.runtime_matches[key] = self.runtime_matches.get(key, 0) + 1
        return mode

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rules": [rule.as_dict() for rule in self.rules],
            "inventory_matches": self.inventory_matches,
            "runtime_matches": dict(sorted(self.runtime_matches.items())),
        }


class QuantizationEmulator:
    """Patch Leap eager forwards to execute the intended quantized arithmetic."""

    def __init__(
        self,
        model: Any,
        capture_operators: bool,
        *,
        dynamic_attention_quantizers: bool = True,
        linear_weight_bits: Callable[[str, int], int] | None = None,
        rescue_policy: FloatRescuePolicy | None = None,
    ) -> None:
        from leap_llm.nn.modules import (
            ConstFakeQuant,
            DynamicQuantLinear,
            DynamicQuantMatmul,
            FakeQuantMatmul,
        )

        self.enabled = False
        self.stage = "unassigned"
        self.capture_operators = capture_operators
        self.dynamic_attention_quantizers = dynamic_attention_quantizers
        self.linear_weight_bits = linear_weight_bits or (lambda _name, bits: bits)
        self.rows: list[dict[str, Any]] = []
        self.original_forwards: dict[Any, Any] = {}
        self.weight_cache: dict[tuple[Any, int], Any] = {}
        self.weight_rows: list[dict[str, Any]] = []
        self.module_inventory: list[tuple[str, str]] = []
        self.static_modules: dict[str, Any] = {}
        self.dynamic_quantizer_patterns: list[str] = []
        self.rescue_policy: FloatRescuePolicy | None = None
        names = {module: name for name, module in model.named_modules()}
        for module, name in names.items():
            if isinstance(module, DynamicQuantLinear):
                self.module_inventory.append((name, "linear"))
                self.original_forwards[module] = module.forward
                module._quant_emulator = self
                module._quant_name = name
                module.forward = types.MethodType(self._linear_forward, module)
            elif isinstance(module, DynamicQuantMatmul):
                self.module_inventory.append((name, "matmul"))
                self.original_forwards[module] = module.forward
                module._quant_emulator = self
                module._quant_name = name
                module.forward = types.MethodType(self._matmul_forward, module)
            elif isinstance(module, FakeQuantMatmul):
                self.module_inventory.append((name, "fake_matmul"))
                self.original_forwards[module] = module.forward
                module._quant_emulator = self
                module._quant_name = name
                module.forward = types.MethodType(self._fake_matmul_forward, module)
            elif isinstance(module, ConstFakeQuant) and float(module.absmax.item()) > 0:
                self.module_inventory.append((name, "static"))
                self.static_modules[name] = module
                self.original_forwards[module] = module.forward
                module._quant_emulator = self
                module._quant_name = name
                module._quant_absmax = float(module.absmax.item())
                module._quant_dynamic_attention = (
                    dynamic_attention_quantizers and is_attention_quantizer(name)
                )
                module.forward = types.MethodType(self._static_forward, module)
        try:
            self.set_rescue_policy(rescue_policy)
        except Exception:
            for module, forward in self.original_forwards.items():
                module.forward = forward
                for attribute in (
                    "_quant_emulator", "_quant_name", "_quant_absmax",
                    "_quant_dynamic_attention",
                ):
                    if hasattr(module, attribute):
                        delattr(module, attribute)
            self.original_forwards.clear()
            self.module_inventory.clear()
            self.static_modules.clear()
            raise

    def _rescue_mode(self, module: Any, kind: str) -> str:
        if not self.enabled:
            return "float"
        if self.rescue_policy is None:
            return "quantized"
        return self.rescue_policy.resolve(
            self.stage, str(module._quant_name), kind
        )

    @staticmethod
    def _linear_forward(module: Any, value: Any) -> Any:
        import torch.nn.functional as functional

        emulator: QuantizationEmulator = module._quant_emulator
        weight = module.weight.data
        bias = module.bias.data if module.bias is not None else None
        rescue_mode = emulator._rescue_mode(module, "linear")
        if rescue_mode == "float":
            return functional.linear(value, weight, bias)

        if rescue_mode == "float_activation":
            quant_value = value
            value_int = value_scale = None
        else:
            quant_value, value_int, value_scale = dynamic_activation_qdq(value)
        weight_bits = emulator.linear_weight_bits(
            str(module._quant_name), int(module.w_bits)
        )
        weight_key = (module, weight_bits)
        if rescue_mode == "float_weight":
            quant_weight = weight
        elif weight_key not in emulator.weight_cache:
            configured_scales = (
                module.scales if module.has_scale and weight_bits == int(module.w_bits)
                else None
            )
            quant_weight, weight_int, weight_scale = weight_qdq(
                weight, weight_bits, configured_scales
            )
            emulator.weight_cache[weight_key] = quant_weight
            emulator.weight_rows.append(
                {
                    "module": module._quant_name,
                    "bits": weight_bits,
                    "comparison": tensor_comparison(weight, quant_weight),
                    "scale_min": float(weight_scale.min().item()),
                    "scale_max": float(weight_scale.max().item()),
                    "saturation_rate": float(
                        (weight_int.abs() >= 2 ** (weight_bits - 1) - 1)
                        .float()
                        .mean()
                        .item()
                    ),
                }
            )
        else:
            quant_weight = emulator.weight_cache[weight_key]
        quant_output = functional.linear(quant_value, quant_weight, bias)

        if emulator.capture_operators:
            float_output = functional.linear(value, weight, bias)
            row = {
                "module": module._quant_name,
                "kind": "dynamic_linear",
                "rescue_mode": rescue_mode,
                "comparison": tensor_comparison(float_output, quant_output),
            }
            if value_int is not None and value_scale is not None:
                row["activation"] = {
                        **quantization_statistics(value, quant_value, value_int, 127),
                        "scale_min": float(value_scale.min().item()),
                        "scale_max": float(value_scale.max().item()),
                        "scale_mean": float(value_scale.float().mean().item()),
                }
            emulator.rows.append(row)
        return quant_output

    @staticmethod
    def _matmul_forward(module: Any, left: Any, right: Any) -> Any:
        import torch

        emulator: QuantizationEmulator = module._quant_emulator
        transpose_rhs = bool(getattr(module, "transpose_rhs", False))

        def matmul(lhs: Any, rhs: Any) -> Any:
            if transpose_rhs:
                rhs = rhs.transpose(-1, -2)
            return torch.matmul(lhs, rhs)

        rescue_mode = emulator._rescue_mode(module, "matmul")
        if rescue_mode == "float":
            return matmul(left, right)

        centered_value_wv = module._quant_name.endswith(".wv_matmul")
        extra: dict[str, Any] = {}
        if centered_value_wv:
            if not transpose_rhs:
                raise RuntimeError("centered-value WV requires the transposed-RHS contract")
            centered = centered_value_wv_qdq(left, right)
            quant_left = centered["quant_attention"]
            left_int = centered["attention_int"]
            left_scale = centered["attention_scale"]
            quant_right = centered["quant_value"]
            right_int = centered["value_int"]
            right_scale = centered["value_scale"]
            quant_output = centered["output"]
            left_reference = left
            right_reference = centered["centered_value"]
            left_scheme = "per_row_s8_symmetric"
            extra = {
                "source_attention": {
                    "shape": list(left.shape),
                    "minimum": float(left.min().item()),
                    "maximum": float(left.max().item()),
                    "mean": float(left.float().mean().item()),
                    "zero_rate": float((left == 0).float().mean().item()),
                },
                "value_mean": {
                    "absmax": float(centered["value_mean"].abs().max().item()),
                    "mean": float(centered["value_mean"].float().mean().item()),
                },
            }
        else:
            quant_left, left_int, left_scale = dynamic_activation_qdq(left)
            quant_right, right_int, right_scale = dynamic_activation_qdq(right)
            quant_output = matmul(quant_left, quant_right)
            left_reference = left
            right_reference = right
            left_scheme = "per_row_s8_symmetric"
        if emulator.capture_operators:
            float_output = matmul(left, right)
            row = {
                "module": module._quant_name,
                "kind": "dynamic_attention_matmul",
                "rescue_mode": rescue_mode,
                "transpose_rhs": transpose_rhs,
                "left_scheme": left_scheme,
                "right_scheme": "per_row_s8_symmetric",
                "comparison": tensor_comparison(float_output, quant_output),
                "left_activation": {
                    **quantization_statistics(
                        left_reference, quant_left, left_int, 127
                    ),
                    "scale_min": float(left_scale.min().item()),
                    "scale_max": float(left_scale.max().item()),
                    "scale_mean": float(left_scale.float().mean().item()),
                },
                "right_activation": {
                    **quantization_statistics(
                        right_reference, quant_right, right_int, 127
                    ),
                    "scale_min": float(right_scale.min().item()),
                    "scale_max": float(right_scale.max().item()),
                    "scale_mean": float(right_scale.float().mean().item()),
                },
            }
            row.update(extra)
            emulator.rows.append(row)
        return quant_output

    @staticmethod
    def _fake_matmul_forward(module: Any, left: Any, right: Any) -> Any:
        import torch

        emulator: QuantizationEmulator = module._quant_emulator
        rescue_mode = emulator._rescue_mode(module, "fake_matmul")
        if rescue_mode == "quantized":
            return emulator.original_forwards[module](left, right)
        float_output = torch.matmul(left, right)
        if rescue_mode == "float":
            return float_output
        if rescue_mode != "centered_value_s8":
            raise RuntimeError(f"unsupported fake matmul mode: {rescue_mode}")
        if not str(module._quant_name).endswith(".wv_matmul"):
            raise RuntimeError("centered_value_s8 is valid only for WV matmul")
        centered = centered_value_wv_qdq(left, right.transpose(-1, -2))
        quant_output = centered["output"]
        if emulator.capture_operators:
            emulator.rows.append(
                {
                    "module": module._quant_name,
                    "kind": "centered_value_s8_matmul",
                    "rescue_mode": rescue_mode,
                    "comparison": tensor_comparison(float_output, quant_output),
                    "attention": quantization_statistics(
                        left,
                        centered["quant_attention"],
                        centered["attention_int"],
                        127,
                    ),
                    "centered_value": quantization_statistics(
                        centered["centered_value"],
                        centered["quant_value"],
                        centered["value_int"],
                        127,
                    ),
                    "value_mean_absmax": float(
                        centered["value_mean"].abs().max().item()
                    ),
                }
            )
        return quant_output

    @staticmethod
    def _static_forward(module: Any, value: Any) -> Any:
        emulator: QuantizationEmulator = module._quant_emulator
        rescue_mode = emulator._rescue_mode(module, "static")
        if rescue_mode == "float":
            return value
        if rescue_mode == "nonnegative_u8":
            quant_value, value_int, scale = dynamic_nonnegative_u8_qdq(value)
            kind = "dynamic_nonnegative_u8_quantizer"
            qmax = 255
        elif module._quant_dynamic_attention:
            quant_value, value_int, scale = dynamic_activation_qdq(value)
            kind = "dynamic_attention_quantizer"
            qmax = 127
        else:
            quant_value, value_int, scale = static_activation_qdq(
                value, module._quant_absmax, int(module.bits)
            )
            kind = "static_quantizer"
            qmax = 2 ** (int(module.bits) - 1) - 1
        if emulator.capture_operators:
            details = {
                "module": module._quant_name,
                "kind": kind,
                "rescue_mode": rescue_mode,
                "comparison": quantization_statistics(
                    value,
                    quant_value,
                    value_int,
                    qmax,
                ),
                "bits": int(module.bits),
            }
            if kind in {
                "dynamic_attention_quantizer",
                "dynamic_nonnegative_u8_quantizer",
            }:
                details.update(
                    calibration_absmax=module._quant_absmax,
                    scale_min=float(scale.min().item()),
                    scale_max=float(scale.max().item()),
                    scale_mean=float(scale.float().mean().item()),
                )
                if kind == "dynamic_nonnegative_u8_quantizer":
                    details.update(signed=False, qmin=0, qmax=255)
            else:
                details.update(absmax=module._quant_absmax, scale=float(scale.item()))
            emulator.rows.append(details)
        return quant_value

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.rows = []

    def set_stage(self, stage: str) -> None:
        if not stage:
            raise ValueError("quantization stage must not be empty")
        self.stage = stage

    def set_rescue_policy(self, rescue_policy: FloatRescuePolicy | None) -> None:
        if rescue_policy is not None:
            rescue_policy.bind(self.module_inventory)
        self.rescue_policy = rescue_policy

    def set_dynamic_quantizer_patterns(self, patterns: list[str]) -> dict[str, Any]:
        compiled = [re.compile(pattern) for pattern in patterns]
        matches: dict[str, list[str]] = {}
        for pattern, expression in zip(patterns, compiled):
            matched = [
                name for name in self.static_modules if expression.fullmatch(name)
            ]
            if not matched:
                raise ValueError(
                    f"dynamic quantizer pattern matched no static modules: {pattern}"
                )
            matches[pattern] = matched
        for name, module in self.static_modules.items():
            module._quant_dynamic_attention = any(
                expression.fullmatch(name) for expression in compiled
            )
        self.dynamic_quantizer_patterns = list(patterns)
        return {"patterns": list(patterns), "matches": matches}

    def close(self) -> None:
        for module, forward in self.original_forwards.items():
            module.forward = forward
            for attribute in (
                "_quant_emulator", "_quant_name", "_quant_absmax",
                "_quant_dynamic_attention",
            ):
                if hasattr(module, attribute):
                    delattr(module, attribute)
        self.original_forwards.clear()
        self.weight_cache.clear()
        self.module_inventory.clear()
        self.static_modules.clear()
        self.dynamic_quantizer_patterns.clear()
        self.rescue_policy = None


class BoundaryCapture:
    """Compare cumulative float and quantized outputs at stable model boundaries."""

    def __init__(self, model: Any, enabled: bool) -> None:
        self.enabled = enabled
        self.phase = "off"
        self.references: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        if not enabled:
            return
        modules: list[tuple[str, Any]] = [("patch_embed", model.patch_embed)]
        modules.extend((f"blocks.{index}", block) for index, block in enumerate(model.blocks))
        modules.extend(
            [
                ("final_layernorm", model.final_layernorm),
                ("merger", model.merger),
            ]
        )
        for name, module in modules:
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def callback(_module: Any, _inputs: Any, output: Any) -> None:
            if self.phase == "reference":
                self.references[name] = output.detach().clone()
            elif self.phase == "candidate":
                if name not in self.references:
                    raise RuntimeError(f"missing float boundary {name}")
                self.rows.append(
                    {
                        "module": name,
                        "kind": "boundary",
                        "comparison": tensor_comparison(self.references[name], output),
                    }
                )

        return callback

    def begin_reference(self) -> None:
        self.references.clear()
        self.rows = []
        self.phase = "reference" if self.enabled else "off"

    def begin_candidate(self) -> None:
        self.rows = []
        self.phase = "candidate" if self.enabled else "off"

    def finish_sample(self) -> None:
        self.phase = "off"
        self.references.clear()
        self.rows = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.references.clear()
