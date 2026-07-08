# Copyright (c) Horizon Robotics. All rights reserved.

_TRACE_ALL_BRANCHES = False


class trace_all_branches:
    """Context manager to force all conditional branches to be executed
    during JIT trace, ensuring every operator is recorded in the trace graph.

    Background:
        JIT trace (used by horizon.quantization.prepare with JIT_STRIP method)
        can only follow one execution path through if-else blocks. For modules
        with prefill/decode branches (e.g. linear attention with conv_state),
        the "lm" shared mode traces with seq_len>1 (prefill path only), leaving
        decode-specific ops (like torch.cat) untraced. Untraced ops fail at
        runtime when inputs become QTensors.

    Usage:
        In calib_converter.py, wrap prepare() with this context manager:

            with trace_all_branches():
                model = horizon.quantization.prepare(model, ...)

        In module forward(), check is_tracing_all_branches() to execute
        all branches sequentially instead of selecting one via if-else.
    """

    def __enter__(self):
        global _TRACE_ALL_BRANCHES
        _TRACE_ALL_BRANCHES = True
        return self

    def __exit__(self, *args):
        global _TRACE_ALL_BRANCHES
        _TRACE_ALL_BRANCHES = False


def is_tracing_all_branches():
    return _TRACE_ALL_BRANCHES
