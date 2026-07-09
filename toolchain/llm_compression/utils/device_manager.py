import torch

from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)


def get_module_device(module: torch.nn.Module) -> torch.device:
    """Get the device of an nn.Module from its first parameter or buffer."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")


class DeviceManager:
    """Memory-aware device manager for model placement.

    Responsibilities:
    - _assign: distribute model parts across GPUs (round-robin)
    - select_device: memory-aware device selection for multi-model scenarios
    """

    def __init__(self, model, model_list):
        self.model = model
        self.model_list = list(model_list)
        self._assign()

    def _assign(self):
        """Assign model parts to GPUs in round-robin order."""
        if not torch.cuda.is_available():
            return
        num_gpus = torch.cuda.device_count()
        for i, part in enumerate(self.model_list):
            submodule = getattr(self.model, part, None)
            if submodule is not None:
                submodule.to(device=torch.device("cuda", i % num_gpus))

    @staticmethod
    def select_device(module, num_copies: int = 1) -> int | None:
        """Check if GPU can hold num_copies of module simultaneously.

        Returns gpu_id for QuantAnalysis device_ids, or None for CPU fallback.
        For non-nn.Module (BC models), returns 0 since size can't be estimated.
        """
        if not torch.cuda.is_available():
            return None

        if not isinstance(module, torch.nn.Module):
            return 0

        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        required = int(param_bytes * num_copies * 1.5)

        best_gpu, best_free = 0, 0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_gpu, best_free = i, free

        if best_free >= required:
            return best_gpu

        logger.info(
            f"GPU memory insufficient for {num_copies} copies "
            f"(need {required / 2**30:.1f}GiB, free {best_free / 2**30:.1f}GiB), "
            f"falling back to CPU"
        )
        return None
