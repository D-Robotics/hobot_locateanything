import logging

from .ppl_dataset import PPLDataset  # noqa: F401

logger = logging.getLogger(__name__)

try:
    from .bfcl_dataset import BFCLDataset  # noqa: F401
except ImportError:
    logger.warning("'bfcl_eval' not installed, BFCL evaluation unavailable. Install: pip install bfcl-eval")

try:
    from .mmlu import MMLUDataset  # noqa: F401
except ImportError:
    logger.warning("'opencompass' not installed, MMLU evaluation unavailable. Install: pip install opencompass")

try:
    from .longbench import LongBenchDataset  # noqa: F401
except ImportError:
    logger.warning("LongBench v1 dependencies missing. Install: pip install rouge jieba fuzzywuzzy python-Levenshtein")

try:
    from .longbench_v2 import LongBenchV2Dataset  # noqa: F401
except ImportError:
    logger.warning("LongBench v2 import failed")

try:
    from .mrcr import MRCRDataset  # noqa: F401
except ImportError:
    logger.warning("MRCR import failed")
