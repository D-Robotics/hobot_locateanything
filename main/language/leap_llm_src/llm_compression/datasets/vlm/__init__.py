import logging

from .vlm_json import VLMJsonDataset  # noqa: F401

logger = logging.getLogger(__name__)

_VLMEVAL_INSTALL_HINT = (
    "'vlmeval' not installed, %s evaluation unavailable. "
    "Install from source: pip install -e . (https://github.com/open-compass/VLMEvalKit)"
)

try:
    from .mmbench import MMBenchDataset  # noqa: F401
except ImportError:
    logger.warning(_VLMEVAL_INSTALL_HINT, "MMBench")

try:
    from .mmstar import MMStarDataset  # noqa: F401
except ImportError:
    logger.warning(_VLMEVAL_INSTALL_HINT, "MMStar")
