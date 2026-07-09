"""Json-stype message Dataset for calibration and evaluation."""

import json
import os
from pathlib import Path

from llm_compression.registry_factory import DATASET_REGISTRY
from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)


@DATASET_REGISTRY("vlm_json")
class VLMJsonDataset:
    """
    A generic VLM dataset backed by JSON files in message-based format.

    This dataset loads samples from a JSON file or a directory of JSON files.
    Each JSON file must contain a list of items, where each item follows the
    message schema commonly used by multimodal chat models.

    Expected JSON format:
        [
            {
                "message": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", ...},
                            {"type": "text", ...}
                        ]
                    }
                ]
            },
            ...
        ]

    Args:
        data_root (str):
            Path to a JSON file or a directory containing multiple JSON files.
        eval_savepath (str):
            Directory used to save inference results. Defaults to current directory.
        preprocess_fn (Callable):
            Callback to convert raw message into model-ready inputs.
            __getitem__ returns preprocess_fn(sample).
        max_steps (int, optional):
            If set, only the first max_steps samples are used. None means full dataset.
    """

    def __init__(self, **kwargs):
        self.data_root = kwargs.get("data_root")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")
        self.data = self.load_data()

    def build_prompt(self, line):
        pass

    def load_data(self):
        """
        Load JSON message data from a file or directory.
        Returns a list of message lists.
        """
        data_root = Path(self.data_root).resolve()
        json_files: list[Path]
        if data_root.is_dir():
            json_files = sorted(p for p in data_root.glob("*.json") if p.is_file())
            if not json_files:
                raise RuntimeError(f"No json files found in {data_root}")
        else:
            if data_root.suffix.lower() != ".json":
                raise ValueError("Message data path must point to a json file or directory!")
            json_files = [data_root]

        dataset = []
        for json_path in json_files:
            with open(json_path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Failed to parse JSON file {json_path}: {exc}") from exc

            if not isinstance(data, list):
                raise ValueError(f"Message data must be a list: {json_path}")

            for item in data:
                if not isinstance(item, dict) or "message" not in item:
                    raise ValueError(f"Invalid message format in {json_path}")
                messages = item["message"]
                if not isinstance(messages, list):
                    raise ValueError(f"Message must be a list: {json_path}")
                dataset.append(messages)
        return dataset

    def eval(self, predictions):
        """
        Save inference results to a JSON file.
        """
        if self.eval_savepath is None:
            self.eval_savepath = "./"
        os.makedirs(self.eval_savepath, exist_ok=True)
        result_file = os.path.join(self.eval_savepath, "infer_result.json")
        logger.info(f"The infer results will be saved to {result_file}")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

    def __len__(self):
        length = len(self.data)
        if self._max_steps is None:
            return length
        return min(self._max_steps, length)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return self.preprocess_fn(sample)
