"""MRCR (Multi-Round Coreference Resolution) Dataset for long-context retrieval evaluation.

Tests the model's ability to retrieve a specific response from a long multi-turn
synthetic conversation containing multiple similar "needle" responses.

Data source: openai/mrcr on HuggingFace or local parquet/JSONL files.
Reference: https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2

Scoring: aligns with Google official metric (run_evaluation.py mrcr_v2_metric):
  1. Extract 12-char hash from target beginning; find its *last* occurrence in prediction (rfind).
  2. SequenceMatcher ratio between content-after-hash in prediction vs target.
  3. Final score = mean of per-sample ratios (continuous 0~1), NOT accuracy.
  max_tokens must be >= 1024 to avoid truncating answers (~900 tokens max for 8-needle 128k).
"""

import difflib
import json
import logging
import os

from llm_compression.registry_factory import DATASET_REGISTRY

logger = logging.getLogger(__name__)

_HASH_LEN = 12  # official: first 12 chars of answer as hash


def _mrcr_score(prediction: str, answer: str) -> float:
    """Compute per-sample MRCR score, matching Google official mrcr_v2_metric.

    Returns ratio in [0.0, 1.0].
    """
    if not prediction or len(answer) < _HASH_LEN:
        return 0.0
    hash_str = answer[:_HASH_LEN]
    idx = prediction.rfind(hash_str)
    if idx == -1:
        return 0.0
    pred_content = prediction[idx + _HASH_LEN :].strip()
    target_content = answer[_HASH_LEN:].strip()
    return difflib.SequenceMatcher(None, target_content, pred_content).ratio()


@DATASET_REGISTRY("mrcr")
class MRCRDataset:
    def __init__(self, **kwargs):
        self.data_root = kwargs.get("data_root")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")
        self.n_needles = kwargs.get("n_needles", 8)
        self.data = self._load_data()

    def _load_data(self):
        if self.data_root:
            data = self._load_local()
            if data is not None:
                return self._filter(data)

        from datasets import load_dataset

        logger.info(f"Loading openai/mrcr from HuggingFace (n_needles={self.n_needles})...")
        ds = load_dataset("openai/mrcr", split="train")
        data = [dict(row) for row in ds]
        return self._filter(data)

    def _load_local(self):
        if os.path.isfile(self.data_root):
            return self._load_file(self.data_root)
        if os.path.isdir(self.data_root):
            for fname in os.listdir(self.data_root):
                if fname.endswith(".parquet") or fname.endswith(".jsonl"):
                    path = os.path.join(self.data_root, fname)
                    logger.info(f"Loading MRCR from {path}")
                    return self._load_file(path)
        return None

    def _load_file(self, path):
        if path.endswith(".parquet"):
            import pandas as pd

            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        # jsonl
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _filter(self, data):
        filtered = [d for d in data if d.get("n_needles") == self.n_needles]
        logger.info(f"MRCR: {len(filtered)} samples after filter (n_needles={self.n_needles})")
        return filtered
        return filtered

    def __len__(self):
        return min(self._max_steps, len(self.data)) if self._max_steps else len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return self.preprocess_fn([{"role": "user", "content": sample["prompt"]}])

    def get_prompt(self, idx):
        return self.data[idx]["prompt"]

    def eval(self, predictions: dict):
        scores, total = [], len(predictions)
        details = []

        for idx in predictions:
            sample = self.data[idx]
            pred = predictions[idx]
            answer = sample["answer"]
            score = _mrcr_score(pred, answer)
            scores.append(score)
            details.append(
                {
                    "index": idx,
                    "score": round(score, 4),
                    "n_chars": sample.get("n_chars", 0),
                    "n_needles": sample.get("n_needles", self.n_needles),
                    "prediction_head": pred[:300],
                    "answer": answer[:200],
                }
            )

        avg_score = sum(scores) / total if total else 0.0
        logger.info(f"MRCR [{self.n_needles}-needle] — avg score: {avg_score:.4f} ({total} samples)")

        if self.eval_savepath:
            os.makedirs(self.eval_savepath, exist_ok=True)
            result_file = os.path.join(self.eval_savepath, f"mrcr_{self.n_needles}needle_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metric": "avg_score",
                        "avg_score": avg_score,
                        "total": total,
                        "n_needles": self.n_needles,
                        "max_chars": self.max_chars,
                        "details": details,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Results saved to {result_file}")

        return avg_score
