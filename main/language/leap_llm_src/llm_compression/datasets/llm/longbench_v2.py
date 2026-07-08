"""LongBench v2 Dataset for long-context multiple-choice evaluation.

503 questions across 6 domains with context lengths from 8K to 2M words.
Unified Accuracy metric with breakdowns by difficulty, length, and domain.

Data source: THUDM/LongBench-v2 on HuggingFace or local JSONL.
Reference: https://github.com/THUDM/LongBench
"""

import json
import logging
import os

from llm_compression.registry_factory import DATASET_REGISTRY

logger = logging.getLogger(__name__)

_MCQ_PROMPT = (
    "Read the following text and answer the multiple-choice question below."
    " Only output the letter of the correct answer (A, B, C, or D),"
    " no explanation.\n\n{context}\n\nQuestion: {question}\n"
    "A. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\nAnswer:"
)


def _first_option(text, options="ABCD"):
    for ch in text.strip():
        if ch in options:
            return ch
    return ""


@DATASET_REGISTRY("longbench_v2")
class LongBenchV2Dataset:
    def __init__(self, **kwargs):
        self.data_root = kwargs.get("data_root")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")
        self._difficulty = kwargs.get("difficulty")
        self._length = kwargs.get("length")
        self._domain = kwargs.get("domain")
        self.data = self._load_data()

    def _load_data(self):
        data = self._load_raw()
        if self._difficulty:
            data = [d for d in data if d.get("difficulty") == self._difficulty]
        if self._length:
            data = [d for d in data if d.get("length") == self._length]
        if self._domain:
            data = [d for d in data if d.get("domain") == self._domain]
        logger.info(
            f"LongBench-v2: {len(data)} samples"
            f" (difficulty={self._difficulty}, length={self._length}, domain={self._domain})"
        )
        return data

    def _load_raw(self):
        if self.data_root:
            if os.path.isfile(self.data_root) and self.data_root.endswith(".jsonl"):
                return self._load_jsonl(self.data_root)
            if os.path.isdir(self.data_root):
                jsonl_path = os.path.join(self.data_root, "longbench_v2.jsonl")
                if os.path.isfile(jsonl_path):
                    return self._load_jsonl(jsonl_path)

        from datasets import load_dataset

        ds = load_dataset("THUDM/LongBench-v2", split="train", trust_remote_code=True)
        return [dict(row) for row in ds]

    @staticmethod
    def _load_jsonl(path):
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} samples from {path}")
        return data

    def build_prompt(self, sample):
        return _MCQ_PROMPT.format(
            context=sample.get("context", ""),
            question=sample.get("question", ""),
            choice_A=sample.get("choice_A", ""),
            choice_B=sample.get("choice_B", ""),
            choice_C=sample.get("choice_C", ""),
            choice_D=sample.get("choice_D", ""),
        )

    def __len__(self):
        return min(self._max_steps, len(self.data)) if self._max_steps else len(self.data)

    def __getitem__(self, idx):
        prompt = self.build_prompt(self.data[idx])
        return self.preprocess_fn([{"role": "user", "content": prompt}])

    def eval(self, predictions: dict):
        correct, total, details = 0, len(predictions), []
        for idx in predictions:
            sample = self.data[idx]
            pred_letter = _first_option(predictions[idx])
            gt = sample.get("answer", "")
            is_correct = pred_letter == gt
            if is_correct:
                correct += 1
            details.append(
                {
                    "index": idx,
                    "prediction": predictions[idx][:200],
                    "pred_letter": pred_letter,
                    "answer": gt,
                    "correct": is_correct,
                    "difficulty": sample.get("difficulty", ""),
                    "length": sample.get("length", ""),
                    "domain": sample.get("domain", ""),
                }
            )

        accuracy = correct / total * 100 if total else 0.0

        breakdowns = {}
        for key in ("difficulty", "length", "domain"):
            groups = {}
            for d in details:
                g = d[key]
                groups.setdefault(g, {"correct": 0, "total": 0})
                groups[g]["total"] += 1
                if d["correct"]:
                    groups[g]["correct"] += 1
            breakdowns[key] = {
                g: f"{v['correct']}/{v['total']} ({v['correct']/v['total']*100:.1f}%)" for g, v in groups.items()
            }

        logger.info(f"LongBench-v2 — accuracy: {accuracy:.2f} ({correct}/{total})")
        for key, groups in breakdowns.items():
            logger.info(f"  {key}: {groups}")

        if self.eval_savepath:
            os.makedirs(self.eval_savepath, exist_ok=True)
            result_file = os.path.join(self.eval_savepath, "longbench_v2_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metric": "accuracy",
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                        "breakdowns": breakdowns,
                        "details": details,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Results saved to {result_file}")
