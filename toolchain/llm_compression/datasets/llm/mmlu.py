"""MMLU Dataset for calibration and evaluation.

Inherits from opencompass.datasets.MMLUDataset to reuse load().
Design philosophy: reuse public library's prompt and evaluate logic.

Requires: pip install opencompass
"""

import json
import logging
import os

from opencompass.configs.datasets.mmlu.mmlu_all_sets import mmlu_all_sets as MMLU_ALL_SETS
from opencompass.datasets.mmlu import MMLUDataset as _OCMMLUDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from tabulate import tabulate

from llm_compression.registry_factory import DATASET_REGISTRY

logger = logging.getLogger(__name__)


# OpenCompass default hint (aligned with opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py)
_OPENCOMPASS_EVAL_HINT = "Answer the question by replying A, B, C or D."
# Concise instruction that forces the model to output the option letter directly
_DIRECT_ANSWER_HINT = "Answer with only the letter (e.g., A), no explanation."


def _build_prompt_template(subject: str, direct_answer: bool = True) -> str:
    """Build prompt template aligned with opencompass mmlu_gen_4d595a.py."""
    answer_hint = _DIRECT_ANSWER_HINT if direct_answer else _OPENCOMPASS_EVAL_HINT
    hint = f"There is a single choice question about {subject.replace('_', ' ')}. {answer_hint}"
    return f"{hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: "


@DATASET_REGISTRY("mmlu")
class MMLUDataset(_OCMMLUDataset):
    """MMLU Dataset implementation reusing OpenCompass load and evaluate logic.

    Aligned with opencompass/configs/datasets/mmlu/mmlu_gen_4d595a.py:
    - opencompass.datasets.mmlu.MMLUDataset.load() for data loading
    - All 57 subjects (mmlu_all_sets) by default
    - prompt_template: subject hint + Question + A/B/C/D options
    - eval_cfg: first_option_postprocess(options='ABCD') + AccEvaluator
    - eval: per-subject accuracy + overall average

    Args:
        data_root: Path to MMLU dataset root. Must contain dev/ and test/ subdirs.
        eval_savepath: Directory to save inference and evaluation results.
        preprocess_fn: Callback to convert raw message into model-ready inputs.
        max_steps: If set, only the first max_steps samples are used.
        subject: 'all'/None for all 57 subjects, or a single/list of subject names.
        direct_answer: If True, use concise direct-answer hint; otherwise use OpenCompass default hint.
    """

    def __init__(self, **kwargs):
        self.data_root = kwargs.get("data_root")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")
        self._subject_arg = kwargs.get("subject")
        self._direct_answer = kwargs.get("direct_answer", True)
        self.data = self.load_data()

    def load_data(self):
        """Load MMLU data using inherited OpenCompass MMLUDataset.load().

        When subject is 'all' or None, loads all 57 subjects (mmlu_all_sets).
        """
        path = os.path.abspath(self.data_root) if self.data_root else ""
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(
                f"MMLU data_root must be a directory: {self.data_root}. "
                "Download from: http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip"
            )

        if self._subject_arg is None or self._subject_arg == "all":
            subjects = MMLU_ALL_SETS
        elif isinstance(self._subject_arg, list):
            subjects = self._subject_arg
        else:
            subjects = [self._subject_arg]

        data = []
        for subj in subjects:
            ds = self.load(path=path, name=subj)
            for item in ds["test"]:
                row = dict(item)
                row["_subject"] = subj
                data.append(row)
        return data

    def build_prompt(self, line: dict) -> str:
        """Build prompt using opencompass mmlu_gen_4d595a.py format."""
        subj = line.get("_subject", "abstract_algebra")
        template = _build_prompt_template(subj, self._direct_answer)
        return template.format(
            input=line["input"],
            A=line["A"],
            B=line["B"],
            C=line["C"],
            D=line["D"],
        )

    def eval(self, predictions: dict):
        """Evaluate MMLU predictions using opencompass AccEvaluator.

        Uses first_option_postprocess(options='ABCD') + AccEvaluator.
        Reports per-subject accuracy and overall average.

        Args:
            predictions: {index: raw_prediction_text}
        """
        indices = list(predictions.keys())
        pred_values = [predictions[i] for i in indices]
        targets = [self.data[i]["target"] for i in indices]
        subjects = [self.data[i].get("_subject", "") for i in indices]

        pred_processed = [first_option_postprocess(p, options="ABCD") for p in pred_values]
        evaluator = AccEvaluator()

        subject_to_indices = {}
        for i, subj in enumerate(subjects):
            subject_to_indices.setdefault(subj, []).append(i)

        rows = []
        total_correct = 0
        total_count = 0
        for subj in sorted(subject_to_indices.keys()):
            idx_list = subject_to_indices[subj]
            subj_preds = [pred_processed[i] for i in idx_list]
            subj_refs = [targets[i] for i in idx_list]
            subj_results = evaluator.score(predictions=subj_preds, references=subj_refs)
            subj_acc = subj_results.get("accuracy", 0.0) / 100.0
            subj_total = len(idx_list)
            subj_correct = round(subj_acc * subj_total)
            total_correct += subj_correct
            total_count += subj_total
            rows.append([f"mmlu_{subj}", f"{subj_acc:.4f} ({subj_correct}/{subj_total})"])

        overall_acc = total_correct / total_count if total_count > 0 else 0.0
        rows.append(["average", f"{overall_acc:.4f} ({total_correct}/{total_count})"])

        if self.eval_savepath:
            os.makedirs(self.eval_savepath, exist_ok=True)
            result_file = os.path.join(self.eval_savepath, "mmlu_infer.jsonl")
            with open(result_file, "w", encoding="utf-8") as f:
                for i, (idx, pred, gt) in enumerate(zip(indices, pred_values, targets, strict=True)):
                    f.write(
                        json.dumps(
                            {
                                "index": idx,
                                "subject": subjects[i],
                                "prediction": pred,
                                "pred_processed": pred_processed[i],
                                "target": gt,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info("Infer results saved to %s", result_file)

        logger.info("MMLU Evaluation Results:\n%s", tabulate(rows))

    def __len__(self):
        length = len(self.data)
        if self._max_steps is None:
            return length
        return min(self._max_steps, length)

    def __getitem__(self, idx):
        line = self.data[idx]
        prompt = self.build_prompt(line)
        message = [{"role": "user", "content": prompt}]
        return self.preprocess_fn(message)
