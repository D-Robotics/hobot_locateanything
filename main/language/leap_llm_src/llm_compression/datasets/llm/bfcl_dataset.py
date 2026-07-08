"""BFCL (Berkeley Function Calling Leaderboard) Dataset for LLM evaluation.

Design principles:
1. Inherit existing architecture, Dataset handles data loading and evaluation logic
2. Both single-turn and multi-turn use BFCL Handler for inference (reuse prompt preprocessing logic)
3. eval() method completes full inference and evaluation internally

BFCL output format requirements:
- Expected: [func_name(params_name=params_value), ...]
- Example: [mv(source='file.pdf', destination='temp')]
"""

import json
import os
from datetime import datetime

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
from bfcl_eval.utils import is_function_calling_format_output, load_dataset_entry, load_ground_truth_entry
from tabulate import tabulate
from tqdm import tqdm

from llm_compression.datasets.llm.bfcl_handler import LLMCompressionHandler
from llm_compression.registry_factory import DATASET_REGISTRY
from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)

BFCL_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "parallel_multiple",
    "multiple",
    "irrelevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]

MULTI_TURN_CATEGORIES = {"multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"}


def is_multi_turn_category(category: str) -> bool:
    return category in MULTI_TURN_CATEGORIES


@DATASET_REGISTRY("bfcl")
class BFCLDataset:
    """BFCL Dataset for Function Calling evaluation.

    Both single-turn and multi-turn use Handler for inference,
    reusing BFCL's prompt preprocessing logic.
    """

    def __init__(self, **kwargs):
        self.category = kwargs.get("category", "simple_python")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.model_name = kwargs.get("model_name", "custom")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")
        self._is_multi_turn = is_multi_turn_category(self.category)

        # Load data
        self.test_data = load_dataset_entry(self.category, include_language_specific_hint=True)
        self.ground_truths = load_ground_truth_entry(self.category)

    def __len__(self) -> int:
        length = len(self.test_data)
        return min(length, self._max_steps) if self._max_steps else length

    def __getitem__(self, idx: int):
        """Return raw data entry (BFCL evaluation does not use dataloader iteration)."""
        return self.test_data[idx]

    def eval(self, predictions=None, model=None, q_model=None, dtype=None, do_sample=False, handler=None):
        """Evaluate: use Handler for inference, then evaluate results.

        Args:
            predictions: Unused (kept for interface compatibility)
            model: Model instance
            q_model: QModel instance
            dtype: Inference data type
            do_sample: Whether to sample
            handler: External handler instance (e.g. GGUFHandler). If provided, model/q_model/dtype are ignored.
        """
        if handler is None:
            handler = LLMCompressionHandler(
                model=model,
                q_model=q_model,
                model_name=self.model_name,
                temperature=0.001,
                dtype=dtype,
                do_sample=do_sample,
            )

        if self._is_multi_turn:
            return self._eval_multi_turn(handler)
        else:
            return self._eval_single_turn(handler)

    def _eval_single_turn(self, handler):
        """Single-turn evaluation: use BFCL Handler inference + ast_checker."""
        # Determine language (ast_checker uses Language, decode_ast uses ReturnFormat)
        if "java" in self.category.lower() and "javascript" not in self.category.lower():
            language = Language.JAVA
            return_format = ReturnFormat.JAVA
        elif "javascript" in self.category.lower():
            language = Language.JAVASCRIPT
            return_format = ReturnFormat.JAVASCRIPT
        else:
            language = Language.PYTHON
            return_format = ReturnFormat.PYTHON

        correct = 0
        total = len(self)
        sample_details = []

        for idx in tqdm(range(total), desc=f"BFCL eval ({self.category})"):
            test_entry = self.test_data[idx]
            possible_answer = self.ground_truths[idx]["ground_truth"]
            func_desc = test_entry["function"]

            # Use BFCL Handler for inference
            output_text, metadata = handler.inference(
                test_entry=test_entry,
                include_input_log=False,
            )

            detail = {"id": test_entry.get("id", idx), "output": output_text}

            # Parse output
            try:
                parsed = handler.decode_ast(output_text, return_format, has_tool_call_tag=False)
            except Exception as e:
                logger.warning("Sample %d decode_ast failed: %s | output: %.200s", idx, e, output_text)
                detail["status"] = "decode_error"
                detail["error"] = str(e)
                sample_details.append(detail)
                continue

            if not is_function_calling_format_output(parsed):
                logger.warning("Sample %d output is not function calling format: %.200s", idx, output_text)
                detail["status"] = "format_error"
                sample_details.append(detail)
                continue

            # Evaluate
            try:
                result = ast_checker(
                    func_description=func_desc,
                    model_output=parsed,
                    possible_answer=possible_answer,
                    language=language,
                    test_category=self.category,
                    model_name=self.model_name,
                )
            except KeyError as e:
                # BFCL library bug: JAVA_TYPE_CONVERSION/JS_TYPE_CONVERSION missing lowercase "string"
                logger.warning("Sample %d ast_checker KeyError: %s (BFCL library bug)", idx, e)
                detail["status"] = "eval_error"
                detail["error"] = f"ast_checker KeyError: {e}"
                sample_details.append(detail)
                continue

            is_valid = result.get("valid", False)
            if is_valid:
                correct += 1

            detail["status"] = "correct" if is_valid else "incorrect"
            sample_details.append(detail)

        accuracy = correct / total if total > 0 else 0
        self._print_results(correct, total, accuracy)
        self._save_results(
            {"correct": correct, "incorrect": total - correct, "total": total, "accuracy": accuracy},
            sample_details,
        )

        return accuracy

    def _eval_multi_turn(self, handler):
        """Multi-turn evaluation: reuse BFCL Handler inference pipeline."""
        correct = 0
        total = len(self)
        sample_details = []

        for idx in tqdm(range(total), desc=f"BFCL eval ({self.category})"):
            test_entry = self.test_data[idx]
            test_entry_id = test_entry["id"]
            ground_truth = self.ground_truths[idx]["ground_truth"]

            # Use BFCL Handler for inference
            result, metadata = handler.inference(
                test_entry=test_entry,
                include_input_log=False,
                exclude_state_log=True,
            )

            # Evaluate
            eval_result = _evaluate_single_multi_turn_entry(
                handler=handler,
                test_entry_id=test_entry_id,
                model_result_list=result,
                ground_truth_list=ground_truth,
                prompt_entry=test_entry,
                model_name=self.model_name,
                test_category=self.category,
            )

            is_valid = eval_result.get("valid", False)
            if is_valid:
                correct += 1

            detail = {
                "id": test_entry_id,
                "status": "correct" if is_valid else "incorrect",
                "result": result,
            }
            if not is_valid:
                detail["error_type"] = eval_result.get("error_type", "unknown")
            sample_details.append(detail)

        accuracy = correct / total if total > 0 else 0
        self._print_results(correct, total, accuracy)
        self._save_results(
            {"correct": correct, "incorrect": total - correct, "total": total, "accuracy": accuracy},
            sample_details,
        )

        return accuracy

    def _print_results(self, correct, total, accuracy):
        """Print evaluation results."""
        print("\n" + "=" * 50)
        print(f"BFCL Evaluation Results ({self.category})")
        print("=" * 50)
        print(
            tabulate(
                [
                    ("correct", correct),
                    ("incorrect", total - correct),
                    ("total", total),
                    ("accuracy", f"{accuracy:.2%}"),
                ],
                headers=["Metric", "Value"],
            )
        )

    def _save_results(self, results: dict, sample_details: list):
        """Save evaluation results with per-sample details."""
        if not self.eval_savepath:
            return

        os.makedirs(self.eval_savepath, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = self.model_name.replace("/", "_")
        result_file = os.path.join(
            self.eval_savepath,
            f"bfcl_{self.category}_{safe_model_name}_{timestamp}.json",
        )

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "category": self.category,
                        "model_name": self.model_name,
                        "is_multi_turn": self._is_multi_turn,
                    },
                    "results": results,
                    "sample_details": sample_details,
                    "timestamp": timestamp,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Results saved to: {result_file}")
