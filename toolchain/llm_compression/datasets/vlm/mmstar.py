"""MMStar Dataset for calibration and evaluation."""

import os

from tabulate import tabulate
from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.smp.file import download_file, dump, load, md5

from llm_compression.registry_factory import DATASET_REGISTRY
from llm_compression.utils.logger import get_logger

logger = get_logger(__name__)

# VLMEvalKit default hint for MCQ datasets
_VLMEVAL_MCQ_HINT = "Please select the correct answer from the options above."
# Concise instruction that forces the model to output the option letter directly
# Using strong imperative language to ensure the model follows the instruction
_DIRECT_ANSWER_HINT = (
    "Answer with ONLY the single letter (A, B, C, or D). Do NOT explain. Do NOT elaborate. Just output one letter."
)


@DATASET_REGISTRY("mmstar")
class MMStarDataset(ImageMCQDataset):
    """
    MMStar Dataset implementation based on VLMEvalKit ImageMCQDataset.

    Args:
    data_root (str):
        Path to the MMStar TSV file. If the file does not exist or MD5
        does not match, it will be downloaded automatically.
    eval_savepath (str):
        Directory used to save inference and evaluation results.
        Defaults to current directory.
    preprocess_fn (Callable):
        Callback to convert raw message into model-ready inputs.
        __getitem__ returns preprocess_fn(message).
    max_steps (int, optional):
        If set, only the first max_steps samples are used. None means full dataset.
    """

    def __init__(self, **kwargs):
        self.data_root = kwargs.get("data_root")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")
        self._direct_answer = kwargs.get("direct_answer", True)
        super().__init__(dataset="MMStar")

    def __len__(self):
        length = super().__len__()
        if self._max_steps is None:
            return length
        return min(self._max_steps, length)

    def build_prompt(self, line):
        """
        Build a prompt using VLMEvalkit for consistency, with custom formatting.
        """
        prompt = super().build_prompt(line)
        content = []
        for s in prompt:
            if s["type"] == "image":
                item = {"type": "image", "image": "file://" + s["value"]}
            elif s["type"] == "text":
                text = s["value"]
                if self._direct_answer:
                    if _VLMEVAL_MCQ_HINT in text:
                        text = text.replace(_VLMEVAL_MCQ_HINT, _DIRECT_ANSWER_HINT)
                    else:
                        text = text.rstrip() + "\n" + _DIRECT_ANSWER_HINT
                item = {"type": "text", "text": text}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def load_data(self, dataset):
        data_path = self.data_root
        if os.path.exists(data_path):
            expected_md5 = self.DATASET_MD5.get(dataset)
            actual_md5 = md5(data_path)
            if expected_md5 and actual_md5 != expected_md5:
                logger.warning(
                    f"MD5 mismatch for {data_path}. "
                    f"Expected: {expected_md5}, Actual: {actual_md5}. "
                    "Data may be corrupted or modified."
                )
            return load(data_path)

        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == "":
            url = dataset + ".tsv"
        logger.warning(f"Cannot find data in {data_path}, will download from url")
        download_file(url, data_path)
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Run official MMStar evaluation.

        This is a thin wrapper over the parent class's evaluate method.
        """
        return super().evaluate(eval_file, **judge_kwargs)

    def eval(self, predictions):
        """Evaluate MMStar predictions using VLMEvalkit and save results.

        Args:
            predictions (dict): {index: prediction}
                index: sample index (from enumerate, 0, 1, 2, ...)
                prediction: model prediction (e.g. from batch_decode)
        """
        # indices from enumerate(dataloader) correspond to dataset positions (iloc),
        indices = list(predictions.keys())
        pred_values = [predictions[i] for i in indices]
        dataset = self.data.iloc[indices].copy()
        dataset["prediction"] = pred_values
        eval_data = dataset
        if self.eval_savepath is None:
            self.eval_savepath = "./"
        os.makedirs(self.eval_savepath, exist_ok=True)
        result_file = os.path.join(self.eval_savepath, "MMStar_infer.xlsx")
        # Remove vlmeval cache to avoid stale accuracy when xlsx is overwritten
        cache_suffixes = [
            "_exact_matching_result.pkl",
            "_exact_matching_result.xlsx",
            "_acc.csv",
        ]
        for suffix in cache_suffixes:
            cache_path = os.path.join(self.eval_savepath, "MMStar_infer" + suffix)
            if os.path.exists(cache_path):
                os.remove(cache_path)
        logger.info(f"The infer results will be saved to {self.eval_savepath}")
        dump(eval_data, result_file)
        acc = self.evaluate(result_file)
        print("Evaluation Results:")
        print(tabulate(acc.T))

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample = self.build_prompt(sample)
        message = [{"role": "user", "content": sample}]
        return self.preprocess_fn(message)
