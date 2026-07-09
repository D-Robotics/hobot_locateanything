"""LongBench v1 Dataset for long-context evaluation.

21 tasks across 6 categories (Single-Doc QA, Multi-Doc QA, Summarization,
Few-shot, Synthetic, Code). Supports both English and Chinese.

Data source: THUDM/LongBench on HuggingFace or local JSONL files.
Reference: https://github.com/THUDM/LongBench

Requires: pip install rouge jieba fuzzywuzzy python-Levenshtein
"""

import json
import logging
import os
import re
import string
from collections import Counter

from llm_compression.registry_factory import DATASET_REGISTRY

logger = logging.getLogger(__name__)

# ---- Task prompt templates (from official LongBench repo) ----

TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question."
        " Answer the question as concisely as you can, using a single phrase if possible."
        " Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on"
        " the story as concisely as you can, using a single phrase if possible."
        " Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as"
        " you can, using a single phrase or sentence if possible. If the question cannot be"
        ' answered based on the information in the article, write "unanswerable". If the question'
        ' is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any'
        " explanation.\n\nArticle: {context}\n\n Answer the question based on the above article"
        " as concisely as you can, using a single phrase or sentence if possible. If the question"
        ' cannot be answered based on the information in the article, write "unanswerable". If'
        ' the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not'
        " provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following"
        " question based on the above text, only give me the answer and do not output any other"
        " words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
        "问题：{input}\n答案："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not"
        " output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the"
        " question based on the given passages. Only give me the answer and do not output any"
        " other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not"
        " output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the"
        " question based on the given passages. Only give me the answer and do not output any"
        " other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not"
        " output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the"
        " question based on the given passages. Only give me the answer and do not output any"
        " other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n"
        "请基于上述文章回答下面的问题。\n\n问题：{input}\n答案："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the"
        " report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the"
        " report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing the information you need to"
        " summarize.\nWrite a concise summary of the relevant"
        " information.\n\nTranscript:\n{context}\n\nQuery: {input}\n\nSummary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news."
        "\n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "vcsum": "下面有一段会议记录，请你简要总结会议的内容。\n\n会议记录：\n{context}\n\n会议总结：",
    "trec": (
        "Please determine the type of the question below. Here are some examples of" " questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not"
        " output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": "Summarize the given dialogue. Here are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates."
        " Please carefully read these paragraphs and determine how many unique paragraphs there"
        " are after removing duplicates. In other words, how many non-repeating paragraphs are"
        " there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after"
        " removing duplicates. The output format should only contain the number, such as 1, 2, 3,"
        " and so on.\n\nThe count of unique paragraphs is:"
    ),
    "passage_retrieval_en": (
        "The following are paragraphs from Wikipedia along with an abstract of a paper. Please"
        " determine which paragraph the abstract is from.\n\n{context}\n\nThe abstract is from"
        " paragraph:"
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一段的摘要。请确定摘要是哪一段的。\n\n{context}\n\n"
        "以上的摘要是第几段的？请回答数字。\n\n答案是第"
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

TASK_MAX_GEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

TASK_METRIC = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_zh",
    "gov_report": "rouge",
    "qmsum": "rouge",
    "multi_news": "rouge",
    "vcsum": "rouge_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge",
    "lsht": "classification",
    "passage_count": "count",
    "passage_retrieval_en": "retrieval",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

ALL_SUBSETS = list(TASK_PROMPTS.keys())


# ---- Evaluation metrics (ported from official LongBench eval.py) ----


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(remove_articles("".join(ch for ch in s.lower() if ch not in string.punctuation)).split())


def qa_f1_score(prediction, ground_truth):
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_zh_score(prediction, ground_truth):
    import jieba

    pred_tokens = [_normalize_answer(t) for t in jieba.cut(prediction, cut_all=False) if t.strip()]
    gt_tokens = [_normalize_answer(t) for t in jieba.cut(ground_truth, cut_all=False) if t.strip()]
    pred_tokens = [t for t in pred_tokens if t]
    gt_tokens = [t for t in gt_tokens if t]
    if not pred_tokens or not gt_tokens:
        return float(prediction.strip() == ground_truth.strip())
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def rouge_score(prediction, ground_truth):
    from rouge import Rouge

    # 截断到 4000 字符防止 rouge 库 _recon_lcs 递归爆栈（默认递归深度 1000）。
    # thinking 模式下 prediction 可能达数万字符，超过 4000 的部分对 ROUGE-L
    # 评分无实质贡献（ground truth 通常只有几百字符）。
    prediction = prediction[:4000] if prediction.strip() else "empty"
    ground_truth = ground_truth if ground_truth.strip() else "empty"
    return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth):
    import jieba
    from rouge import Rouge

    # 同 rouge_score，截断防递归爆栈
    prediction = " ".join(jieba.cut(prediction[:4000], cut_all=False)) if prediction.strip() else "empty"
    ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False)) if ground_truth.strip() else "empty"
    return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]


def classification_score(prediction, ground_truth, all_classes):
    em_match_list = [cls for cls in all_classes if cls in prediction]
    for cls in em_match_list[:]:
        for other in em_match_list:
            if cls != other and cls in other:
                em_match_list.remove(cls)
                break
    if not em_match_list:
        return 0.0
    return 1.0 / len(em_match_list) if ground_truth in em_match_list else 0.0


def code_sim_score(prediction, ground_truth):
    from fuzzywuzzy import fuzz

    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            return fuzz.ratio(line, ground_truth) / 100.0
    return 0.0


def count_score(prediction, ground_truth):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for n in numbers if n == str(ground_truth)) / len(numbers)


def retrieval_score(prediction, ground_truth):
    pattern = re.findall(r"Paragraph (\d+)", ground_truth)
    if not pattern:
        return 0.0
    gt_id = pattern[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for n in numbers if n == gt_id) / len(numbers)


def retrieval_zh_score(prediction, ground_truth):
    gt_numbers = re.findall(r"\d+", ground_truth)
    if not gt_numbers:
        return 0.0
    gt_id = gt_numbers[0]
    pred_numbers = re.findall(r"\d+", prediction)
    if not pred_numbers:
        return 0.0
    return sum(1 for n in pred_numbers if n == gt_id) / len(pred_numbers)


_METRIC_FN = {
    "qa_f1": qa_f1_score,
    "qa_f1_zh": qa_f1_zh_score,
    "rouge": rouge_score,
    "rouge_zh": rouge_zh_score,
    "code_sim": code_sim_score,
    "count": count_score,
    "retrieval": retrieval_score,
    "retrieval_zh": retrieval_zh_score,
}


def _compute_score(prediction, answers, metric_name, all_classes=None):
    if metric_name == "classification":
        return max(classification_score(prediction, ans, all_classes) for ans in answers)
    return max(_METRIC_FN[metric_name](prediction, ans) for ans in answers)


# ---- Dataset ----


@DATASET_REGISTRY("longbench")
class LongBenchDataset:
    def __init__(self, **kwargs):
        self.data_root = kwargs.get("data_root")
        self.subset = kwargs.get("subset", "hotpotqa")
        self.eval_savepath = kwargs.get("eval_savepath")
        self.preprocess_fn = kwargs["preprocess_fn"]
        self._max_steps = kwargs.get("max_steps")

        if self.subset not in TASK_PROMPTS:
            raise ValueError(f"Unknown LongBench subset '{self.subset}'. Available: {ALL_SUBSETS}")
        self.max_gen_tokens = TASK_MAX_GEN[self.subset]
        self.data = self._load_data()

    def _load_data(self):
        if self.data_root and os.path.isdir(self.data_root):
            jsonl_path = os.path.join(self.data_root, f"{self.subset}.jsonl")
            if os.path.isfile(jsonl_path):
                data = []
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                logger.info(f"Loaded {len(data)} samples from {jsonl_path}")
                return data

        from datasets import load_dataset

        ds = load_dataset("THUDM/LongBench", self.subset, split="test")
        data = [dict(row) for row in ds]
        logger.info(f"Loaded {len(data)} samples from HuggingFace THUDM/LongBench/{self.subset}")
        return data

    def build_prompt(self, sample):
        return TASK_PROMPTS[self.subset].format(context=sample.get("context", ""), input=sample.get("input", ""))

    def __len__(self):
        return min(self._max_steps, len(self.data)) if self._max_steps else len(self.data)

    def __getitem__(self, idx):
        prompt = self.build_prompt(self.data[idx])
        return self.preprocess_fn([{"role": "user", "content": prompt}])

    def eval(self, predictions: dict):
        metric_name = TASK_METRIC[self.subset]
        scores, details = [], []
        for idx in predictions:
            sample = self.data[idx]
            pred = predictions[idx]
            answers = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            score = _compute_score(pred, answers, metric_name, sample.get("all_classes"))
            scores.append(score)
            details.append({"index": idx, "prediction": pred[:200], "answers": answers[:3], "score": score})

        avg_score = sum(scores) / len(scores) * 100 if scores else 0.0
        logger.info(f"LongBench [{self.subset}] — {metric_name}: {avg_score:.2f} ({len(scores)} samples)")

        if self.eval_savepath:
            os.makedirs(self.eval_savepath, exist_ok=True)
            result_file = os.path.join(self.eval_savepath, f"longbench_{self.subset}_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "subset": self.subset,
                        "metric": metric_name,
                        "score": avg_score,
                        "num_samples": len(scores),
                        "details": details,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Results saved to {result_file}")

        return avg_score, metric_name
