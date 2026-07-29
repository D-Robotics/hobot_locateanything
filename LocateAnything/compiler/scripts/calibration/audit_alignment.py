#!/usr/bin/env python3
"""Independent audit of prompt/target semantics in a selected LA manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REF_RE = re.compile(r"<ref>(.*?)</ref>")
BOX_RE = re.compile(r"<box>((?:<\d+>){2}|(?:<\d+>){4})</box>")
COORD_RE = re.compile(r"<(\d+)>")


def normalize_phrase(value) -> str:
    phrase = " ".join(str(value or "").strip().split())
    return phrase.rstrip(" \t\r\n.,!?;:")


def expected_prompt(row: dict) -> str:
    task = row["task"]
    phrase = normalize_phrase(row.get("phrase"))
    labels = list(dict.fromkeys(normalize_phrase(value) for value in REF_RE.findall(str(row.get("target_response") or ""))))
    if task == "detection":
        categories = [normalize_phrase(value) for value in row.get("categories") or labels]
        return (
            "Locate all the instances that matches the following description: "
            + "</c>".join(categories)
            + "."
        )
    if task == "gui":
        return (f"Point to: {phrase}." if row.get("output_type") == "point"
                else f"Locate the region that matches the following description: {phrase}.")
    if task == "referring":
        return f"Locate a single instance that matches the following description: {phrase}."
    if task == "ocr":
        return "Detect all the text in box format."
    if task == "layout":
        categories = [normalize_phrase(value) for value in row.get("categories") or labels]
        return (
            "Detect all the objects in the image that belong to the category set: "
            + "</c>".join(categories)
            + "."
        )
    if task == "pointing":
        return f"Point to: {phrase}."
    raise ValueError(task)


def audit(rows: list[dict]) -> tuple[list[dict], Counter]:
    errors = []
    counts = Counter()
    for row in rows:
        task = row.get("task")
        refs = [value.strip() for value in REF_RE.findall(str(row.get("target_response") or "")) if value.strip()]
        geometries = [tuple(int(v) for v in COORD_RE.findall(m.group(1))) for m in BOX_RE.finditer(str(row.get("target_response") or ""))]
        reasons = []
        if row.get("prompt") != expected_prompt(row):
            reasons.append("prompt differs from canonical policy")
        if not refs or not geometries:
            reasons.append("missing ref or geometry")
        target_count = (row.get("metadata") or {}).get("target_count")
        if not isinstance(target_count, int) or target_count != len(geometries):
            reasons.append("metadata target_count disagrees with geometry count")
        if task in {"gui", "referring", "pointing"} and {normalize_phrase(value) for value in refs} != {normalize_phrase(row.get("phrase"))}:
            reasons.append("phrase/ref mismatch")
        if task == "detection" and set(refs) != set(row.get("categories") or []):
            reasons.append("detection refs disagree with categories")
        if task == "layout" and not set(refs).issubset(set(row.get("categories") or [])):
            reasons.append("layout ref outside categories")
        if task == "ocr":
            source_count = (
                ((row.get("metadata") or {}).get("hiertext_filter") or {})
                .get("parsed_word_boxes")
            )
            if source_count != target_count:
                reasons.append("OCR parsed source count disagrees with target count")
        if task == "layout":
            source_count = (
                ((row.get("metadata") or {}).get("layout_filter") or {})
                .get("unique_valid_boxes")
            )
            if source_count != target_count:
                reasons.append("layout parsed source count disagrees with target count")
        for geometry in geometries:
            if len(geometry) == 4 and (geometry[2] <= geometry[0] or geometry[3] <= geometry[1]):
                reasons.append("degenerate box")
        if reasons:
            errors.append({"bundle_id": row.get("bundle_id"), "task": task, "reasons": reasons})
        else:
            counts[task] += 1
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.selected_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors, counts = audit(rows)
    result = {"sample_count": len(rows), "aligned_count": sum(counts.values()), "error_count": len(errors), "aligned_by_task": dict(counts)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_jsonl.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
