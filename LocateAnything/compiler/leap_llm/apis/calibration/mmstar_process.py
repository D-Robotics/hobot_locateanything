import base64
import io
import os
import string

import pandas as pd
from PIL import Image


def dump_mmstar_image(line):
    def read_ok(img_path):
        if not os.path.exists(img_path):
            return False
        try:
            im = Image.open(img_path)
            assert im.size[0] > 0 and im.size[1] > 0
            return True
        except Exception:
            return False

    img_root = "/tmp/mmstar_infer_images"
    os.makedirs(img_root, exist_ok=True)
    tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
    if not read_ok(tgt_path):
        image_data = base64.b64decode(line["image"])
        image = Image.open(io.BytesIO(image_data))
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        image.save(tgt_path)
    tgt_path = [tgt_path]
    return tgt_path


def build_tsv_prompt(line):
    tgt_path = dump_mmstar_image(line)
    question = line["question"]
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
    prompt = ""
    if hint is not None:
        prompt += f"Hint: {hint}\n"
    prompt += f"Question: {question}\n"
    if len(options):
        prompt += options_prompt
        prompt += "Please select the correct answer from the options above. \n"
    msgs = []
    if isinstance(tgt_path, list):
        msgs.extend([dict(type="image", value=p) for p in tgt_path])
    else:
        msgs = [dict(type="image", value=tgt_path)]
    msgs.append(dict(type="text", value=prompt))

    return {"role": "user", "content": msgs}


def prepare_tsv_content(message):
    min_pixels = 1280 * 28 * 28
    max_pixels = 5120 * 28 * 28
    content = []
    for s in message:
        if s["type"] == "image":
            item = {"type": "image", "image": "file://" + s["value"]}
            item["min_pixels"] = min_pixels
            item["max_pixels"] = max_pixels
        elif s["type"] == "text":
            item = {"type": "text", "text": s["value"]}
        else:
            raise ValueError(f"Invalid message type: {s['type']}, {s}")
        content.append(item)

    return content
