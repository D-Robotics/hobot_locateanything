import os
import subprocess
from pathlib import Path


def pick_free_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"Using existing CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        return

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        print(f"Failed to query GPU with nvidia-smi: {exc}")
        return

    gpus = []
    for line in output.strip().splitlines():
        index, free_mem = line.split(",")
        gpus.append((int(index.strip()), int(free_mem.strip())))

    if not gpus:
        return

    best_gpu, best_free_mem = max(gpus, key=lambda item: item[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Auto selected GPU {best_gpu} with {best_free_mem} MiB free")


pick_free_gpu()

from PIL import Image

from locateanything_worker import LocateAnythingWorker

embodied_root = Path(__file__).resolve().parents[2] / "eagle" / "Embodied"
model_dir = embodied_root / "LocateAnything-3B"
image_path = Path(__file__).resolve().parent / "test-cat.jpg"
golden_dir = embodied_root / "deploy_s600" / "golden"

worker = LocateAnythingWorker(str(model_dir), device="cuda")

img = Image.open(image_path).convert("RGB")

result = worker.detect(img, ["cat"], max_new_tokens=256, verbose=False)

answer = result["answer"]
boxes = worker.parse_boxes(answer, img.width, img.height)

print("answer:", answer)
print("boxes:", boxes)

golden_dir.mkdir(parents=True, exist_ok=True)
(golden_dir / "official_answer.txt").write_text(answer, encoding="utf-8")
(golden_dir / "official_boxes.txt").write_text(str(boxes), encoding="utf-8")
