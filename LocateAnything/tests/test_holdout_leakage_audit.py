import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "calibration/audit_holdout.py"
spec = importlib.util.spec_from_file_location("audit_calibration_holdout_leakage", SCRIPT)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def test_hamming_distance():
    assert audit.hamming_distance(0, 0) == 0
    assert audit.hamming_distance(0, (1 << 64) - 1) == 64
    assert audit.hamming_distance(0b1010, 0b0011) == 2


def test_dhash_is_deterministic(tmp_path):
    image = audit.Image.new("RGB", (32, 32), "white")
    path = tmp_path / "image.png"
    image.save(path)
    assert audit.difference_hash(path) == audit.difference_hash(path)
