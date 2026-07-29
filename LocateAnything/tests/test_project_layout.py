from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from oe_locateanything.paths import RepositoryPaths


ROOT = Path(__file__).resolve().parents[1]


def test_product_has_one_active_source_tree():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "oe-locateanything"
    assert metadata["project"]["version"] == "0.4.0"
    assert metadata["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    for directory in ("assets", "compiler", "deploy", "docs", "src", "tests", "workspace"):
        assert (ROOT / directory).is_dir(), directory
    for removed in ("main", "baselines", "configs", "examples", "oellm", "reports"):
        assert not (ROOT / removed).exists(), removed


def test_active_entrypoints_are_direct_implementations():
    expected = (
        "compiler/quantize.py",
        "compiler/config.yaml",
        "compiler/scripts/build/language.sh",
        "compiler/scripts/build/vision.sh",
        "compiler/scripts/calibration/prepare.sh",
        "compiler/scripts/calibration/calibrate.sh",
        "compiler/scripts/validate/compare_pipeline.py",
        "compiler/scripts/validate/hbm_sanity.py",
        "compiler/scripts/validate/evaluate_grounding.py",
        "deploy/deploy_locateanything_s600.sh",
        "deploy/run_s600_benchmark.sh",
        "deploy/run_locateanything_demo.sh",
    )
    for entrypoint in expected:
        assert (ROOT / entrypoint).is_file(), entrypoint


def test_central_path_defaults_match_product_layout(monkeypatch):
    for name in (
        "LA_WORKSPACE",
        "LA_COMPILER_ROOT",
        "LA_COMPILER_SCRIPTS_ROOT",
        "LA_DEPLOY_ROOT",
        "LA_CALIBRATION_ROOT",
        "LA_EVALUATION_ROOT",
        "LA_ARTIFACT_ROOT",
        "LA_RUN_ROOT",
        "LA_MODEL_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    paths = RepositoryPaths.discover(ROOT)
    assert paths.workspace_root == (ROOT / "workspace").resolve()
    assert paths.compiler_root == (ROOT / "compiler").resolve()
    assert paths.compiler_scripts_root == (ROOT / "compiler" / "scripts").resolve()
    assert paths.deploy_root == (ROOT / "deploy").resolve()
    assert paths.calibration_root == (ROOT / "workspace" / "calibration").resolve()
    assert paths.evaluation_root == (ROOT / "workspace" / "evaluation").resolve()
    assert paths.artifact_root == (ROOT / "workspace" / "artifacts").resolve()
    assert paths.model_root == (ROOT / "workspace" / "models").resolve()
