"""Resolve LocateAnything product paths without depending on the checkout name."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class RepositoryPaths:
    repo_root: Path
    artifacts_root: Path
    compiler_root: Path
    compiler_scripts_root: Path
    deploy_root: Path
    calibration_root: Path
    evaluation_root: Path
    build_root: Path
    release_root: Path
    run_root: Path
    log_root: Path
    model_root: Path

    @classmethod
    def discover(cls, repo_root: Path | None = None) -> "RepositoryPaths":
        root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        artifacts = _path_env("LA_ARTIFACTS_ROOT", root / "artifacts")
        return cls(
            repo_root=root,
            artifacts_root=artifacts,
            compiler_root=_path_env("LA_COMPILER_ROOT", root / "compiler"),
            compiler_scripts_root=_path_env(
                "LA_COMPILER_SCRIPTS_ROOT", root / "compiler" / "scripts"
            ),
            deploy_root=_path_env("LA_DEPLOY_ROOT", root / "deploy"),
            calibration_root=_path_env("LA_CALIBRATION_ROOT", artifacts / "calibration"),
            evaluation_root=_path_env("LA_EVALUATION_ROOT", artifacts / "evaluation"),
            build_root=_path_env("LA_BUILD_ROOT", artifacts / "builds"),
            release_root=_path_env("LA_RELEASE_ROOT", artifacts / "releases"),
            run_root=_path_env("LA_RUN_ROOT", artifacts / "runs"),
            log_root=_path_env("LA_LOG_ROOT", artifacts / "logs"),
            model_root=_path_env("LA_MODEL_ROOT", artifacts / "models"),
        )

    def as_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in self.__dict__.items()}


PATHS = RepositoryPaths.discover()
