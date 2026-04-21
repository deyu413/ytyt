from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ambient_engine.core.paths import ProjectPaths


@dataclass
class ProviderCapability:
    available: bool
    reason: str


@dataclass
class RuntimeContext:
    mode: str
    python_executable: str
    ffmpeg_executable: str | None
    ffprobe_executable: str | None
    provider_capabilities: dict[str, ProviderCapability] = field(default_factory=dict)

    @property
    def gpu_enabled(self) -> bool:
        return self.mode == "gpu"


def _find_imageio_binary(kind: str) -> str | None:
    module_name = "imageio_ffmpeg"
    if importlib.util.find_spec(module_name) is None:
        return None
    import imageio_ffmpeg

    if kind == "ffmpeg":
        return imageio_ffmpeg.get_ffmpeg_exe()
    return None


def detect_runtime(mode: str, project_paths: ProjectPaths) -> RuntimeContext:
    ffmpeg_executable = (
        _find_imageio_binary("ffmpeg")
        or shutil.which("ffmpeg")
        or None
    )
    ffprobe_executable = shutil.which("ffprobe")

    capabilities = {
        "procedural": ProviderCapability(True, "Bundled DSP provider."),
        "ace_step": _detect_repo_provider(
            env_var="AMBIENT_ACESTEP_PATH",
            fallback_path=project_paths.root / "third_party" / "ACE-Step-1.5",
            mode=mode,
            gpu_required=True,
            label="ACE-Step 1.5",
        ),
        "yue": _detect_repo_provider(
            env_var="AMBIENT_YUE_PATH",
            fallback_path=project_paths.root / "third_party" / "YuE",
            mode=mode,
            gpu_required=True,
            label="YuE",
        ),
        "stable_audio_open": _detect_optional_package(
            env_var="AMBIENT_STABLE_AUDIO_PATH",
            package_name="stable_audio_tools",
            mode=mode,
            gpu_required=True,
            label="Stable Audio Open",
        ),
    }

    return RuntimeContext(
        mode=mode,
        python_executable=os.environ.get("PYTHON_EXECUTABLE", "python"),
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        provider_capabilities=capabilities,
    )


def _detect_repo_provider(
    env_var: str,
    fallback_path: Path,
    mode: str,
    gpu_required: bool,
    label: str,
) -> ProviderCapability:
    configured = os.environ.get(env_var)
    candidate = Path(configured) if configured else fallback_path
    if gpu_required and mode != "gpu":
        return ProviderCapability(False, f"{label} disabled because runtime mode is cpu-safe.")
    if candidate.exists():
        return ProviderCapability(True, f"{label} detected at {candidate}.")
    return ProviderCapability(False, f"{label} not found. Set {env_var} or install under third_party/.")


def _detect_optional_package(
    env_var: str,
    package_name: str,
    mode: str,
    gpu_required: bool,
    label: str,
) -> ProviderCapability:
    if gpu_required and mode != "gpu":
        return ProviderCapability(False, f"{label} disabled because runtime mode is cpu-safe.")
    if os.environ.get(env_var):
        return ProviderCapability(True, f"{label} configured via {env_var}.")
    if importlib.util.find_spec(package_name) is not None:
        return ProviderCapability(True, f"{label} python package detected.")
    return ProviderCapability(False, f"{label} optional dependency not installed.")

