from __future__ import annotations

from pathlib import Path

import yaml

from ambient_engine.profiles.schema import Profile, build_profile


def load_profile(profile_path: Path) -> Profile:
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    return build_profile(raw)


def load_profile_by_id(profiles_dir: Path, profile_id: str) -> Profile:
    profile_path = profiles_dir / f"{profile_id}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    return load_profile(profile_path)


def list_profiles(profiles_dir: Path) -> list[str]:
    return sorted(path.stem for path in profiles_dir.glob("*.yaml"))

