from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ambient_engine.core.durations import parse_duration


@dataclass(frozen=True)
class SectionTemplate:
    role: str
    share: float
    emotional_goal: str
    density: float
    texture_policy: str
    harmonic_drift_policy: str
    transition_policy: str
    max_reuse: int
    layer_budget: int
    allowed_generators: list[str]


@dataclass(frozen=True)
class Profile:
    profile_id: str
    language: str
    mood: str
    pulse_density: str
    tonal_center: dict[str, Any]
    scale_family: str
    instrumentation: dict[str, Any]
    texture_mix: dict[str, float]
    section_schema: list[SectionTemplate]
    default_target_length_seconds: int
    loudness_target_lufs: float
    thumbnail_style: dict[str, Any]
    title_families: list[str]
    forbidden_artifacts: list[str]
    branding: dict[str, Any]


REQUIRED_KEYS = {
    "profile_id",
    "language",
    "mood",
    "pulse_density",
    "tonal_center",
    "scale_family",
    "instrumentation",
    "texture_mix",
    "section_schema",
    "target_length",
    "loudness_target",
    "thumbnail_style",
    "title_families",
    "forbidden_artifacts",
    "branding",
}


def build_profile(data: dict[str, Any]) -> Profile:
    missing = sorted(REQUIRED_KEYS.difference(data))
    if missing:
        raise ValueError(f"Profile missing required keys: {', '.join(missing)}")

    section_schema = []
    for entry in data["section_schema"]:
        section_schema.append(
            SectionTemplate(
                role=entry["role"],
                share=float(entry["share"]),
                emotional_goal=entry["emotional_goal"],
                density=float(entry["density"]),
                texture_policy=entry["texture_policy"],
                harmonic_drift_policy=entry["harmonic_drift_policy"],
                transition_policy=entry["transition_policy"],
                max_reuse=int(entry["max_reuse"]),
                layer_budget=int(entry["layer_budget"]),
                allowed_generators=list(entry["allowed_generators"]),
            )
        )

    duration = parse_duration(data["target_length"]).seconds

    return Profile(
        profile_id=data["profile_id"],
        language=data["language"],
        mood=data["mood"],
        pulse_density=data["pulse_density"],
        tonal_center=dict(data["tonal_center"]),
        scale_family=data["scale_family"],
        instrumentation=dict(data["instrumentation"]),
        texture_mix={key: float(value) for key, value in data["texture_mix"].items()},
        section_schema=section_schema,
        default_target_length_seconds=duration,
        loudness_target_lufs=float(data["loudness_target"]),
        thumbnail_style=dict(data["thumbnail_style"]),
        title_families=list(data["title_families"]),
        forbidden_artifacts=list(data["forbidden_artifacts"]),
        branding=dict(data["branding"]),
    )

