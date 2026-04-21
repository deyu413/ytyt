from __future__ import annotations

from ambient_engine.profiles.schema import Profile


DEFAULT_STEM_LEVELS_DB = {
    "drone": -2.5,
    "motion": -5.0,
    "texture": -9.5,
    "accents": -12.0,
    "rhythm": -11.5,
}


def resolve_stem_levels(profile: Profile) -> dict[str, float]:
    configured = profile.instrumentation.get("stem_levels_db", {})
    levels = DEFAULT_STEM_LEVELS_DB.copy()
    for key, value in configured.items():
        levels[key] = float(value)
    return levels
