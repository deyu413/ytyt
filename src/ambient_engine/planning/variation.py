from __future__ import annotations

import random

from ambient_engine.profiles.schema import Profile


class VariationPlanner:
    def __init__(self, profile: Profile, seed: int) -> None:
        self.profile = profile
        self.random = random.Random(seed + 13)

    def build(self) -> dict[str, object]:
        title_family = self.random.choice(self.profile.title_families)
        reference_dark = _is_reference_dark_bass_profile(self.profile)
        sleep_safe = _is_sleep_safe_profile(self.profile)
        if reference_dark:
            subject_alignment = self.random.choice(["left", "center"])
            accent_variant = self.random.choice(["halo", "soft-grid"])
            pacing_variant = "long-breath"
            texture_variant = self.random.choice(["sub-room", "black-room"])
            rhythm_variant = "breathing-swell"
            movement_bias = round(self.random.uniform(0.46, 0.68), 3)
            motif_density = round(self.random.uniform(0.34, 0.56), 3)
            stereo_profile = self.random.choice(["balanced", "wide-tail"])
            harmonic_color = self.random.choice(["modal-third", "minor-seventh", "root-fifth"])
        elif sleep_safe:
            subject_alignment = self.random.choice(["left", "center"])
            accent_variant = self.random.choice(["soft-grid", "halo"])
            pacing_variant = self.random.choice(["glacial", "slow-bloom"])
            texture_variant = self.random.choice(["room-hush", "velvet-room"])
            rhythm_variant = self.random.choice(["muted-step", "heartbeat"])
            movement_bias = round(self.random.uniform(0.22, 0.44), 3)
            motif_density = round(self.random.uniform(0.24, 0.46), 3)
            stereo_profile = self.random.choice(["narrow-core", "balanced"])
            harmonic_color = self.random.choice(["root-fifth", "modal-third"])
        else:
            subject_alignment = self.random.choice(["left", "center", "right"])
            accent_variant = self.random.choice(["soft-grid", "rain-lines", "halo", "orbital"])
            pacing_variant = self.random.choice(["glacial", "slow-bloom", "steady-fall"])
            texture_variant = self.random.choice(["diffuse-air", "room-hush", "mist-rain", "tape-halo"])
            rhythm_variant = self.random.choice(["tidal", "muted-step", "glass-pulse", "heartbeat", "rainwalk"])
            movement_bias = round(self.random.uniform(0.35, 0.78), 3)
            motif_density = round(self.random.uniform(0.42, 0.88), 3)
            stereo_profile = self.random.choice(["narrow-core", "balanced", "wide-tail"])
            harmonic_color = self.random.choice(["root-fifth", "modal-third", "minor-seventh", "suspended-second"])
        instrumentation = self.profile.instrumentation.get("primary", [])
        chosen_primary = instrumentation[:]
        self.random.shuffle(chosen_primary)
        return {
            "title_family": title_family,
            "subject_alignment": subject_alignment,
            "accent_variant": accent_variant,
            "pacing_variant": pacing_variant,
            "texture_variant": texture_variant,
            "rhythm_variant": rhythm_variant,
            "movement_bias": movement_bias,
            "motif_density": motif_density,
            "stereo_profile": stereo_profile,
            "harmonic_color": harmonic_color,
            "primary_instruments": chosen_primary[: max(1, min(3, len(chosen_primary)))],
        }


def _is_sleep_safe_profile(profile: Profile) -> bool:
    mood_text = " ".join(
        [
            profile.mood,
            " ".join(profile.forbidden_artifacts),
            " ".join(str(item) for item in profile.instrumentation.get("primary", [])),
            " ".join(str(item) for item in profile.instrumentation.get("secondary", [])),
        ]
    ).lower()
    return any(token in mood_text for token in ["sleep", "rest", "insomnia", "solitude", "calm mind"])


def _is_reference_dark_bass_profile(profile: Profile) -> bool:
    return str(profile.instrumentation.get("production_dna", "")).lower() == "reference_dark_bass"
