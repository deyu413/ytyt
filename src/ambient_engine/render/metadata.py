from __future__ import annotations

import random

from ambient_engine.core.durations import humanize_seconds
from ambient_engine.profiles.schema import Profile


class MetadataBuilder:
    def __init__(self, profile: Profile, seed: int) -> None:
        self.profile = profile
        self.random = random.Random(seed + 7)

    def build(self, target_seconds: int, variation: dict[str, object]) -> dict[str, object]:
        duration_label = humanize_seconds(target_seconds)
        title_template = self.random.choice(self.profile.title_families)
        title = (
            title_template.replace("{DURATION}", duration_label)
            .replace("{MOOD}", self.profile.mood.title())
            .replace("{SERIES}", self.profile.branding.get("series_name", "Ambient Series"))
        )
        branding = self.profile.branding
        description_lines = [
            title,
            "",
            branding.get("description_hook", "Long-form ambient music for late-night listening, sleep, and deep calm."),
            "",
            f"Profile: {self.profile.profile_id}",
            f"Mood: {self.profile.mood}",
            f"Duration: {duration_label}",
            f"Series: {branding.get('series_name', 'Ambient Series')}",
            f"Engine: {branding.get('engine_tag', 'Ambient Engine')}",
            "",
            "This render was generated with a free/open local-first pipeline.",
            "Publishing stays optional and decoupled from the audio engine.",
            "",
            "AI disclosure: synthetic audio / synthetic visual.",
        ]
        tags = branding.get("tags", []) + [self.profile.profile_id, str(variation.get("title_family", "ambient"))]
        return {
            "title": title[:100],
            "description": "\n".join(description_lines)[:5000],
            "tags": tags[:20],
            "language": self.profile.language,
            "privacy_status": branding.get("privacy_status", "private"),
            "channel_name": branding.get("channel_name", "Ambient Engine"),
            "thumbnail_text": branding.get("thumbnail_text", self.profile.mood.title()),
            "hud_label": branding.get("hud_label", self.profile.profile_id.replace("_", " ").title()),
        }

