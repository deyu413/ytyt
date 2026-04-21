from __future__ import annotations

from ambient_engine.generation.contracts import BaseProvider, SectionRenderRequest, SectionRenderResult


class StableAudioOpenProvider(BaseProvider):
    name = "stable_audio_open"

    def render_section(self, request: SectionRenderRequest) -> SectionRenderResult:
        raise RuntimeError(
            "Stable Audio Open is optional and not part of the default commercial-safe path. "
            "Enable it explicitly only if the license terms fit your use case."
        )

