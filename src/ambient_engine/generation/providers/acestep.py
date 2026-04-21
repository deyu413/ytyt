from __future__ import annotations

from ambient_engine.generation.contracts import BaseProvider, SectionRenderRequest, SectionRenderResult


class AceStepProvider(BaseProvider):
    name = "ace_step_1_5"

    def render_section(self, request: SectionRenderRequest) -> SectionRenderResult:
        raise RuntimeError(
            "ACE-Step provider is configured as optional. Install ACE-Step 1.5 and wire "
            "an inference adapter before using it in this environment."
        )

