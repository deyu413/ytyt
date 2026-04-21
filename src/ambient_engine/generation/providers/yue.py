from __future__ import annotations

from ambient_engine.generation.contracts import BaseProvider, SectionRenderRequest, SectionRenderResult


class YuEProvider(BaseProvider):
    name = "yue"

    def render_section(self, request: SectionRenderRequest) -> SectionRenderResult:
        raise RuntimeError(
            "YuE provider is optional and intended for structured experiments. "
            "Configure a local YuE install before using it."
        )

