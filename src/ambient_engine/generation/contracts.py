from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ambient_engine.core.paths import SessionPaths
from ambient_engine.planning.sections import SectionPlan
from ambient_engine.profiles.schema import Profile


@dataclass
class SectionRenderRequest:
    session_paths: SessionPaths
    profile: Profile
    section: SectionPlan
    sample_rate: int
    block_seconds: int
    channels: int
    stem_names: list[str]
    session_seed: int
    variation: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionRenderResult:
    section_index: int
    section_role: str
    provider_name: str
    transition_policy: str
    stem_files: dict[str, Path]
    duration_seconds: int
    sample_rate: int
    notes: dict[str, Any] = field(default_factory=dict)


class BaseProvider:
    name = "base"

    def __init__(self, available: bool, reason: str) -> None:
        self.available = available
        self.reason = reason

    def render_section(self, request: SectionRenderRequest) -> SectionRenderResult:
        raise NotImplementedError
