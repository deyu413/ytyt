from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SectionPlan:
    index: int
    role: str
    duration_seconds: int
    emotional_goal: str
    density: float
    texture_policy: str
    harmonic_drift_policy: str
    transition_policy: str
    max_reuse: int
    layer_budget: int
    generator_chain: list[str]
    variation_seed: int
    notes: list[str] = field(default_factory=list)


SECTION_ROLE_ORDER = [
    "intro",
    "settle",
    "drift_a",
    "drift_b",
    "sparse_break",
    "return",
    "low_energy_tail",
]

