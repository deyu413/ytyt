from __future__ import annotations

import random

from ambient_engine.planning.sections import SECTION_ROLE_ORDER, SectionPlan
from ambient_engine.profiles.schema import Profile


class MacrostructurePlanner:
    def __init__(self, profile: Profile, seed: int) -> None:
        self.profile = profile
        self.seed = seed
        self.random = random.Random(seed)

    def build(self, target_seconds: int) -> list[SectionPlan]:
        templates = sorted(
            self.profile.section_schema,
            key=lambda section: SECTION_ROLE_ORDER.index(section.role)
            if section.role in SECTION_ROLE_ORDER
            else len(SECTION_ROLE_ORDER),
        )

        base_total = sum(section.share for section in templates)
        if base_total <= 0:
            raise ValueError("Section shares must sum to a positive value.")
        min_duration = max(8, min(120, target_seconds // max(1, len(templates) * 2)))

        durations: list[int] = []
        remaining = target_seconds
        for index, template in enumerate(templates):
            if index == len(templates) - 1:
                duration = remaining
            else:
                share_ratio = template.share / base_total
                jitter = self.random.uniform(-0.08, 0.08)
                duration = max(min_duration, int(target_seconds * max(0.03, share_ratio + jitter * share_ratio)))
                remaining -= duration
            durations.append(duration)

        correction = target_seconds - sum(durations)
        durations[-1] += correction

        plan: list[SectionPlan] = []
        for index, template in enumerate(templates):
            density_jitter = self.random.uniform(-0.12, 0.12)
            density = min(0.95, max(0.05, template.density * (1.0 + density_jitter)))
            plan.append(
                SectionPlan(
                    index=index,
                    role=template.role,
                    duration_seconds=durations[index],
                    emotional_goal=template.emotional_goal,
                    density=density,
                    texture_policy=template.texture_policy,
                    harmonic_drift_policy=template.harmonic_drift_policy,
                    transition_policy=template.transition_policy,
                    max_reuse=template.max_reuse,
                    layer_budget=template.layer_budget,
                    generator_chain=list(template.allowed_generators),
                    variation_seed=self.seed + (index + 1) * 97,
                    notes=[
                        f"Share={template.share:.2f}",
                        f"Layer budget={template.layer_budget}",
                        f"Density jitter={density_jitter:+.03f}",
                    ],
                )
            )
        return plan
