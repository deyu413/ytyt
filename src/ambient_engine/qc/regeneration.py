from __future__ import annotations

from ambient_engine.generation.contracts import SectionRenderResult


class RegenerationPlanner:
    def plan(
        self,
        qc_metrics: dict[str, float],
        section_results: list[SectionRenderResult],
        max_sections: int = 2,
    ) -> dict[str, object]:
        if (
            qc_metrics["repetition_score"] < 0.90
            and qc_metrics["fatigue_risk"] < 0.60
            and qc_metrics.get("interference_risk", 0.0) < 0.24
            and qc_metrics["harshness"] < 0.18
            and qc_metrics.get("lowmid_boxiness_risk", 0.0) < 0.86
        ):
            return {"sections_to_regenerate": [], "reason": "No regeneration required."}
        target_roles = {"settle", "drift_a", "drift_b", "return"}
        candidates = [
            result.section_index
            for result in section_results
            if result.section_role in target_roles
        ]
        return {
            "sections_to_regenerate": candidates[:max_sections],
            "reason": "Interference, harshness, repetition, fatigue, or low-mid boxiness exceeded target; refresh the highest-exposure mid-body sections.",
        }
