from __future__ import annotations


WEIGHTS = {
    "true_peak": 0.14,
    "integrated_loudness": 0.12,
    "silence_ratio": 0.08,
    "repetition_score": 0.12,
    "section_boundary_smoothness": 0.12,
    "spectral_monotony": 0.10,
    "harshness": 0.10,
    "interference_risk": 0.10,
    "mono_collapse_risk": 0.08,
    "dynamic_flatness": 0.07,
    "artifact_spike_ratio": 0.07,
    "fatigue_risk": 0.10,
}

OPTIONAL_WEIGHTS = {
    "bass_anchor_score": 0.08,
    "dynamic_breath_score": 0.08,
    "stereo_depth_score": 0.06,
    "dark_balance_score": 0.05,
    "lowmid_boxiness_risk": 0.07,
    "reference_dna_score": 0.08,
}


def score_metrics(metrics: dict[str, float], target_lufs: float, true_peak_ceiling: float) -> dict[str, object]:
    loudness_distance = abs(metrics["integrated_loudness"] - target_lufs)
    subscores = {
        "true_peak": max(0.0, 1.0 - max(0.0, metrics["true_peak"] - true_peak_ceiling) * 6.0),
        "integrated_loudness": max(0.0, 1.0 - loudness_distance / 8.0),
        "silence_ratio": max(0.0, 1.0 - metrics["silence_ratio"] * 4.0),
        "repetition_score": max(0.0, 1.0 - metrics["repetition_score"]),
        "section_boundary_smoothness": metrics["section_boundary_smoothness"],
        "spectral_monotony": max(0.0, 1.0 - metrics["spectral_monotony"]),
        "harshness": max(0.0, 1.0 - metrics["harshness"] * 3.5),
        "interference_risk": max(0.0, 1.0 - metrics["interference_risk"] * 1.8),
        "mono_collapse_risk": max(0.0, 1.0 - metrics["mono_collapse_risk"] * 1.6),
        "dynamic_flatness": max(0.0, 1.0 - metrics["dynamic_flatness"]),
        "artifact_spike_ratio": max(0.0, 1.0 - metrics["artifact_spike_ratio"] * 6.0),
        "fatigue_risk": max(0.0, 1.0 - metrics["fatigue_risk"]),
    }
    optional_subscores = {
        "bass_anchor_score": metrics.get("bass_anchor_score", 0.0),
        "dynamic_breath_score": metrics.get("dynamic_breath_score", 0.0),
        "stereo_depth_score": metrics.get("stereo_depth_score", 0.0),
        "dark_balance_score": metrics.get("dark_balance_score", 0.0),
        "lowmid_boxiness_risk": max(0.0, 1.0 - metrics.get("lowmid_boxiness_risk", 0.0)),
        "reference_dna_score": metrics.get("reference_dna_score", 0.0),
    }
    for name, value in optional_subscores.items():
        if name in metrics:
            subscores[name] = max(0.0, min(1.0, float(value)))
    active_weights = dict(WEIGHTS)
    active_weights.update({name: weight for name, weight in OPTIONAL_WEIGHTS.items() if name in metrics})
    weight_total = sum(active_weights.values()) or 1.0
    score = 0.0
    for name, weight in active_weights.items():
        score += subscores[name] * weight
    score /= weight_total
    return {
        "global_score": round(score * 100.0, 2),
        "subscores": {key: round(value * 100.0, 2) for key, value in subscores.items()},
    }
