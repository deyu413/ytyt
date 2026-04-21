from __future__ import annotations


def gate_render(metrics: dict[str, float], score_card: dict[str, object], minimum_score: float, true_peak_ceiling: float) -> dict[str, object]:
    failures = []
    if score_card["global_score"] < minimum_score:
        failures.append(f"Global score below threshold ({score_card['global_score']} < {minimum_score}).")
    if metrics["true_peak"] > true_peak_ceiling:
        failures.append("True peak exceeds configured ceiling.")
    if metrics["silence_ratio"] > 0.15:
        failures.append("Too much low-level silence for the target format.")
    if metrics["repetition_score"] > 0.92:
        failures.append("Repetition score indicates audible mass-produced looping risk.")
    if metrics.get("interference_risk", 0.0) > 0.24:
        failures.append("Interference and upper-mid instability risk too high for ambient listening.")
    if metrics["harshness"] > 0.18:
        failures.append("Harsh presence exceeded the sleep-safe threshold.")
    if metrics.get("lowmid_boxiness_risk", 0.0) > 0.86:
        failures.append("Low-mid boxiness is too high; the render lacks a usable bass anchor.")
    if metrics["fatigue_risk"] > 0.62:
        failures.append("Fatigue risk too high for long listening sessions.")
    return {
        "accepted": len(failures) == 0,
        "failures": failures,
    }
