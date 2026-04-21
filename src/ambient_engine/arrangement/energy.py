from __future__ import annotations


def role_energy(role: str) -> float:
    mapping = {
        "intro": 0.58,
        "settle": 0.63,
        "drift_a": 0.77,
        "drift_b": 0.82,
        "sparse_break": 0.42,
        "return": 0.74,
        "low_energy_tail": 0.3,
    }
    return mapping.get(role, 0.65)

