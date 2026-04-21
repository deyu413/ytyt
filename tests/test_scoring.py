import unittest

from ambient_engine.qc.scoring import score_metrics


class ScoringTests(unittest.TestCase):
    def test_global_score_is_normalized_to_100(self) -> None:
        metrics = {
            "true_peak": 0.5,
            "integrated_loudness": -18.0,
            "silence_ratio": 0.0,
            "repetition_score": 0.0,
            "section_boundary_smoothness": 1.0,
            "spectral_monotony": 0.0,
            "harshness": 0.0,
            "interference_risk": 0.0,
            "mono_collapse_risk": 0.0,
            "dynamic_flatness": 0.0,
            "artifact_spike_ratio": 0.0,
            "fatigue_risk": 0.0,
        }
        score_card = score_metrics(metrics, target_lufs=-18.0, true_peak_ceiling=0.8912509381337456)
        self.assertLessEqual(score_card["global_score"], 100.0)
        self.assertGreaterEqual(score_card["global_score"], 99.0)

    def test_interference_penalty_reduces_score(self) -> None:
        base = {
            "true_peak": 0.5,
            "integrated_loudness": -18.0,
            "silence_ratio": 0.0,
            "repetition_score": 0.0,
            "section_boundary_smoothness": 1.0,
            "spectral_monotony": 0.0,
            "harshness": 0.02,
            "interference_risk": 0.0,
            "mono_collapse_risk": 0.0,
            "dynamic_flatness": 0.0,
            "artifact_spike_ratio": 0.0,
            "fatigue_risk": 0.0,
        }
        clean = score_metrics(base, target_lufs=-18.0, true_peak_ceiling=0.8912509381337456)
        dirty = score_metrics({**base, "interference_risk": 0.55}, target_lufs=-18.0, true_peak_ceiling=0.8912509381337456)
        self.assertLess(dirty["global_score"], clean["global_score"])

    def test_reference_dna_subscores_affect_score_when_present(self) -> None:
        base = {
            "true_peak": 0.5,
            "integrated_loudness": -19.0,
            "silence_ratio": 0.0,
            "repetition_score": 0.0,
            "section_boundary_smoothness": 1.0,
            "spectral_monotony": 0.2,
            "harshness": 0.0,
            "interference_risk": 0.0,
            "mono_collapse_risk": 0.0,
            "dynamic_flatness": 0.0,
            "artifact_spike_ratio": 0.0,
            "fatigue_risk": 0.0,
            "bass_anchor_score": 1.0,
            "dynamic_breath_score": 1.0,
            "stereo_depth_score": 1.0,
            "dark_balance_score": 1.0,
            "lowmid_boxiness_risk": 0.0,
            "reference_dna_score": 1.0,
        }
        good = score_metrics(base, target_lufs=-19.0, true_peak_ceiling=0.8912509381337456)
        bad = score_metrics(
            {
                **base,
                "bass_anchor_score": 0.0,
                "dynamic_breath_score": 0.0,
                "stereo_depth_score": 0.0,
                "lowmid_boxiness_risk": 1.0,
                "reference_dna_score": 0.0,
            },
            target_lufs=-19.0,
            true_peak_ceiling=0.8912509381337456,
        )

        self.assertLess(bad["global_score"], good["global_score"])


if __name__ == "__main__":
    unittest.main()
