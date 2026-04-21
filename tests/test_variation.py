from pathlib import Path
import unittest

from ambient_engine.planning.variation import VariationPlanner
from ambient_engine.profiles.loader import load_profile_by_id


class VariationPlannerTests(unittest.TestCase):
    def test_afterblue_sleep_uses_sleep_safe_variation_space(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "afterblue_sleep")
        variation = VariationPlanner(profile, seed=42).build()

        self.assertIn(variation["texture_variant"], {"room-hush", "velvet-room"})
        self.assertIn(variation["harmonic_color"], {"root-fifth", "modal-third"})
        self.assertIn(variation["stereo_profile"], {"narrow-core", "balanced"})
        self.assertLessEqual(float(variation["movement_bias"]), 0.44)
        self.assertLessEqual(float(variation["motif_density"]), 0.46)

    def test_afterblue_reference_sleep_uses_long_breath_variation_space(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "afterblue_reference_sleep")
        variation = VariationPlanner(profile, seed=42).build()

        self.assertEqual(variation["pacing_variant"], "long-breath")
        self.assertIn(variation["texture_variant"], {"sub-room", "black-room"})
        self.assertEqual(variation["rhythm_variant"], "breathing-swell")
        self.assertIn(variation["stereo_profile"], {"balanced", "wide-tail"})


if __name__ == "__main__":
    unittest.main()
