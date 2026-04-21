from pathlib import Path
import unittest

from ambient_engine.profiles.loader import load_profile_by_id


class ProfileLoaderTests(unittest.TestCase):
    def test_load_afterblue_sleep_profile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "afterblue_sleep")
        self.assertEqual(profile.profile_id, "afterblue_sleep")
        self.assertEqual(profile.language, "en")
        self.assertGreater(len(profile.section_schema), 5)
        self.assertIn("obvious looping", profile.forbidden_artifacts)

    def test_load_afterblue_reference_sleep_profile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "afterblue_reference_sleep")
        self.assertEqual(profile.profile_id, "afterblue_reference_sleep")
        self.assertEqual(profile.tonal_center["root"], "C#")
        self.assertEqual(profile.scale_family, "afterblue_minor_cluster")
        self.assertEqual(profile.instrumentation["production_dna"], "reference_dark_bass")
        self.assertEqual(profile.loudness_target_lufs, -19.0)


if __name__ == "__main__":
    unittest.main()
