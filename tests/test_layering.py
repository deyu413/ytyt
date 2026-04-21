from pathlib import Path
import unittest

from ambient_engine.arrangement.layering import resolve_stem_levels
from ambient_engine.profiles.loader import load_profile_by_id


class LayeringTests(unittest.TestCase):
    def test_profile_can_override_rhythm_stem_level(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "quiet_night_focus")
        levels = resolve_stem_levels(profile)
        self.assertIn("rhythm", levels)
        self.assertAlmostEqual(levels["rhythm"], -10.8)


if __name__ == "__main__":
    unittest.main()
