from pathlib import Path
import unittest

from ambient_engine.planning.macrostructure import MacrostructurePlanner
from ambient_engine.profiles.loader import load_profile_by_id


class MacrostructurePlannerTests(unittest.TestCase):
    def test_target_duration_is_preserved(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "blue_hour_rest")
        planner = MacrostructurePlanner(profile, seed=42)
        plan = planner.build(3600)
        self.assertEqual(sum(section.duration_seconds for section in plan), 3600)
        self.assertEqual(plan[0].role, "intro")
        self.assertEqual(plan[-1].role, "low_energy_tail")


if __name__ == "__main__":
    unittest.main()

