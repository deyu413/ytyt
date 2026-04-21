from pathlib import Path
import unittest

from ambient_engine.core.paths import ProjectPaths
from ambient_engine.core.runtime import detect_runtime
from ambient_engine.generation.registry import ProviderRegistry


class ProviderRegistryTests(unittest.TestCase):
    def test_cpu_safe_runtime_falls_back_to_procedural(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runtime = detect_runtime("cpu-safe", ProjectPaths(root))
        registry = ProviderRegistry(runtime)
        selection = registry.select("section:intro", ["ace_step_1_5", "procedural_dsp"])
        self.assertEqual(selection.provider_name, "procedural_dsp")
        self.assertIn("procedural_dsp", selection.fallback_chain)


if __name__ == "__main__":
    unittest.main()

