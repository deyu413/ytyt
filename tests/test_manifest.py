from pathlib import Path
import tempfile
import unittest

from ambient_engine.core.manifest import SessionManifest


class ManifestTests(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        manifest = SessionManifest(session_id="session", profile_id="afterblue_sleep", runtime_mode="cpu-safe", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest.save(path)
            loaded = SessionManifest.load(path)
            self.assertEqual(loaded.session_id, "session")
            self.assertEqual(loaded.profile_id, "afterblue_sleep")
            self.assertEqual(loaded.seed, 42)


if __name__ == "__main__":
    unittest.main()

