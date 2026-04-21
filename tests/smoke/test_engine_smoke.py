from pathlib import Path
import shutil
import tempfile
import unittest

from ambient_engine.app import AmbientEngine


class EngineSmokeTests(unittest.TestCase):
    def test_dry_run_render_creates_manifest(self) -> None:
        root = Path(__file__).resolve().parents[2]
        engine = AmbientEngine(root)
        session_id = "smoke_dry_run_afterblue"
        try:
            result = engine.render(
                profile_id="afterblue_sleep",
                target_length="2h",
                runtime_mode="cpu-safe",
                seed=42,
                dry_run=True,
                session_id=session_id,
            )
            manifest_path = root / "sessions" / session_id / "manifests" / "session_manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(result["dry_run"])
        finally:
            shutil.rmtree(root / "sessions" / session_id, ignore_errors=True)

    def test_publish_dry_run_on_fixture_session(self) -> None:
        root = Path(__file__).resolve().parents[2]
        engine = AmbientEngine(root)
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir)
            (session / "exports").mkdir(parents=True, exist_ok=True)
            (session / "manifests").mkdir(parents=True, exist_ok=True)
            (session / "exports" / "hud_video.mp4").write_bytes(b"video")
            (session / "exports" / "thumbnail.png").write_bytes(b"png")
            (session / "manifests" / "metadata.json").write_text(
                '{"title": "Smoke Test", "description": "ok", "tags": ["ambient"], "language": "en"}',
                encoding="utf-8",
            )
            result = engine.publish(session, dry_run=True)
            self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()

