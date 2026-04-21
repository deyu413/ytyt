from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from ambient_engine.profiles.loader import load_profile_by_id
from ambient_engine.render.shorts import build_short_concepts, select_highlight_excerpt


class ShortsTests(unittest.TestCase):
    def test_afterblue_short_concepts_follow_expected_editorial_shape(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "afterblue_sleep")
        metadata = {"title": "Afterblue Sleep | 1h", "tags": ["afterblue"]}

        concepts = build_short_concepts(profile, metadata)

        self.assertEqual(len(concepts), 3)
        self.assertEqual(concepts[0].kind, "emotional_hook")
        self.assertIn("let your mind rest", concepts[0].primary_text)
        self.assertEqual(concepts[1].kind, "mood_hook")
        self.assertEqual(concepts[2].kind, "longform_teaser")

    def test_highlight_selector_prefers_mid_track_energy_rise(self) -> None:
        sample_rate = 1600
        seconds = 80
        t = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
        audio = 0.004 * np.sin(2 * np.pi * 48.0 * t)
        center_start = 34 * sample_rate
        center_end = 56 * sample_rate
        audio[center_start:center_end] += 0.26 * np.sin(2 * np.pi * 92.0 * t[: center_end - center_start])
        audio[center_start:center_end] += 0.11 * np.sin(2 * np.pi * 150.0 * t[: center_end - center_start] + 0.4)

        section_plan = [
            {"role": "intro", "duration_seconds": 12, "density": 0.18},
            {"role": "drift_a", "duration_seconds": 22, "density": 0.32},
            {"role": "drift_b", "duration_seconds": 24, "density": 0.42},
            {"role": "return", "duration_seconds": 22, "density": 0.28},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "highlight.wav"
            sf.write(audio_path, np.column_stack([audio, audio]).astype(np.float32), sample_rate)
            highlight = select_highlight_excerpt(audio_path, sample_rate=sample_rate, section_plan=section_plan, total_seconds=seconds, window_seconds=16)

        self.assertGreaterEqual(int(highlight["start_seconds"]), 28)
        self.assertLessEqual(int(highlight["start_seconds"]), 42)

    def test_calm_hook_selector_avoids_bright_peak_for_afterblue(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_profile_by_id(root / "profiles", "afterblue_sleep")
        sample_rate = 1600
        seconds = 90
        t = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
        audio = 0.01 * np.sin(2 * np.pi * 42.0 * t)

        calm_start = 18 * sample_rate
        calm_end = 42 * sample_rate
        audio[calm_start:calm_end] += 0.09 * np.sin(2 * np.pi * 74.0 * t[: calm_end - calm_start])

        bright_start = 56 * sample_rate
        bright_end = 76 * sample_rate
        audio[bright_start:bright_end] += 0.20 * np.sin(2 * np.pi * 108.0 * t[: bright_end - bright_start])
        audio[bright_start:bright_end] += 0.10 * np.sin(2 * np.pi * 6800.0 * t[: bright_end - bright_start])

        section_plan = [
            {"role": "intro", "duration_seconds": 12, "density": 0.18},
            {"role": "settle", "duration_seconds": 22, "density": 0.24},
            {"role": "drift_a", "duration_seconds": 20, "density": 0.28},
            {"role": "drift_b", "duration_seconds": 20, "density": 0.31},
            {"role": "return", "duration_seconds": 16, "density": 0.25},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "calm_hook.wav"
            sf.write(audio_path, np.column_stack([audio, audio]).astype(np.float32), sample_rate)
            highlight = select_highlight_excerpt(
                audio_path,
                sample_rate=sample_rate,
                section_plan=section_plan,
                total_seconds=seconds,
                window_seconds=16,
                profile=profile,
            )

        self.assertGreaterEqual(int(highlight["start_seconds"]), 14)
        self.assertLessEqual(int(highlight["start_seconds"]), 34)


if __name__ == "__main__":
    unittest.main()
