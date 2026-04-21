from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from ambient_engine.qc.analyzers import analyze_audio


class QcAnalyzerTests(unittest.TestCase):
    def test_repetition_metric_separates_loop_from_evolving_material(self) -> None:
        sample_rate = 400
        seconds = 8
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate

        repeated_block = 0.18 * np.sin(2 * np.pi * 48.0 * t)
        repeated = np.tile(repeated_block, seconds)

        evolving_blocks = []
        for index, freq in enumerate([40.0, 45.0, 52.0, 60.0, 68.0, 79.0, 92.0, 108.0]):
            amp = 0.14 + index * 0.01
            block = amp * np.sin(2 * np.pi * freq * t + index * 0.18)
            block += 0.015 * np.sin(2 * np.pi * (freq * 0.5) * t)
            evolving_blocks.append(block.astype(np.float32))
        evolving = np.concatenate(evolving_blocks)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repeated_path = root / "repeated.wav"
            evolving_path = root / "evolving.wav"
            sf.write(repeated_path, np.column_stack([repeated, repeated]).astype(np.float32), sample_rate)
            sf.write(evolving_path, np.column_stack([evolving, evolving]).astype(np.float32), sample_rate)

            repeated_metrics = analyze_audio(repeated_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)
            evolving_metrics = analyze_audio(evolving_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)

        self.assertGreater(float(repeated_metrics["repetition_score"]), 0.45)
        self.assertLess(float(evolving_metrics["repetition_score"]), float(repeated_metrics["repetition_score"]))
        self.assertGreater(
            float(repeated_metrics["repetition_score"]) - float(evolving_metrics["repetition_score"]),
            0.35,
        )

    def test_mono_collapse_metric_penalizes_narrow_material(self) -> None:
        sample_rate = 400
        seconds = 8
        t = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate

        base = 0.18 * np.sin(2 * np.pi * 52.0 * t)
        mono = np.column_stack([base, base]).astype(np.float32)

        decor = 0.08 * np.sin(2 * np.pi * 71.0 * t + 0.7)
        wide_left = base + decor
        wide_right = base - 0.85 * decor
        wide = np.column_stack([wide_left, wide_right]).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mono_path = root / "mono.wav"
            wide_path = root / "wide.wav"
            sf.write(mono_path, mono, sample_rate)
            sf.write(wide_path, wide, sample_rate)

            mono_metrics = analyze_audio(mono_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)
            wide_metrics = analyze_audio(wide_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)

        self.assertGreater(float(mono_metrics["mono_collapse_risk"]), 0.9)
        self.assertLess(float(wide_metrics["mono_collapse_risk"]), float(mono_metrics["mono_collapse_risk"]))
        self.assertGreater(float(wide_metrics["stereo_width_mean"]), float(mono_metrics["stereo_width_mean"]))

    def test_silence_metric_ignores_soft_edges_but_flags_mid_track_dropouts(self) -> None:
        sample_rate = 1600
        block = np.arange(sample_rate, dtype=np.float32) / sample_rate

        quiet = np.zeros_like(block)
        active = 0.16 * np.sin(2 * np.pi * 48.0 * block)
        edge_safe = np.concatenate([quiet, active, active, active, active, quiet]).astype(np.float32)
        mid_dropout = np.concatenate([active, active, quiet, quiet, active, active]).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            edge_path = root / "edge_safe.wav"
            dropout_path = root / "dropout.wav"
            sf.write(edge_path, np.column_stack([edge_safe, edge_safe]).astype(np.float32), sample_rate)
            sf.write(dropout_path, np.column_stack([mid_dropout, mid_dropout]).astype(np.float32), sample_rate)

            edge_metrics = analyze_audio(edge_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)
            dropout_metrics = analyze_audio(dropout_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)

        self.assertEqual(float(edge_metrics["silence_ratio"]), 0.0)
        self.assertGreater(float(dropout_metrics["silence_ratio"]), 0.3)

    def test_interference_metric_penalizes_presence_heavy_beating_signal(self) -> None:
        sample_rate = 4000
        seconds = 10
        t = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate

        calm = 0.16 * np.sin(2 * np.pi * 90.0 * t)
        bad = calm.copy()
        bad += 0.06 * np.sin(2 * np.pi * 1650.0 * t) * (0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * t))
        bad += 0.05 * np.sin(2 * np.pi * 1820.0 * t + 0.2) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t + 0.4))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calm_path = root / "calm.wav"
            bad_path = root / "bad.wav"
            sf.write(calm_path, np.column_stack([calm, calm]).astype(np.float32), sample_rate)
            sf.write(bad_path, np.column_stack([bad, bad]).astype(np.float32), sample_rate)

            calm_metrics = analyze_audio(calm_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)
            bad_metrics = analyze_audio(bad_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=1)

        self.assertLess(float(calm_metrics["interference_risk"]), 0.2)
        self.assertGreater(float(bad_metrics["interference_risk"]), float(calm_metrics["interference_risk"]))
        self.assertGreater(float(bad_metrics["harshness"]), float(calm_metrics["harshness"]))

    def test_reference_dna_metrics_penalize_boxy_lowmid_without_bass_anchor(self) -> None:
        sample_rate = 1600
        seconds = 12
        t = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate

        bass_breath = 0.18 * np.sin(2 * np.pi * 80.0 * t)
        bass_breath *= 0.55 + 0.45 * np.sin(2 * np.pi * 0.08 * t)
        bass_breath += 0.08 * np.sin(2 * np.pi * 105.0 * t + 0.4)
        wide = np.column_stack([bass_breath, bass_breath * 0.94 + 0.04 * np.sin(2 * np.pi * 87.0 * t)]).astype(np.float32)

        boxy = 0.18 * np.sin(2 * np.pi * 190.0 * t)
        boxy += 0.05 * np.sin(2 * np.pi * 225.0 * t)
        narrow = np.column_stack([boxy, boxy]).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_like_path = root / "reference_like.wav"
            boxy_path = root / "boxy.wav"
            sf.write(reference_like_path, wide, sample_rate)
            sf.write(boxy_path, narrow, sample_rate)

            reference_like = analyze_audio(reference_like_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=2)
            boxy_metrics = analyze_audio(boxy_path, sample_rate=sample_rate, boundary_frames=[], block_seconds=2)

        self.assertGreater(float(reference_like["bass_anchor_score"]), float(boxy_metrics["bass_anchor_score"]))
        self.assertGreater(float(boxy_metrics["lowmid_boxiness_risk"]), float(reference_like["lowmid_boxiness_risk"]))
        self.assertGreater(float(reference_like["reference_dna_score"]), float(boxy_metrics["reference_dna_score"]))


if __name__ == "__main__":
    unittest.main()
