from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from ambient_engine.arrangement.transitions import assemble_stem_sequence, cosine_crossfade


class TransitionTests(unittest.TestCase):
    def test_assembled_length_avoids_duplicate_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_rate = 10
            first = root / "first.wav"
            second = root / "second.wav"
            out = root / "out.wav"
            data_a = np.column_stack([np.ones(10), np.ones(10)]).astype(np.float32)
            data_b = np.column_stack([np.ones(10) * 2, np.ones(10) * 2]).astype(np.float32)
            sf.write(first, data_a, sample_rate)
            sf.write(second, data_b, sample_rate)
            assemble_stem_sequence([first, second], out, sample_rate, 2, crossfade_seconds=0.2, block_frames=4)
            assembled, _ = sf.read(out, dtype="float32", always_2d=True)
            self.assertEqual(len(assembled), 18)

    def test_transition_policy_can_extend_texture_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_rate = 10
            first = root / "first.wav"
            second = root / "second.wav"
            out = root / "out.wav"
            data_a = np.column_stack([np.ones(10), np.ones(10)]).astype(np.float32)
            data_b = np.column_stack([np.ones(10) * 2, np.ones(10) * 2]).astype(np.float32)
            sf.write(first, data_a, sample_rate)
            sf.write(second, data_b, sample_rate)
            assemble_stem_sequence(
                [first, second],
                out,
                sample_rate,
                2,
                crossfade_seconds=0.2,
                transition_policies=["silk_crossfade", "long_blend"],
                stem_name="texture",
                block_frames=4,
            )
            assembled, _ = sf.read(out, dtype="float32", always_2d=True)
            self.assertEqual(len(assembled), 16)

    def test_cosine_crossfade_softens_bright_entry(self) -> None:
        sample_rate = 24000
        t = np.arange(4096, dtype=np.float32) / sample_rate
        previous = 0.18 * np.sin(2 * np.pi * 180.0 * t)
        previous = np.column_stack([previous, previous * 0.98]).astype(np.float32)

        current = 0.28 * np.sin(2 * np.pi * 180.0 * t)
        current += 0.12 * np.sin(2 * np.pi * 7200.0 * t + 0.3)
        current = np.column_stack([current, current]).astype(np.float32)

        curve = np.linspace(0.0, np.pi / 2.0, len(previous), dtype=np.float32)
        baseline = previous * np.cos(curve)[:, None] + current * np.sin(curve)[:, None]
        matched = cosine_crossfade(previous, current, sample_rate=sample_rate)

        self.assertLess(_hf_ratio(matched, sample_rate), _hf_ratio(baseline, sample_rate))


def _hf_ratio(audio: np.ndarray, sample_rate: int) -> float:
    mono = audio.mean(axis=1)
    window = np.hanning(len(mono)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    spectral_sum = float(np.sum(spectrum)) + 1e-9
    return float(np.sum(spectrum[freqs >= 6000.0]) / spectral_sum)


if __name__ == "__main__":
    unittest.main()
