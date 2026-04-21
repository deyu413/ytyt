import unittest

import numpy as np

from ambient_engine.arrangement.spectral_balance import stabilize_master_block


class SpectralBalanceTests(unittest.TestCase):
    def test_stabilize_master_block_expands_narrow_stereo_toward_target(self) -> None:
        t = np.arange(2400, dtype=np.float32) / 400.0
        base = 0.16 * np.sin(2 * np.pi * 48.0 * t)
        decor = 0.012 * np.sin(2 * np.pi * 83.0 * t + 0.5)
        block = np.column_stack([base + decor, base - 0.2 * decor]).astype(np.float32)

        before_mid = 0.5 * (block[:, 0] + block[:, 1])
        before_side = 0.5 * (block[:, 0] - block[:, 1])
        before_width = float(np.sqrt(np.mean(before_side ** 2)) / max(1e-6, np.sqrt(np.mean(before_mid ** 2))))

        widened = stabilize_master_block(block, target_width=0.09, max_width=0.24)
        after_mid = 0.5 * (widened[:, 0] + widened[:, 1])
        after_side = 0.5 * (widened[:, 0] - widened[:, 1])
        after_width = float(np.sqrt(np.mean(after_side ** 2)) / max(1e-6, np.sqrt(np.mean(after_mid ** 2))))

        self.assertLess(before_width, 0.05)
        self.assertGreater(after_width, before_width)
        self.assertLessEqual(after_width, 0.24)

    def test_stabilize_master_block_reduces_harsh_high_band(self) -> None:
        sample_rate = 24000
        t = np.arange(4096, dtype=np.float32) / sample_rate
        base = 0.16 * np.sin(2 * np.pi * 220.0 * t)
        bright = 0.14 * np.sin(2 * np.pi * 7600.0 * t + 0.4)
        block = np.column_stack([base + bright, base + bright * 0.92]).astype(np.float32)

        before_ratio = _hf_ratio(block, sample_rate)
        softened = stabilize_master_block(block, sample_rate=sample_rate, target_hf_ratio=0.07)
        after_ratio = _hf_ratio(softened, sample_rate)

        self.assertGreater(before_ratio, 0.12)
        self.assertLess(after_ratio, before_ratio)

    def test_stabilize_master_block_tames_presence_band(self) -> None:
        sample_rate = 24000
        t = np.arange(8192, dtype=np.float32) / sample_rate
        base = 0.17 * np.sin(2 * np.pi * 220.0 * t)
        presence = 0.08 * np.sin(2 * np.pi * 1850.0 * t)
        block = np.column_stack([base + presence, base + 0.94 * presence]).astype(np.float32)

        before_ratio = _presence_ratio(block, sample_rate)
        softened = stabilize_master_block(block, sample_rate=sample_rate, target_presence_ratio=0.10)
        after_ratio = _presence_ratio(softened, sample_rate)

        self.assertGreater(before_ratio, 0.12)
        self.assertLess(after_ratio, before_ratio)


def _hf_ratio(block: np.ndarray, sample_rate: int) -> float:
    mono = block.mean(axis=1)
    window = np.hanning(len(mono)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    spectral_sum = float(np.sum(spectrum)) + 1e-9
    return float(np.sum(spectrum[freqs >= 6000.0]) / spectral_sum)


def _presence_ratio(block: np.ndarray, sample_rate: int) -> float:
    mono = block.mean(axis=1)
    window = np.hanning(len(mono)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    spectral_sum = float(np.sum(spectrum)) + 1e-9
    presence = float(np.sum(spectrum[(freqs >= 1000.0) & (freqs < 4000.0)]))
    return presence / spectral_sum


if __name__ == "__main__":
    unittest.main()
