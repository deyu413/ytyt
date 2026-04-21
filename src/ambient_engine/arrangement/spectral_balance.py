from __future__ import annotations

import numpy as np


def stabilize_master_block(
    block: np.ndarray,
    ceiling: float = 0.8912509381337456,
    target_width: float = 0.09,
    max_width: float = 0.24,
    sample_rate: int = 48000,
    target_hf_ratio: float = 0.085,
    target_presence_ratio: float = 0.14,
) -> np.ndarray:
    if len(block) == 0:
        return block
    centered = block - np.mean(block, axis=0, keepdims=True)
    if centered.shape[1] == 2:
        mid = (centered[:, :1] + centered[:, 1:2]) * 0.5
        side = (centered[:, :1] - centered[:, 1:2]) * 0.5
        mid_rms = float(np.sqrt(np.mean(mid ** 2) + 1e-12))
        side_rms = float(np.sqrt(np.mean(side ** 2) + 1e-12))
        width = side_rms / max(mid_rms, 1e-6)
        if width < target_width:
            side *= min(2.8, target_width / max(width, 1e-4))
        elif width > max_width:
            side *= max_width / width
        mid_peak = float(np.max(np.abs(mid)))
        side_peak = float(np.max(np.abs(side)))
        if side_peak > 0 and mid_peak > 0 and side_peak > mid_peak * 1.2:
            side *= (mid_peak * 1.2) / side_peak
        centered = np.concatenate([mid + side, mid - side], axis=1)
    centered = _soft_deharsh(
        centered,
        sample_rate=sample_rate,
        target_hf_ratio=target_hf_ratio,
        target_presence_ratio=target_presence_ratio,
    )
    peak = np.max(np.abs(centered))
    if peak > ceiling and peak > 0:
        centered = np.tanh(centered / peak * 1.15) * ceiling
    return centered.astype(np.float32)


def db_to_linear(value_db: float) -> float:
    return float(10 ** (value_db / 20.0))


def _soft_deharsh(block: np.ndarray, sample_rate: int, target_hf_ratio: float, target_presence_ratio: float) -> np.ndarray:
    if len(block) < 256:
        return block.astype(np.float32)
    centered = block.astype(np.float32)
    mono = centered.mean(axis=1)
    hf_ratio = _hf_ratio(mono, sample_rate)
    if hf_ratio <= target_hf_ratio:
        softened = centered
    else:
        intensity = float(np.clip((hf_ratio - target_hf_ratio) / 0.12, 0.0, 1.0))
        cutoff_hz = float(np.clip(6800.0 - intensity * 2400.0, 3800.0, 6800.0))
        low = _lowpass_channels(centered, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
        high = centered - low
        softened = low + high * (1.0 - 0.42 * intensity)
    presence_ratio = _presence_ratio(softened.mean(axis=1), sample_rate)
    if presence_ratio <= target_presence_ratio:
        return softened.astype(np.float32)
    intensity = float(np.clip((presence_ratio - target_presence_ratio) / 0.24, 0.0, 1.0))
    cutoff_hz = float(np.clip(4700.0 - intensity * 2100.0, 2300.0, 4700.0))
    low = _lowpass_channels(softened, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    high = softened - low
    return (low + high * (1.0 - 0.56 * intensity)).astype(np.float32)


def _hf_ratio(signal: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(len(signal)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    spectral_sum = float(np.sum(spectrum)) + 1e-9
    hf_mask = freqs >= 6000.0
    if not np.any(hf_mask):
        return 0.0
    return float(np.sum(spectrum[hf_mask]) / spectral_sum)


def _presence_ratio(signal: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(len(signal)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    presence_mask = (freqs >= 1000.0) & (freqs < 4000.0)
    spectral_sum = float(np.sum(spectrum)) + 1e-9
    presence = float(np.sum(spectrum[presence_mask]))
    return presence / spectral_sum


def _lowpass_channels(block: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if block.ndim == 1:
        return _one_pole_lowpass(block, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    channels = [_one_pole_lowpass(block[:, channel], sample_rate=sample_rate, cutoff_hz=cutoff_hz) for channel in range(block.shape[1])]
    return np.column_stack(channels).astype(np.float32)


def _one_pole_lowpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if len(signal) == 0:
        return signal.astype(np.float32)
    alpha = float(np.exp(-2.0 * np.pi * cutoff_hz / max(1.0, sample_rate)))
    filtered = np.empty_like(signal, dtype=np.float32)
    filtered[0] = np.float32(signal[0])
    mix = np.float32(1.0 - alpha)
    decay = np.float32(alpha)
    for index in range(1, len(signal)):
        filtered[index] = mix * np.float32(signal[index]) + decay * filtered[index - 1]
    return filtered.astype(np.float32)
