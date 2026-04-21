from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def cosine_crossfade(previous_tail: np.ndarray, current_head: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    overlap = min(len(previous_tail), len(current_head))
    if overlap <= 0:
        return current_head
    previous_tail = previous_tail[:overlap]
    current_head = current_head[:overlap]
    previous_tail = _center_overlap(previous_tail)
    current_head = _center_overlap(current_head)
    current_head = _match_overlap_level(previous_tail, current_head)
    current_head = _match_overlap_width(previous_tail, current_head)
    current_head = _match_overlap_tone(previous_tail, current_head, sample_rate=sample_rate)
    curve = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    fade_out = np.cos(curve)
    fade_in = np.sin(curve)
    if previous_tail.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in = fade_in[:, None]
    return previous_tail * fade_out + current_head * fade_in


def assemble_stem_sequence(
    input_paths: list[Path],
    output_path: Path,
    sample_rate: int,
    channels: int,
    crossfade_seconds: float,
    transition_policies: list[str] | None = None,
    stem_name: str | None = None,
    block_frames: int = 65536,
) -> list[int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_frames: list[int] = []
    carry: np.ndarray | None = None
    total_written = 0

    with sf.SoundFile(output_path, mode="w", samplerate=sample_rate, channels=channels, subtype="PCM_24") as writer:
        for index, path in enumerate(input_paths):
            with sf.SoundFile(path, mode="r") as section_reader:
                frames = len(section_reader)
                if frames <= 0:
                    continue
                crossfade_frames = _transition_crossfade_frames(
                    sample_rate=sample_rate,
                    crossfade_seconds=crossfade_seconds,
                    frames=frames,
                    transition_policy=_resolve_transition_policy(transition_policies, index),
                    stem_name=stem_name,
                )
                overlap = min(crossfade_frames, max(0, frames // 2))
                if overlap == 0 and carry is not None:
                    writer.write(carry)
                    total_written += len(carry)
                    carry = None

                if index == 0:
                    write_frames = max(0, frames - overlap)
                    _copy_range(section_reader, writer, 0, write_frames, block_frames)
                    total_written += write_frames
                    carry = _read_range(section_reader, frames - overlap, overlap) if overlap else None
                    continue

                head = _read_range(section_reader, 0, overlap) if overlap else np.zeros((0, channels), dtype=np.float32)
                if carry is not None:
                    transition = cosine_crossfade(carry, head, sample_rate=sample_rate)
                    boundary_frames.append(total_written + len(transition) // 2)
                    writer.write(transition)
                    total_written += len(transition)

                body_start = overlap
                body_frames = max(0, frames - overlap - body_start)
                _copy_range(section_reader, writer, body_start, body_frames, block_frames)
                total_written += body_frames
                carry = _read_range(section_reader, max(body_start, frames - overlap), min(overlap, frames)) if overlap else None

        if carry is not None and len(carry):
            writer.write(carry)
            total_written += len(carry)

    return boundary_frames


def _copy_range(
    reader: sf.SoundFile,
    writer: sf.SoundFile,
    start_frame: int,
    frame_count: int,
    block_frames: int,
) -> None:
    remaining = frame_count
    reader.seek(start_frame)
    while remaining > 0:
        chunk = reader.read(min(block_frames, remaining), dtype="float32", always_2d=True)
        if len(chunk) == 0:
            break
        writer.write(chunk)
        remaining -= len(chunk)


def _read_range(reader: sf.SoundFile, start_frame: int, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros((0, reader.channels), dtype=np.float32)
    reader.seek(start_frame)
    return reader.read(frame_count, dtype="float32", always_2d=True)


def _resolve_transition_policy(transition_policies: list[str] | None, index: int) -> str:
    if not transition_policies:
        return "silk_crossfade"
    if index < len(transition_policies):
        return transition_policies[index]
    return transition_policies[-1]


def _transition_crossfade_frames(
    sample_rate: int,
    crossfade_seconds: float,
    frames: int,
    transition_policy: str,
    stem_name: str | None,
) -> int:
    policy_scale = {
        "silk_crossfade": 1.0,
        "long_blend": 1.35,
        "dissolve": 1.18,
        "vanish": 1.12,
    }.get(transition_policy, 1.0)
    stem_scale = {
        "drone": 1.16,
        "motion": 1.04,
        "texture": 1.34,
        "accents": 0.92,
        "rhythm": 0.84,
    }.get(stem_name or "", 1.0)
    scaled = int(round(sample_rate * crossfade_seconds * policy_scale * stem_scale))
    return min(max(0, scaled), max(0, frames // 2))


def _match_overlap_level(previous_tail: np.ndarray, current_head: np.ndarray) -> np.ndarray:
    previous_rms = float(np.sqrt(np.mean(previous_tail ** 2) + 1e-12))
    current_rms = float(np.sqrt(np.mean(current_head ** 2) + 1e-12))
    if previous_rms <= 1e-6 or current_rms <= 1e-6:
        return current_head
    gain = np.clip(previous_rms / current_rms, 0.76, 1.32)
    gain_curve = np.linspace(gain, 1.0, len(current_head), dtype=np.float32)
    if current_head.ndim == 2:
        gain_curve = gain_curve[:, None]
    return current_head * gain_curve


def _center_overlap(audio: np.ndarray) -> np.ndarray:
    return (audio - np.mean(audio, axis=0, keepdims=True)).astype(np.float32)


def _match_overlap_width(previous_tail: np.ndarray, current_head: np.ndarray) -> np.ndarray:
    if previous_tail.ndim != 2 or previous_tail.shape[1] < 2 or current_head.ndim != 2 or current_head.shape[1] < 2:
        return current_head
    previous_width = _stereo_width(previous_tail)
    current_width = _stereo_width(current_head)
    if previous_width <= 1e-5 or current_width <= 1e-5:
        return current_head
    width_gain = np.clip(previous_width / current_width, 0.78, 1.32)
    width_curve = np.linspace(width_gain, 1.0, len(current_head), dtype=np.float32)[:, None]
    mid = 0.5 * (current_head[:, :1] + current_head[:, 1:2])
    side = 0.5 * (current_head[:, :1] - current_head[:, 1:2]) * width_curve
    return np.concatenate([mid + side, mid - side], axis=1).astype(np.float32)


def _match_overlap_tone(previous_tail: np.ndarray, current_head: np.ndarray, sample_rate: int) -> np.ndarray:
    previous_hf = _hf_ratio(previous_tail, sample_rate)
    current_hf = _hf_ratio(current_head, sample_rate)
    if current_hf <= previous_hf + 0.006:
        return current_head
    intensity = float(np.clip((current_hf - previous_hf - 0.006) / 0.12, 0.0, 1.0))
    cutoff_hz = float(np.clip(7200.0 - 2600.0 * intensity, 4200.0, 7200.0))
    softened = _lowpass_channels(current_head, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    blend = np.linspace(1.0, 0.0, len(current_head), dtype=np.float32)
    if current_head.ndim == 2:
        blend = blend[:, None]
    return (softened * blend + current_head * (1.0 - blend)).astype(np.float32)


def _stereo_width(audio: np.ndarray) -> float:
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    return float(np.sqrt(np.mean(side ** 2) + 1e-12) / max(1e-6, np.sqrt(np.mean(mid ** 2) + 1e-12)))


def _hf_ratio(audio: np.ndarray, sample_rate: int) -> float:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    window = np.hanning(len(mono)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    spectral_sum = float(np.sum(spectrum)) + 1e-9
    hf_mask = freqs >= 6000.0
    if not np.any(hf_mask):
        return 0.0
    return float(np.sum(spectrum[hf_mask]) / spectral_sum)


def _lowpass_channels(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if audio.ndim == 1:
        return _one_pole_lowpass(audio, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    filtered = [_one_pole_lowpass(audio[:, channel], sample_rate=sample_rate, cutoff_hz=cutoff_hz) for channel in range(audio.shape[1])]
    return np.column_stack(filtered).astype(np.float32)


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
