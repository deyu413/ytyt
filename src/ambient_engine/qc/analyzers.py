from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_audio(master_path: Path, sample_rate: int, boundary_frames: list[int], block_seconds: int = 15) -> dict[str, float | int | list[float]]:
    block_frames = max(1024, sample_rate * block_seconds)
    rms_db_values: list[float] = []
    spectral_centroids: list[float] = []
    hf_ratios: list[float] = []
    sub_shares: list[float] = []
    bass_shares: list[float] = []
    lowmid_shares: list[float] = []
    body_shares: list[float] = []
    presence_shares: list[float] = []
    stereo_corrs: list[float] = []
    stereo_widths: list[float] = []
    feature_vectors: list[np.ndarray] = []
    total_samples = 0
    sumsq = 0.0
    peak = 0.0
    clipped_samples = 0
    artifact_spikes = 0
    previous_spectrum: np.ndarray | None = None

    with sf.SoundFile(master_path, mode="r") as reader:
        while True:
            block = reader.read(block_frames, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            mono = block.mean(axis=1)
            rms = float(np.sqrt(np.mean(block ** 2) + 1e-12))
            rms_db = 20.0 * np.log10(max(rms, 1e-9))
            peak = max(peak, float(np.max(np.abs(block))))
            clipped_samples += int(np.sum(np.abs(block) >= 0.999))
            total_samples += int(block.size)
            sumsq += float(np.sum(block ** 2))
            rms_db_values.append(rms_db)

            spectrum, freqs = _averaged_spectrum(mono, sample_rate)
            if len(spectrum) < 32:
                continue
            spectral_sum = float(np.sum(spectrum)) + 1e-9
            centroid = float(np.sum(freqs * spectrum) / spectral_sum)
            sub_mask = (freqs >= 20.0) & (freqs < 60.0)
            bass_mask = (freqs >= 60.0) & (freqs < 120.0)
            musical_lowmid_mask = (freqs >= 120.0) & (freqs < 300.0)
            body_mask = (freqs >= 300.0) & (freqs < 800.0)
            hf_mask = freqs >= 6000.0
            sub_share = float(np.sum(spectrum[sub_mask]) / spectral_sum) if np.any(sub_mask) else 0.0
            bass_share = float(np.sum(spectrum[bass_mask]) / spectral_sum) if np.any(bass_mask) else 0.0
            musical_lowmid_share = float(np.sum(spectrum[musical_lowmid_mask]) / spectral_sum) if np.any(musical_lowmid_mask) else 0.0
            body_share = float(np.sum(spectrum[body_mask]) / spectral_sum) if np.any(body_mask) else 0.0
            hf_ratio = float(np.sum(spectrum[hf_mask]) / spectral_sum) if np.any(hf_mask) else 0.0
            low_mask = freqs <= 220.0
            low_ratio = float(np.sum(spectrum[low_mask]) / spectral_sum) if np.any(low_mask) else 0.0
            lowmid_mask = (freqs >= 250.0) & (freqs < 1000.0)
            presence_mask = (freqs >= 1000.0) & (freqs < 4000.0)
            lowmid_ratio = float(np.sum(spectrum[lowmid_mask]) / spectral_sum) if np.any(lowmid_mask) else 0.0
            presence_share = float(np.sum(spectrum[presence_mask]) / spectral_sum) if np.any(presence_mask) else 0.0
            spectral_centroids.append(centroid)
            hf_ratios.append(hf_ratio)
            sub_shares.append(sub_share)
            bass_shares.append(bass_share)
            lowmid_shares.append(musical_lowmid_share)
            body_shares.append(body_share)
            presence_shares.append(presence_share)

            left = block[:, 0]
            right = block[:, 1] if block.shape[1] > 1 else left
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)
            if np.std(left) > 1e-6 and np.std(right) > 1e-6:
                stereo_corrs.append(float(np.corrcoef(left, right)[0, 1]))
            else:
                stereo_corrs.append(1.0)
            stereo_widths.append(float(np.sqrt(np.mean(side ** 2)) / max(1e-6, np.sqrt(np.mean(mid ** 2)))))

            crest = float(np.max(np.abs(mono)) / max(1e-6, np.sqrt(np.mean(mono ** 2))))
            zero_cross = float(np.mean(np.abs(np.diff(np.signbit(mono)).astype(np.float32))))
            flux = _spectral_flux(previous_spectrum, spectrum)
            previous_spectrum = spectrum
            feature_vector = np.array(
                [
                    rms_db / 36.0,
                    centroid / 4200.0,
                    hf_ratio * 5.0,
                    low_ratio * 4.0,
                    lowmid_ratio * 6.0,
                    min(1.0, presence_share / 0.24),
                    stereo_widths[-1] * 4.0,
                    flux * 3.0,
                    crest / 9.0,
                    zero_cross * 8.0,
                ],
                dtype=np.float32,
            )
            feature_vectors.append(feature_vector)

            slew = float(np.percentile(np.abs(np.diff(mono)), 99.7)) if len(mono) > 1 else 0.0
            if slew > 0.17 and hf_ratio > 0.12 and rms_db > -36.0:
                artifact_spikes += 1

    integrated_loudness = 20.0 * np.log10(max(np.sqrt(sumsq / max(1, total_samples)), 1e-9))
    repetition_score = _compute_repetition(feature_vectors)
    spectral_monotony = _compute_spectral_monotony(spectral_centroids)
    hf_harshness = float(np.clip(np.percentile(hf_ratios, 88) if hf_ratios else 0.0, 0.0, 1.0))
    presence_p88 = float(np.percentile(presence_shares, 88) if presence_shares else 0.0)
    presence_instability = _presence_instability(presence_shares)
    presence_excess = float(np.clip((presence_p88 - 0.09) / 0.09, 0.0, 1.0))
    harshness = float(np.clip(max(hf_harshness, 0.55 * presence_excess), 0.0, 1.0))
    interference_risk = float(np.clip(0.9 * presence_excess + 0.1 * presence_instability, 0.0, 1.0))
    mono_collapse_risk = _compute_mono_collapse_risk(stereo_corrs, stereo_widths)
    dynamic_flatness = _compute_dynamic_flatness(rms_db_values)
    dynamic_breath_score = _compute_dynamic_breath_score(rms_db_values)
    bass_anchor_score = _compute_bass_anchor_score(sub_shares, bass_shares, lowmid_shares)
    lowmid_boxiness_risk = _compute_lowmid_boxiness_risk(bass_shares, lowmid_shares, body_shares)
    stereo_depth_score = _compute_stereo_depth_score(stereo_widths)
    dark_balance_score = _compute_dark_balance_score(spectral_centroids, hf_ratios, presence_shares)
    reference_dna_score = float(
        np.clip(
            0.30 * bass_anchor_score
            + 0.24 * dynamic_breath_score
            + 0.18 * stereo_depth_score
            + 0.18 * dark_balance_score
            + 0.10 * (1.0 - lowmid_boxiness_risk),
            0.0,
            1.0,
        )
    )
    silence_ratio = _compute_silence_issue_ratio(rms_db_values)
    section_boundary_smoothness = _boundary_smoothness(master_path, sample_rate, boundary_frames)
    artifact_spike_ratio = artifact_spikes / max(1, len(rms_db_values))
    fatigue_risk = float(
        np.clip(
            0.28 * repetition_score
            + 0.18 * spectral_monotony
            + 0.14 * harshness * 3.2
            + 0.18 * interference_risk
            + 0.14 * dynamic_flatness
            + 0.08 * mono_collapse_risk * 2.4
            + 0.12 * artifact_spike_ratio
            + 0.06 * max(0.0, 0.9 - section_boundary_smoothness),
            0.0,
            1.0,
        )
    )

    return {
        "clipping_ratio": clipped_samples / max(1, total_samples),
        "true_peak": peak,
        "integrated_loudness": integrated_loudness,
        "silence_ratio": silence_ratio,
        "repetition_score": repetition_score,
        "section_boundary_smoothness": section_boundary_smoothness,
        "spectral_monotony": spectral_monotony,
        "harshness": harshness,
        "interference_risk": interference_risk,
        "sub_share_mean": float(np.mean(sub_shares)) if sub_shares else 0.0,
        "bass_share_mean": float(np.mean(bass_shares)) if bass_shares else 0.0,
        "lowmid_share_mean": float(np.mean(lowmid_shares)) if lowmid_shares else 0.0,
        "body_share_mean": float(np.mean(body_shares)) if body_shares else 0.0,
        "bass_anchor_score": bass_anchor_score,
        "lowmid_boxiness_risk": lowmid_boxiness_risk,
        "dynamic_breath_score": dynamic_breath_score,
        "stereo_depth_score": stereo_depth_score,
        "dark_balance_score": dark_balance_score,
        "reference_dna_score": reference_dna_score,
        "stereo_correlation_mean": float(np.mean(stereo_corrs)) if stereo_corrs else 1.0,
        "stereo_width_mean": float(np.mean(stereo_widths)) if stereo_widths else 0.0,
        "mono_collapse_risk": mono_collapse_risk,
        "dynamic_flatness": dynamic_flatness,
        "artifact_spike_ratio": artifact_spike_ratio,
        "fatigue_risk": fatigue_risk,
        "analysis_windows": len(rms_db_values),
    }


def _averaged_spectrum(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    excerpt_frames = min(len(signal), max(4096, min(sample_rate * 2, 65536)))
    if excerpt_frames < 32:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    if len(signal) <= excerpt_frames:
        starts = [0]
    else:
        starts = np.linspace(0, len(signal) - excerpt_frames, 3, dtype=int).tolist()
    spectra: list[np.ndarray] = []
    for start in starts:
        excerpt = signal[start : start + excerpt_frames]
        window = np.hanning(len(excerpt)).astype(np.float32)
        spectra.append(np.abs(np.fft.rfft(excerpt * window)).astype(np.float32))
    spectrum = np.mean(np.vstack(spectra), axis=0)
    freqs = np.fft.rfftfreq(excerpt_frames, d=1.0 / sample_rate).astype(np.float32)
    return spectrum, freqs


def _spectral_flux(previous_spectrum: np.ndarray | None, current_spectrum: np.ndarray) -> float:
    if previous_spectrum is None or len(previous_spectrum) != len(current_spectrum):
        return 0.0
    delta = np.maximum(current_spectrum - previous_spectrum, 0.0)
    return float(np.sum(delta) / max(1e-9, np.sum(current_spectrum)))


def _compute_repetition(feature_vectors: list[np.ndarray]) -> float:
    if len(feature_vectors) < 4:
        return 0.0
    matrix = np.vstack(feature_vectors).astype(np.float32)
    movement = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
    stationarity = float(np.clip(1.0 - np.mean(movement) / 0.22, 0.0, 1.0))

    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    scales = np.std(centered, axis=0, keepdims=True)
    normalized = centered / np.maximum(scales, 1e-4)
    lag_scores = []
    max_lag = min(8, max(2, len(normalized) // 2))
    for lag in range(2, max_lag + 1):
        lag_scores.append(_rowwise_cosine_mean(normalized[:-lag], normalized[lag:]))
    periodicity = 0.0
    if lag_scores:
        periodicity = float(np.clip((max(lag_scores) - 0.88) / 0.10, 0.0, 1.0))
    local_identity = float(np.clip(1.0 - np.median(movement) / 0.10, 0.0, 1.0))
    return float(np.clip(0.48 * periodicity + 0.32 * stationarity + 0.20 * local_identity, 0.0, 1.0))


def _rowwise_cosine_mean(left: np.ndarray, right: np.ndarray) -> float:
    numerators = np.sum(left * right, axis=1)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominators = left_norm * right_norm
    similarities = np.where(denominators > 1e-6, numerators / denominators, 1.0)
    return float(np.mean(np.clip(similarities, -1.0, 1.0)))


def _compute_spectral_monotony(spectral_centroids: list[float]) -> float:
    if len(spectral_centroids) < 3:
        return 1.0
    centroids = np.asarray(spectral_centroids, dtype=np.float32)
    centroid_std = float(np.std(centroids))
    centroid_delta = float(np.mean(np.abs(np.diff(centroids))))
    movement = 0.62 * (centroid_std / 260.0) + 0.38 * (centroid_delta / 90.0)
    return float(np.clip(1.0 - movement, 0.0, 1.0))


def _presence_instability(presence_shares: list[float]) -> float:
    if len(presence_shares) < 2:
        return 0.0
    deltas = np.abs(np.diff(np.asarray(presence_shares, dtype=np.float32)))
    return float(np.clip(np.percentile(deltas, 85) / 0.04, 0.0, 1.0))


def _compute_mono_collapse_risk(stereo_corrs: list[float], stereo_widths: list[float]) -> float:
    if not stereo_corrs:
        return 1.0
    corr_mean = float(np.mean(stereo_corrs))
    width_mean = float(np.mean(stereo_widths)) if stereo_widths else 0.0
    corr_risk = np.clip((corr_mean - 0.975) / 0.02, 0.0, 1.0)
    width_risk = np.clip((0.055 - width_mean) / 0.055, 0.0, 1.0)
    return float(np.clip(0.68 * corr_risk + 0.32 * width_risk, 0.0, 1.0))


def _compute_dynamic_flatness(rms_db_values: list[float]) -> float:
    if len(rms_db_values) < 3:
        return 1.0
    rms_array = np.asarray(rms_db_values, dtype=np.float32)
    rms_std = float(np.std(rms_array))
    delta = float(np.mean(np.abs(np.diff(rms_array))))
    movement = 0.7 * (rms_std / 1.8) + 0.3 * (delta / 0.7)
    return float(np.clip(1.0 - movement, 0.0, 1.0))


def _compute_dynamic_breath_score(rms_db_values: list[float]) -> float:
    if len(rms_db_values) < 3:
        return 0.0
    values = np.asarray(rms_db_values, dtype=np.float32)
    p_range = float(np.percentile(values, 90) - np.percentile(values, 10))
    delta = float(np.mean(np.abs(np.diff(values))))
    range_score = np.clip((p_range - 2.5) / 9.5, 0.0, 1.0)
    movement_score = np.clip(delta / 1.15, 0.0, 1.0)
    return float(np.clip(0.74 * range_score + 0.26 * movement_score, 0.0, 1.0))


def _compute_bass_anchor_score(sub_shares: list[float], bass_shares: list[float], lowmid_shares: list[float]) -> float:
    if not bass_shares:
        return 0.0
    sub = float(np.median(sub_shares)) if sub_shares else 0.0
    bass = float(np.median(bass_shares))
    lowmid = float(np.median(lowmid_shares)) if lowmid_shares else 0.0
    bass_score = np.clip((bass - 0.18) / 0.32, 0.0, 1.0)
    lowmid_balance = 1.0 - np.clip(abs(lowmid - 0.36) / 0.34, 0.0, 1.0)
    sub_penalty = np.clip((sub - 0.18) / 0.42, 0.0, 1.0)
    return float(np.clip(0.72 * bass_score + 0.28 * lowmid_balance - 0.22 * sub_penalty, 0.0, 1.0))


def _compute_lowmid_boxiness_risk(bass_shares: list[float], lowmid_shares: list[float], body_shares: list[float]) -> float:
    if not lowmid_shares:
        return 0.0
    bass = float(np.median(bass_shares)) if bass_shares else 0.0
    lowmid = float(np.median(lowmid_shares))
    body = float(np.median(body_shares)) if body_shares else 0.0
    lowmid_excess = np.clip((lowmid - 0.62) / 0.28, 0.0, 1.0)
    missing_bass = np.clip((0.22 - bass) / 0.22, 0.0, 1.0)
    missing_body = np.clip((0.035 - body) / 0.035, 0.0, 1.0)
    return float(np.clip(0.62 * lowmid_excess + 0.28 * missing_bass + 0.10 * missing_body, 0.0, 1.0))


def _compute_stereo_depth_score(stereo_widths: list[float]) -> float:
    if not stereo_widths:
        return 0.0
    width = float(np.median(stereo_widths))
    if width <= 0.26:
        return float(np.clip((width - 0.055) / 0.205, 0.0, 1.0))
    return float(np.clip(1.0 - (width - 0.26) / 0.42, 0.0, 1.0))


def _compute_dark_balance_score(spectral_centroids: list[float], hf_ratios: list[float], presence_shares: list[float]) -> float:
    if not spectral_centroids:
        return 0.0
    centroid = float(np.median(spectral_centroids))
    hf = float(np.percentile(hf_ratios, 88)) if hf_ratios else 0.0
    presence = float(np.percentile(presence_shares, 88)) if presence_shares else 0.0
    centroid_score = 1.0 - np.clip((centroid - 260.0) / 900.0, 0.0, 1.0)
    hf_score = 1.0 - np.clip(hf / 0.08, 0.0, 1.0)
    presence_score = 1.0 - np.clip(presence / 0.08, 0.0, 1.0)
    return float(np.clip(0.56 * centroid_score + 0.22 * hf_score + 0.22 * presence_score, 0.0, 1.0))


def _compute_silence_issue_ratio(rms_db_values: list[float]) -> float:
    if not rms_db_values:
        return 0.0
    values = np.asarray(rms_db_values, dtype=np.float32)
    if len(values) <= 4:
        return float(np.mean(values < -56.0))

    grace = max(1, min(3, int(round(len(values) * 0.12))))
    issue_blocks = 0
    core_length = max(1, len(values) - grace * 2)
    index = grace
    while index < len(values) - grace:
        if values[index] >= -56.0:
            index += 1
            continue
        run_end = index
        while run_end < len(values) - grace and values[run_end] < -56.0:
            run_end += 1
        run_values = values[index:run_end]
        severe = bool(np.min(run_values) < -60.5)
        if severe or len(run_values) >= 2:
            issue_blocks += len(run_values)
        index = run_end
    return float(issue_blocks / core_length)


def _boundary_smoothness(master_path: Path, sample_rate: int, boundary_frames: list[int]) -> float:
    if not boundary_frames:
        return 1.0
    window = min(sample_rate * 3, sample_rate * 6)
    jumps = []
    with sf.SoundFile(master_path, mode="r") as reader:
        for boundary in boundary_frames:
            start = max(0, boundary - window)
            reader.seek(start)
            excerpt = reader.read(window * 2, dtype="float32", always_2d=True)
            if len(excerpt) < window * 2:
                continue
            first = excerpt[:window]
            second = excerpt[window : window * 2]
            jump = abs(
                20.0 * np.log10(max(np.sqrt(np.mean(first ** 2)), 1e-9))
                - 20.0 * np.log10(max(np.sqrt(np.mean(second ** 2)), 1e-9))
            )
            jumps.append(jump)
    if not jumps:
        return 1.0
    mean_jump = float(np.mean(jumps))
    return float(np.clip(1.0 - mean_jump / 12.0, 0.0, 1.0))
