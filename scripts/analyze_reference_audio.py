from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np

try:
    import imageio_ffmpeg
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"imageio_ffmpeg is required in .vendor: {exc}") from exc

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover - optional in bundled deps
    pyln = None


BANDS_HZ: list[tuple[str, float, float]] = [
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 120.0),
    ("low_mid", 120.0, 300.0),
    ("body", 300.0, 800.0),
    ("mids", 800.0, 2000.0),
    ("presence", 2000.0, 5000.0),
    ("air", 5000.0, 12000.0),
    ("hiss", 8000.0, 16000.0),
]

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class Probe:
    duration_seconds: float
    sample_rate: int | None
    channels: str | None
    bitrate: str | None
    raw: str


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_audio(path: Path) -> Probe:
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stderr + proc.stdout
    duration = 0.0
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        hh, mm, ss = match.groups()
        duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
    sample_rate = None
    match = re.search(r"Audio:.*?(\d+)\s*Hz", text)
    if match:
        sample_rate = int(match.group(1))
    channels = None
    match = re.search(r"Audio:.*?Hz,\s*([^,]+),", text)
    if match:
        channels = match.group(1).strip()
    bitrate = None
    match = re.search(r"bitrate:\s*([^,\n]+)", text)
    if match:
        bitrate = match.group(1).strip()
    return Probe(duration, sample_rate, channels, bitrate, text)


def decode_segment(path: Path, start: float, seconds: float, sample_rate: int) -> np.ndarray:
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-t",
        f"{seconds:.3f}",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed at {start:.2f}s: {err}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return data.reshape(-1, 2)


def safe_db(value: float, floor: float = -120.0) -> float:
    if value <= 1e-12 or not math.isfinite(value):
        return floor
    return float(20.0 * math.log10(value))


def frame_rms(mono: np.ndarray, sample_rate: int, frame_seconds: float = 1.0, hop_seconds: float = 0.5) -> np.ndarray:
    frame = max(1, int(frame_seconds * sample_rate))
    hop = max(1, int(hop_seconds * sample_rate))
    if mono.size < frame:
        return np.array([float(np.sqrt(np.mean(mono * mono) + 1e-12))], dtype=np.float64)
    values = []
    for start in range(0, mono.size - frame + 1, hop):
        block = mono[start : start + frame]
        values.append(float(np.sqrt(np.mean(block * block) + 1e-12)))
    return np.asarray(values, dtype=np.float64)


def spectral_features(mono: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if mono.size == 0:
        return {}
    max_samples = min(mono.size, sample_rate * 45)
    mono = mono[:max_samples].astype(np.float64, copy=False)
    window = np.hanning(mono.size)
    spec = np.abs(np.fft.rfft(mono * window)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    audible = (freqs >= 20.0) & (freqs <= min(20000.0, sample_rate / 2.0))
    total = float(np.sum(spec[audible]) + 1e-18)

    bands = {}
    for name, low, high in BANDS_HZ:
        mask = (freqs >= low) & (freqs < high)
        bands[name] = float(np.sum(spec[mask]) / total)

    centroid = float(np.sum(freqs[audible] * spec[audible]) / total)
    cumulative = np.cumsum(spec[audible])
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff_freqs = freqs[audible]
    rolloff_85 = float(rolloff_freqs[min(rolloff_idx, rolloff_freqs.size - 1)])
    geometric = float(np.exp(np.mean(np.log(spec[audible] + 1e-18))))
    arithmetic = float(np.mean(spec[audible] + 1e-18))
    flatness = float(geometric / arithmetic)

    low_peak_mask = (freqs >= 35.0) & (freqs <= 600.0)
    peak_freqs = freqs[low_peak_mask]
    peak_spec = spec[low_peak_mask]
    top_low_peaks: list[dict[str, float]] = []
    if peak_freqs.size > 0:
        idxs = np.argpartition(peak_spec, -min(12, peak_spec.size))[-min(12, peak_spec.size) :]
        idxs = idxs[np.argsort(peak_spec[idxs])[::-1]]
        max_peak = float(peak_spec[idxs[0]] + 1e-18)
        used: list[float] = []
        for idx in idxs:
            freq = float(peak_freqs[idx])
            if any(abs(freq - prev) < 8.0 for prev in used):
                continue
            used.append(freq)
            top_low_peaks.append({"hz": round(freq, 2), "relative": round(float(peak_spec[idx] / max_peak), 4)})
            if len(top_low_peaks) >= 6:
                break

    chroma = np.zeros(12, dtype=np.float64)
    chroma_mask = (freqs >= 55.0) & (freqs <= 4000.0)
    chroma_freqs = freqs[chroma_mask]
    chroma_spec = spec[chroma_mask]
    for freq, energy in zip(chroma_freqs, chroma_spec):
        if freq <= 0:
            continue
        midi = int(round(69 + 12 * math.log2(float(freq) / 440.0)))
        chroma[midi % 12] += float(energy)
    if float(np.sum(chroma)) > 0:
        chroma = chroma / float(np.sum(chroma))
    top_chroma_idx = np.argsort(chroma)[::-1][:5]
    top_chroma = [{"pc": PITCH_CLASSES[int(i)], "share": round(float(chroma[int(i)]), 4)} for i in top_chroma_idx]

    return {
        "band_energy_share": {k: round(v, 6) for k, v in bands.items()},
        "spectral_centroid_hz": round(centroid, 2),
        "rolloff_85_hz": round(rolloff_85, 2),
        "spectral_flatness": round(flatness, 6),
        "top_low_peaks": top_low_peaks,
        "top_pitch_classes": top_chroma,
    }


def stereo_features(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {"correlation": 0.0, "side_to_mid": 0.0, "balance_l_minus_r_db": 0.0}
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    corr = float(np.corrcoef(left, right)[0, 1]) if left.size > 4 else 0.0
    if not math.isfinite(corr):
        corr = 0.0
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(mid * mid) + 1e-12))
    side_rms = float(np.sqrt(np.mean(side * side) + 1e-12))
    left_rms = float(np.sqrt(np.mean(left * left) + 1e-12))
    right_rms = float(np.sqrt(np.mean(right * right) + 1e-12))
    return {
        "correlation": round(corr, 4),
        "side_to_mid": round(side_rms / max(mid_rms, 1e-9), 4),
        "balance_l_minus_r_db": round(safe_db(left_rms / max(right_rms, 1e-9)), 3),
    }


def modulation_features(mono: np.ndarray, sample_rate: int) -> dict[str, float]:
    env = frame_rms(mono, sample_rate, frame_seconds=0.25, hop_seconds=0.1)
    if env.size < 32:
        return {"modulation_depth_db": 0.0, "dominant_modulation_hz": 0.0, "pulse_strength": 0.0}
    env_db = np.array([safe_db(v) for v in env])
    modulation_depth = float(np.percentile(env_db, 95) - np.percentile(env_db, 5))
    env_centered = env - np.mean(env)
    spec = np.abs(np.fft.rfft(env_centered * np.hanning(env_centered.size))) ** 2
    freqs = np.fft.rfftfreq(env_centered.size, 0.1)
    mask = (freqs >= 0.03) & (freqs <= 4.0)
    if not np.any(mask):
        return {"modulation_depth_db": round(modulation_depth, 3), "dominant_modulation_hz": 0.0, "pulse_strength": 0.0}
    masked = spec[mask]
    masked_freqs = freqs[mask]
    idx = int(np.argmax(masked))
    dominant = float(masked_freqs[idx])
    pulse_strength = float(masked[idx] / (np.mean(masked) + 1e-18))
    return {
        "modulation_depth_db": round(modulation_depth, 3),
        "dominant_modulation_hz": round(dominant, 4),
        "pulse_strength": round(pulse_strength, 3),
    }


def segment_metrics(audio: np.ndarray, start: float, sample_rate: int) -> dict[str, Any]:
    if audio.size == 0:
        return {"start_seconds": start, "error": "empty decode"}
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    mono = np.mean(audio, axis=1)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    frame_values = frame_rms(mono, sample_rate)
    frame_db = np.array([safe_db(v) for v in frame_values])
    loudness = None
    if pyln is not None:
        try:
            meter = pyln.Meter(sample_rate)
            loudness = float(meter.integrated_loudness(audio))
        except Exception:
            loudness = None

    features: dict[str, Any] = {
        "start_seconds": round(float(start), 3),
        "start_time": seconds_to_time(start),
        "rms_dbfs": round(safe_db(rms), 3),
        "peak_dbfs": round(safe_db(peak), 3),
        "crest_factor_db": round(safe_db(peak / max(rms, 1e-9)), 3),
        "short_term_range_db": round(float(np.percentile(frame_db, 95) - np.percentile(frame_db, 5)), 3),
        "p05_rms_dbfs": round(float(np.percentile(frame_db, 5)), 3),
        "p50_rms_dbfs": round(float(np.percentile(frame_db, 50)), 3),
        "p95_rms_dbfs": round(float(np.percentile(frame_db, 95)), 3),
    }
    if loudness is not None and math.isfinite(loudness):
        features["approx_lufs"] = round(loudness, 3)
    features.update(spectral_features(mono, sample_rate))
    features["stereo"] = stereo_features(audio)
    features["modulation"] = modulation_features(mono, sample_rate)
    return features


def seconds_to_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": round(float(np.min(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def aggregate(segments: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    agg: dict[str, Any] = {
        "duration_time": seconds_to_time(duration),
        "segment_count": len(segments),
    }
    scalar_keys = [
        "rms_dbfs",
        "peak_dbfs",
        "crest_factor_db",
        "short_term_range_db",
        "p50_rms_dbfs",
        "approx_lufs",
        "spectral_centroid_hz",
        "rolloff_85_hz",
        "spectral_flatness",
    ]
    for key in scalar_keys:
        vals = [float(s[key]) for s in segments if key in s]
        agg[key] = summarize(vals)

    band_summary: dict[str, Any] = {}
    for name, _, _ in BANDS_HZ:
        vals = [float(s["band_energy_share"][name]) for s in segments if "band_energy_share" in s]
        band_summary[name] = summarize(vals)
    agg["band_energy_share"] = band_summary

    stereo_summary: dict[str, Any] = {}
    for key in ["correlation", "side_to_mid", "balance_l_minus_r_db"]:
        vals = [float(s["stereo"][key]) for s in segments if "stereo" in s]
        stereo_summary[key] = summarize(vals)
    agg["stereo"] = stereo_summary

    modulation_summary: dict[str, Any] = {}
    for key in ["modulation_depth_db", "dominant_modulation_hz", "pulse_strength"]:
        vals = [float(s["modulation"][key]) for s in segments if "modulation" in s]
        modulation_summary[key] = summarize(vals)
    agg["modulation"] = modulation_summary

    chroma_totals = {pc: 0.0 for pc in PITCH_CLASSES}
    low_peaks: list[float] = []
    for segment in segments:
        for item in segment.get("top_pitch_classes", []):
            chroma_totals[item["pc"]] += float(item["share"])
        for item in segment.get("top_low_peaks", []):
            low_peaks.append(float(item["hz"]))
    agg["dominant_pitch_classes"] = sorted(
        [{"pc": pc, "score": round(score, 4)} for pc, score in chroma_totals.items()],
        key=lambda item: item["score"],
        reverse=True,
    )[:7]
    agg["common_low_peak_hz"] = cluster_frequencies(low_peaks)
    agg["section_shift_candidates"] = detect_shift_candidates(segments)
    agg["production_interpretation"] = interpret_aggregate(agg)
    return agg


def cluster_frequencies(freqs: list[float]) -> list[dict[str, float]]:
    if not freqs:
        return []
    freqs = sorted(freqs)
    clusters: list[list[float]] = []
    for freq in freqs:
        if not clusters or abs(freq - float(np.mean(clusters[-1]))) > 8.0:
            clusters.append([freq])
        else:
            clusters[-1].append(freq)
    result = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        result.append({"hz": round(float(np.mean(cluster)), 2), "hits": len(cluster)})
    return sorted(result, key=lambda item: item["hits"], reverse=True)[:10]


def detect_shift_candidates(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(segments) < 3:
        return []
    feature_rows = []
    times = []
    for segment in segments:
        bands = segment.get("band_energy_share", {})
        stereo = segment.get("stereo", {})
        modulation = segment.get("modulation", {})
        row = [
            float(segment.get("p50_rms_dbfs", 0.0)),
            float(segment.get("spectral_centroid_hz", 0.0)) / 1000.0,
            float(segment.get("spectral_flatness", 0.0)) * 100.0,
            float(bands.get("bass", 0.0)) * 10.0,
            float(bands.get("low_mid", 0.0)) * 10.0,
            float(bands.get("presence", 0.0)) * 20.0,
            float(bands.get("hiss", 0.0)) * 20.0,
            float(stereo.get("side_to_mid", 0.0)) * 5.0,
            float(modulation.get("modulation_depth_db", 0.0)),
        ]
        feature_rows.append(row)
        times.append(segment["start_time"])
    matrix = np.asarray(feature_rows, dtype=np.float64)
    std = np.std(matrix, axis=0) + 1e-9
    z = (matrix - np.mean(matrix, axis=0)) / std
    distances = np.linalg.norm(np.diff(z, axis=0), axis=1)
    if distances.size == 0:
        return []
    threshold = max(float(np.percentile(distances, 82)), float(np.mean(distances) + 0.45 * np.std(distances)))
    candidates = []
    for idx, distance in enumerate(distances):
        if distance >= threshold:
            candidates.append({"time": times[idx + 1], "change_score": round(float(distance), 3)})
    return sorted(candidates, key=lambda item: item["change_score"], reverse=True)[:12]


def interpret_aggregate(agg: dict[str, Any]) -> dict[str, str]:
    bands = agg["band_energy_share"]
    bass = bands["bass"]["median"]
    low_mid = bands["low_mid"]["median"]
    presence = bands["presence"]["median"]
    hiss = bands["hiss"]["median"]
    side = agg["stereo"]["side_to_mid"]["median"]
    dynamic = agg["short_term_range_db"]["median"]
    pulse = agg["modulation"]["pulse_strength"]["median"]
    centroid = agg["spectral_centroid_hz"]["median"]

    base = []
    if bass + low_mid > 0.45:
        base.append("low-mid anchored")
    if centroid < 700:
        base.append("dark/warm")
    elif centroid < 1200:
        base.append("warm with controlled brightness")
    else:
        base.append("brighter than sleep-safe ambient")
    if dynamic < 4.0:
        base.append("very stable")
    elif dynamic < 8.0:
        base.append("slowly breathing")
    else:
        base.append("noticeably animated")

    texture = []
    if hiss < 0.02:
        texture.append("little broadband hiss")
    elif hiss < 0.06:
        texture.append("soft high-frequency veil")
    else:
        texture.append("prominent hiss/noise layer")
    if presence < 0.04:
        texture.append("presence band is restrained")
    elif presence < 0.1:
        texture.append("presence band is audible but controlled")
    else:
        texture.append("presence band is forward")

    stereo = []
    if side < 0.18:
        stereo.append("narrow mono-compatible core")
    elif side < 0.45:
        stereo.append("moderate stereo width")
    else:
        stereo.append("wide/decorrelated spatial bed")

    rhythm = []
    if pulse < 4.0:
        rhythm.append("no obvious rhythmic pulse")
    elif pulse < 9.0:
        rhythm.append("subtle slow pulse")
    else:
        rhythm.append("clear repeated modulation")

    return {
        "base": ", ".join(base),
        "texture": ", ".join(texture),
        "stereo": ", ".join(stereo),
        "rhythm": ", ".join(rhythm),
    }


def write_markdown(path: Path, source: Path, probe: Probe, segments: list[dict[str, Any]], agg: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Reference audio analysis")
    lines.append("")
    lines.append(f"- Source: `{source}`")
    lines.append(f"- Duration: `{agg['duration_time']}`")
    lines.append(f"- Input sample rate: `{probe.sample_rate}`")
    lines.append(f"- Channels: `{probe.channels}`")
    lines.append(f"- Bitrate: `{probe.bitrate}`")
    lines.append(f"- Analyzed windows: `{agg['segment_count']}`")
    lines.append("")
    lines.append("## Aggregate metrics")
    scalar_items = [
        ("Approx loudness LUFS", "approx_lufs"),
        ("RMS dBFS", "rms_dbfs"),
        ("Peak dBFS", "peak_dbfs"),
        ("Crest factor dB", "crest_factor_db"),
        ("Short-term range dB", "short_term_range_db"),
        ("Spectral centroid Hz", "spectral_centroid_hz"),
        ("Rolloff 85 Hz", "rolloff_85_hz"),
        ("Spectral flatness", "spectral_flatness"),
    ]
    for label, key in scalar_items:
        if key in agg:
            item = agg[key]
            lines.append(f"- {label}: median `{item['median']}`, p10 `{item['p10']}`, p90 `{item['p90']}`")
    lines.append("")
    lines.append("## Band energy share")
    for name, _, _ in BANDS_HZ:
        item = agg["band_energy_share"][name]
        lines.append(f"- {name}: median `{item['median']}`, p10 `{item['p10']}`, p90 `{item['p90']}`")
    lines.append("")
    lines.append("## Stereo and movement")
    for label, key in [("Correlation", "correlation"), ("Side to mid", "side_to_mid"), ("L-R balance dB", "balance_l_minus_r_db")]:
        item = agg["stereo"][key]
        lines.append(f"- {label}: median `{item['median']}`, p10 `{item['p10']}`, p90 `{item['p90']}`")
    for label, key in [("Modulation depth dB", "modulation_depth_db"), ("Dominant modulation Hz", "dominant_modulation_hz"), ("Pulse strength", "pulse_strength")]:
        item = agg["modulation"][key]
        lines.append(f"- {label}: median `{item['median']}`, p10 `{item['p10']}`, p90 `{item['p90']}`")
    lines.append("")
    lines.append("## Tonal clues")
    lines.append("- Dominant pitch classes: " + ", ".join(f"{i['pc']}={i['score']}" for i in agg["dominant_pitch_classes"]))
    lines.append("- Common low peaks: " + ", ".join(f"{i['hz']}Hz({i['hits']})" for i in agg["common_low_peak_hz"]))
    lines.append("")
    lines.append("## Section shift candidates")
    if agg["section_shift_candidates"]:
        for item in agg["section_shift_candidates"]:
            lines.append(f"- {item['time']}: change score `{item['change_score']}`")
    else:
        lines.append("- No strong shift candidates detected from sampled windows.")
    lines.append("")
    lines.append("## Production interpretation")
    for key, value in agg["production_interpretation"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Segment map")
    for segment in segments:
        bands = segment.get("band_energy_share", {})
        stereo = segment.get("stereo", {})
        modulation = segment.get("modulation", {})
        lines.append(
            "- "
            f"{segment['start_time']} | "
            f"LUFS {segment.get('approx_lufs', 'n/a')} | "
            f"centroid {segment.get('spectral_centroid_hz', 'n/a')}Hz | "
            f"bass {bands.get('bass', 'n/a')} | "
            f"low_mid {bands.get('low_mid', 'n/a')} | "
            f"presence {bands.get('presence', 'n/a')} | "
            f"hiss {bands.get('hiss', 'n/a')} | "
            f"side/mid {stereo.get('side_to_mid', 'n/a')} | "
            f"pulse {modulation.get('pulse_strength', 'n/a')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a long ambient reference without loading it all into memory.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "reference_audio")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--window-seconds", type=float, default=45.0)
    parser.add_argument("--windows", type=int, default=60)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Input not found: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_audio(source)
    if probe.duration_seconds <= 0:
        raise SystemExit("Could not determine duration with ffmpeg.")

    max_start = max(0.0, probe.duration_seconds - args.window_seconds)
    if args.windows <= 1:
        starts = [0.0]
    else:
        starts = np.linspace(0.0, max_start, args.windows)

    segments = []
    for index, start in enumerate(starts, start=1):
        print(f"[{index:02d}/{len(starts):02d}] decode/analyze {seconds_to_time(float(start))}", flush=True)
        audio = decode_segment(source, float(start), args.window_seconds, args.sample_rate)
        segments.append(segment_metrics(audio, float(start), args.sample_rate))

    agg = aggregate(segments, probe.duration_seconds)
    result = {
        "source": str(source),
        "probe": {
            "duration_seconds": probe.duration_seconds,
            "duration_time": seconds_to_time(probe.duration_seconds),
            "sample_rate": probe.sample_rate,
            "channels": probe.channels,
            "bitrate": probe.bitrate,
        },
        "analysis_settings": {
            "sample_rate": args.sample_rate,
            "window_seconds": args.window_seconds,
            "windows": args.windows,
        },
        "aggregate": agg,
        "segments": segments,
    }
    json_path = args.output_dir / "reference_analysis.json"
    md_path = args.output_dir / "reference_analysis.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(md_path, source, probe, segments, agg)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
