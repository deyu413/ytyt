from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ambient_engine.core.durations import humanize_seconds
from ambient_engine.core.manifest import SessionManifest
from ambient_engine.profiles.schema import Profile
from ambient_engine.render.static_frame import render_static_frame


@dataclass(frozen=True)
class ShortConcept:
    slug: str
    kind: str
    duration_seconds: int
    primary_text: str
    secondary_text: str
    tertiary_text: str


def render_session_shorts(
    profile: Profile,
    metadata: dict[str, Any],
    manifest: SessionManifest,
    session_dir: Path,
    ffmpeg_executable: str | None,
) -> dict[str, Any]:
    if not ffmpeg_executable:
        raise RuntimeError("FFmpeg executable not available. Install imageio-ffmpeg or ffmpeg.")

    session_dir = session_dir.resolve()
    exports_dir = session_dir / "exports"
    shorts_dir = exports_dir / "shorts"
    shorts_dir.mkdir(parents=True, exist_ok=True)

    master_path = exports_dir / "master.wav"
    static_frame_path = exports_dir / "static_frame.png"
    background_image = manifest.asset_lineage.get("background_image")
    background_image_path = Path(background_image) if background_image else None
    if background_image_path is not None and not background_image_path.exists():
        background_image_path = None
    if background_image_path is None:
        background_image_path = static_frame_path

    with sf.SoundFile(master_path, mode="r") as reader:
        sample_rate = int(reader.samplerate)
        total_seconds = max(1, int(math.ceil(len(reader) / max(1, reader.samplerate))))

    highlight = select_highlight_excerpt(
        master_path=master_path,
        sample_rate=sample_rate,
        section_plan=manifest.section_plan,
        total_seconds=total_seconds,
        profile=profile,
    )
    concepts = build_short_concepts(profile, metadata)
    rendered_shorts: list[dict[str, Any]] = []

    for index, concept in enumerate(concepts):
        start_seconds = _derive_short_start(
            highlight=highlight,
            duration_seconds=concept.duration_seconds,
            total_seconds=total_seconds,
            variant_index=index,
        )
        frame_path = shorts_dir / f"{concept.slug}_frame.png"
        video_path = shorts_dir / f"{concept.slug}.mp4"
        metadata_path = shorts_dir / f"{concept.slug}.json"

        render_short_frame(
            profile=profile,
            metadata=metadata,
            variation=manifest.variation,
            concept=concept,
            output_path=frame_path,
            background_image_path=background_image_path,
        )
        render_short_video(
            ffmpeg_executable=ffmpeg_executable,
            frame_path=frame_path,
            audio_path=master_path,
            output_path=video_path,
            start_seconds=start_seconds,
            duration_seconds=concept.duration_seconds,
        )

        short_metadata = build_short_metadata(
            profile=profile,
            metadata=metadata,
            concept=concept,
            manifest=manifest,
            start_seconds=start_seconds,
        )
        metadata_path.write_text(json.dumps(short_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        rendered_shorts.append(
            {
                "slug": concept.slug,
                "kind": concept.kind,
                "video": str(video_path),
                "frame": str(frame_path),
                "metadata": str(metadata_path),
                "start_seconds": start_seconds,
                "duration_seconds": concept.duration_seconds,
                "primary_text": concept.primary_text,
            }
        )

    payload = {
        "source_session_id": manifest.session_id,
        "profile_id": profile.profile_id,
        "highlight": highlight,
        "shorts": rendered_shorts,
    }
    manifest_path = session_dir / "manifests" / "shorts_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"shorts_manifest": str(manifest_path), "shorts": rendered_shorts}


def build_short_concepts(profile: Profile, metadata: dict[str, Any]) -> list[ShortConcept]:
    shorts_config = dict(profile.branding.get("shorts", {}))
    emotional_hooks = list(shorts_config.get("emotional_hooks", []))
    if not emotional_hooks:
        emotional_hooks = [f"it's late. let your mind rest."]
    mood_words = [str(word).upper() for word in shorts_config.get("mood_words", _default_mood_words(profile))]
    teaser_line = str(shorts_config.get("teaser_line", "FULL VERSION ON CHANNEL")).upper()
    teaser_cta = str(shorts_config.get("teaser_cta", "watch the long version")).upper()
    series_name = str(profile.branding.get("series_name", profile.mood)).upper()

    return [
        ShortConcept(
            slug="short_01_emotional_hook",
            kind="emotional_hook",
            duration_seconds=18,
            primary_text=emotional_hooks[0],
            secondary_text=series_name,
            tertiary_text="LATE-NIGHT AMBIENT",
        ),
        ShortConcept(
            slug="short_02_mood_hook",
            kind="mood_hook",
            duration_seconds=15,
            primary_text=mood_words[0] if mood_words else "QUIET",
            secondary_text=series_name,
            tertiary_text="",
        ),
        ShortConcept(
            slug="short_03_longform_teaser",
            kind="longform_teaser",
            duration_seconds=20,
            primary_text=teaser_line,
            secondary_text=str(metadata["title"]).upper(),
            tertiary_text=teaser_cta,
        ),
    ]


def select_highlight_excerpt(
    master_path: Path,
    sample_rate: int,
    section_plan: list[dict[str, Any]],
    total_seconds: int | None = None,
    window_seconds: int = 20,
    profile: Profile | None = None,
) -> dict[str, Any]:
    features = _scan_audio_features(master_path, sample_rate)
    if not features:
        return {"start_seconds": 0, "duration_seconds": window_seconds, "score": 0.0}
    total_seconds = total_seconds or len(features)
    usable_window = min(window_seconds, len(features))
    if len(features) <= usable_window:
        return {"start_seconds": 0, "duration_seconds": usable_window, "score": 1.0}

    rms = np.asarray([row["rms_db"] for row in features], dtype=np.float32)
    hf = np.asarray([row["hf_ratio"] for row in features], dtype=np.float32)
    flux = np.asarray([row["flux"] for row in features], dtype=np.float32)
    density = np.asarray([_section_density_at(second + 0.5, section_plan) for second in range(len(features))], dtype=np.float32)
    highlight_mode = _highlight_mode(profile)
    role_bias = np.asarray([_section_role_bias_at(second + 0.5, section_plan, highlight_mode) for second in range(len(features))], dtype=np.float32)

    rms_norm = _minmax_normalize(rms)
    hf_norm = _minmax_normalize(hf)
    flux_norm = _minmax_normalize(flux)
    density_norm = _minmax_normalize(density)

    edge_grace = min(max(6, usable_window // 2), max(6, int(total_seconds * 0.08)))
    best_start = edge_grace
    best_score = -1.0
    end_limit = max(edge_grace + 1, len(features) - edge_grace - usable_window + 1)
    for start in range(edge_grace, end_limit):
        end = start + usable_window
        window_rms = float(np.mean(rms_norm[start:end]))
        window_peak_rms = float(np.percentile(rms_norm[start:end], 80))
        window_flux = float(np.mean(flux_norm[start:end]))
        window_hf = float(np.mean(hf_norm[start:end]))
        window_density = float(np.mean(density_norm[start:end]))
        window_role = float(np.mean(role_bias[start:end]))
        silence_penalty = 0.18 if float(np.mean(rms[start:end])) < -42.0 else 0.0
        if highlight_mode == "calm_hook":
            target_rms = 0.5
            target_flux = 0.24
            energy_fit = 1.0 - abs(window_rms - target_rms) * 1.12
            flux_fit = 1.0 - abs(window_flux - target_flux) * 1.08
            hf_penalty = max(0.0, window_hf - 0.22)
            peak_penalty = max(0.0, window_peak_rms - 0.7)
            score = (
                0.28 * max(0.0, energy_fit)
                + 0.10 * max(0.0, flux_fit)
                + 0.10 * window_density
                + 0.20 * window_role
                + 0.06 * (1.0 - window_hf)
                - 0.24 * hf_penalty
                - 0.16 * peak_penalty
                - silence_penalty
            )
        else:
            score = (
                0.34 * window_rms
                + 0.20 * window_peak_rms
                + 0.14 * window_flux
                + 0.08 * window_hf
                + 0.14 * window_density
                + 0.10 * window_role
                - silence_penalty
            )
        if score > best_score:
            best_score = score
            best_start = start
    return {"start_seconds": int(best_start), "duration_seconds": int(usable_window), "score": round(float(best_score), 4)}


def render_short_frame(
    profile: Profile,
    metadata: dict[str, Any],
    variation: dict[str, Any],
    concept: ShortConcept,
    output_path: Path,
    background_image_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = output_path.parent / f"{concept.slug}_base.png"
    render_static_frame(
        profile=profile,
        variation=variation,
        output_path=base_path,
        size=(1080, 1920),
        background_image_path=background_image_path,
    )
    image = Image.open(base_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    accent_rgb = _hex(profile.thumbnail_style.get("accent", "#8fd0ff"))

    badge_font = _font(40)
    footer_font = _font(28)
    badge_box = (70, 84, 430, 160)
    draw.rounded_rectangle(badge_box, radius=22, fill=(4, 7, 15, 176), outline=accent_rgb + (180,), width=2)
    draw.text((96, 105), str(metadata["hud_label"]).upper(), font=badge_font, fill=(255, 255, 255, 236))

    text_panel = (58, 420, 1022, 1280)
    if concept.kind == "mood_hook":
        text_panel = (90, 500, 990, 1180)
    draw.rounded_rectangle(text_panel, radius=42, fill=(4, 7, 15, 120), outline=accent_rgb + (96,), width=2)
    draw.rounded_rectangle((76, 1400, 1004, 1840), radius=36, fill=(2, 5, 11, 104), outline=accent_rgb + (70,), width=2)

    if concept.kind == "emotional_hook":
        primary_lines = _wrap_text(concept.primary_text, max_chars=20)
        primary_font = _fit_font(primary_lines, max_width=820, start_size=108)
        y = 560
        for line in primary_lines[:3]:
            draw.text((110, y), line, font=primary_font, fill=(255, 255, 255, 242))
            y += primary_font.size + 18
        draw.text((112, 1022), concept.secondary_text, font=_font(34), fill=accent_rgb + (228,))
        draw.text((112, 1080), concept.tertiary_text, font=_font(28), fill=(255, 255, 255, 172))
    elif concept.kind == "mood_hook":
        primary_font = _fit_font([concept.primary_text], max_width=820, start_size=210)
        text_width = primary_font.getbbox(concept.primary_text)[2]
        draw.text(((1080 - text_width) // 2, 700), concept.primary_text, font=primary_font, fill=(255, 255, 255, 246))
        draw.text((388, 965), concept.secondary_text, font=_font(36), fill=accent_rgb + (222,))
    else:
        primary_lines = _wrap_text(concept.primary_text, max_chars=18)
        primary_font = _fit_font(primary_lines, max_width=820, start_size=112)
        y = 560
        for line in primary_lines[:2]:
            draw.text((110, y), line, font=primary_font, fill=(255, 255, 255, 242))
            y += primary_font.size + 16
        secondary_lines = _wrap_text(concept.secondary_text, max_chars=24)
        secondary_font = _fit_font(secondary_lines, max_width=820, start_size=44)
        for line in secondary_lines[:2]:
            draw.text((112, y + 20), line, font=secondary_font, fill=accent_rgb + (216,))
            y += secondary_font.size + 10
        draw.text((112, 1048), concept.tertiary_text, font=_font(28), fill=(255, 255, 255, 186))

    draw.text((84, 1864), "AUDIO PREVIEW FROM THE LONG-FORM SESSION", font=footer_font, fill=(255, 255, 255, 154))
    result = Image.alpha_composite(image, overlay)
    result = Image.alpha_composite(result, _short_glow(result.size, accent_rgb))
    result.save(output_path)
    return output_path


def render_short_video(
    ffmpeg_executable: str,
    frame_path: Path,
    audio_path: Path,
    output_path: Path,
    start_seconds: int,
    duration_seconds: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_attempts = [
        (
            "[0:v]format=rgba,scale=1188:2112,crop=1080:1920:x='(iw-ow)/2+18*sin(t/5)':y='(ih-oh)/2+14*sin(t/7)',setsar=1[bg];"
            "[1:a]aformat=channel_layouts=stereo,asplit=2[a1][a2];"
            "[a1]showfreqs=s=860x220:mode=bar:fscale=log:ascale=sqrt:colors=0x9FD9FF,format=rgba,colorchannelmixer=aa=0.82[freq];"
            "[a2]showwaves=s=860x86:mode=line:colors=0xFFFFFF,format=rgba,colorchannelmixer=aa=0.86[waves];"
            "[bg][freq]overlay=110:1450[tmp];"
            "[tmp][waves]overlay=110:1698[v]"
        ),
        (
            "[0:v]format=rgba,scale=1188:2112,crop=1080:1920:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1[bg];"
            "[1:a]showwaves=s=900x140:mode=cline:colors=0x9FD9FF@0.88,format=rgba[w];"
            "[bg][w]overlay=90:1600[v]"
        ),
    ]
    for filter_graph in filter_attempts:
        cmd = [
            ffmpeg_executable,
            "-y",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(frame_path),
            "-ss",
            f"{start_seconds:.2f}",
            "-t",
            f"{duration_seconds:.2f}",
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path
    raise RuntimeError(f"Short render failed: {result.stderr[-400:]}")


def build_short_metadata(
    profile: Profile,
    metadata: dict[str, Any],
    concept: ShortConcept,
    manifest: SessionManifest,
    start_seconds: int,
) -> dict[str, Any]:
    title_core = concept.primary_text.replace(".", "").replace("\n", " ").strip()
    title = f"{profile.branding.get('series_name', profile.mood)} Short | {title_core}"
    duration_label = humanize_seconds(concept.duration_seconds)
    description_lines = [
        concept.primary_text,
        "",
        f"Short type: {concept.kind}",
        f"Excerpt start: {start_seconds}s",
        f"Excerpt length: {duration_label}",
        "",
        f"Source long-form: {metadata['title']}",
        f"Source session: {manifest.session_id}",
        "CTA: watch the full version on the channel.",
        "",
        "AI disclosure: synthetic audio / synthetic visual.",
    ]
    tags = list(metadata.get("tags", [])) + ["youtube shorts", concept.kind.replace("_", " "), profile.profile_id]
    return {
        "title": title[:100],
        "description": "\n".join(description_lines)[:5000],
        "tags": tags[:20],
        "privacy_status": metadata.get("privacy_status", "private"),
        "source_title": metadata["title"],
        "source_session_id": manifest.session_id,
        "kind": concept.kind,
        "start_seconds": start_seconds,
        "duration_seconds": concept.duration_seconds,
    }


def _derive_short_start(highlight: dict[str, Any], duration_seconds: int, total_seconds: int, variant_index: int) -> int:
    center = int(highlight["start_seconds"]) + int(highlight["duration_seconds"]) // 2
    offset = {-1: 0, 0: -3, 1: 0, 2: 2}.get(variant_index, 0)
    start = center - duration_seconds // 2 + offset
    return int(np.clip(start, 0, max(0, total_seconds - duration_seconds)))


def _scan_audio_features(master_path: Path, sample_rate: int, block_seconds: int = 1) -> list[dict[str, float]]:
    block_frames = max(1024, sample_rate * block_seconds)
    features: list[dict[str, float]] = []
    previous_spectrum: np.ndarray | None = None
    with sf.SoundFile(master_path, mode="r") as reader:
        while True:
            block = reader.read(block_frames, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            mono = block.mean(axis=1)
            rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
            rms_db = 20.0 * np.log10(max(rms, 1e-9))
            excerpt = mono[: min(len(mono), 8192)]
            window = np.hanning(len(excerpt)).astype(np.float32)
            spectrum = np.abs(np.fft.rfft(excerpt * window))
            freqs = np.fft.rfftfreq(len(excerpt), d=1.0 / sample_rate)
            spectral_sum = float(np.sum(spectrum)) + 1e-9
            hf_ratio = float(np.sum(spectrum[freqs >= 6000.0]) / spectral_sum) if np.any(freqs >= 6000.0) else 0.0
            flux = _spectral_flux(previous_spectrum, spectrum)
            previous_spectrum = spectrum
            features.append({"rms_db": rms_db, "hf_ratio": hf_ratio, "flux": flux})
    return features


def _spectral_flux(previous_spectrum: np.ndarray | None, current_spectrum: np.ndarray) -> float:
    if previous_spectrum is None or len(previous_spectrum) != len(current_spectrum):
        return 0.0
    delta = np.maximum(current_spectrum - previous_spectrum, 0.0)
    return float(np.sum(delta) / max(1e-9, np.sum(current_spectrum)))


def _section_density_at(second_mark: float, section_plan: list[dict[str, Any]]) -> float:
    elapsed = 0.0
    for section in section_plan:
        end = elapsed + float(section["duration_seconds"])
        if second_mark < end:
            return float(section.get("density", 0.25))
        elapsed = end
    return float(section_plan[-1].get("density", 0.25)) if section_plan else 0.25


def _section_role_bias_at(second_mark: float, section_plan: list[dict[str, Any]], highlight_mode: str) -> float:
    role_bias = {
        "peak_tension": {
            "intro": -0.16,
            "settle": 0.04,
            "drift_a": 0.11,
            "drift_b": 0.18,
            "sparse_break": -0.2,
            "return": 0.14,
            "low_energy_tail": -0.22,
        },
        "calm_hook": {
            "intro": -0.08,
            "settle": 0.22,
            "drift_a": 0.16,
            "drift_b": -0.2,
            "sparse_break": -0.18,
            "return": 0.20,
            "low_energy_tail": -0.24,
        },
    }.get(highlight_mode, {})
    elapsed = 0.0
    for section in section_plan:
        end = elapsed + float(section["duration_seconds"])
        if second_mark < end:
            return role_bias.get(str(section["role"]), 0.0)
        elapsed = end
    return 0.0


def _highlight_mode(profile: Profile | None) -> str:
    if profile is None:
        return "peak_tension"
    configured = str(profile.branding.get("shorts", {}).get("highlight_mode", "")).strip().lower()
    if configured in {"peak_tension", "calm_hook"}:
        return configured
    mood_text = " ".join([profile.mood, *profile.forbidden_artifacts]).lower()
    if any(token in mood_text for token in ["sleep", "rest", "insomnia", "solitude"]):
        return "calm_hook"
    return "peak_tension"


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if abs(upper - lower) < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lower) / (upper - lower)).astype(np.float32)


def _default_mood_words(profile: Profile) -> list[str]:
    mood_tokens = [token.upper() for token in profile.mood.replace("_", " ").split() if token]
    if not mood_tokens:
        return ["QUIET", "REST", "BLUE"]
    shortlist = [token for token in mood_tokens if len(token) <= 8]
    return shortlist[:3] or mood_tokens[:3]


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _fit_font(lines: list[str], max_width: int, start_size: int) -> ImageFont.ImageFont:
    for size in range(start_size, 30, -4):
        font = _font(size)
        if all(font.getbbox(line)[2] <= max_width for line in lines if line):
            return font
    return _font(32)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _short_glow(size: tuple[int, int], accent: tuple[int, int, int]) -> Image.Image:
    width, height = size
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.ellipse((170, 260, width - 170, height - 460), fill=accent + (18,), outline=accent + (0,))
    return glow.filter(ImageFilter.GaussianBlur(radius=84))


def _hex(value: str) -> tuple[int, int, int]:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))
