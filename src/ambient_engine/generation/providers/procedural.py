from __future__ import annotations

import math
from typing import Any

import numpy as np
import soundfile as sf

from ambient_engine.generation.contracts import BaseProvider, SectionRenderRequest, SectionRenderResult


NOTE_OFFSETS = {
    "C": -9,
    "C#": -8,
    "Db": -8,
    "D": -7,
    "D#": -6,
    "Eb": -6,
    "E": -5,
    "F": -4,
    "F#": -3,
    "Gb": -3,
    "G": -2,
    "G#": -1,
    "Ab": -1,
    "A": 0,
    "A#": 1,
    "Bb": 1,
    "B": 2,
}


SCALE_INTERVALS = {
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "afterblue_minor_cluster": [0, 1, 3, 5, 6, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
}


TEXTURE_PRESETS = {
    "air_soft": {"air": 0.75, "room": 0.55, "rain": 0.0, "tape": 0.25},
    "air_wide": {"air": 0.92, "room": 0.38, "rain": 0.0, "tape": 0.2},
    "almost_dry": {"air": 0.28, "room": 0.22, "rain": 0.0, "tape": 0.16},
    "sub_room": {"air": 0.02, "room": 0.84, "rain": 0.0, "tape": 0.06},
    "black_room": {"air": 0.01, "room": 0.7, "rain": 0.0, "tape": 0.04},
    "room_hush": {"air": 0.14, "room": 0.74, "rain": 0.0, "tape": 0.06},
    "velvet_room": {"air": 0.2, "room": 0.66, "rain": 0.0, "tape": 0.08},
    "rain_soft": {"air": 0.58, "room": 0.48, "rain": 0.55, "tape": 0.18},
    "rain_wide": {"air": 0.76, "room": 0.34, "rain": 0.82, "tape": 0.16},
    "rain_only": {"air": 0.08, "room": 0.28, "rain": 0.95, "tape": 0.08},
}


BASE_PALETTE = {
    "warmth": 0.44,
    "brightness": 0.2,
    "softness": 0.58,
    "grain": 0.24,
    "air": 0.42,
    "room": 0.34,
    "rain_bias": 0.0,
    "pulse": 0.18,
    "mallet": 0.12,
    "sparkle": 0.16,
    "sub": 0.32,
    "bloom": 0.24,
    "bowed": 0.14,
    "glass": 0.12,
    "choir": 0.08,
    "brush": 0.14,
    "texture_bias": 0.22,
}


KEYWORD_PALETTE = {
    "pad": {"warmth": 0.16, "softness": 0.12, "bloom": 0.14},
    "drone": {"sub": 0.16, "warmth": 0.1},
    "bass": {"sub": 0.24, "warmth": 0.1, "bloom": 0.06},
    "subharmonic": {"sub": 0.32, "warmth": 0.08},
    "glass": {"brightness": 0.22, "glass": 0.24, "sparkle": 0.18},
    "bowed": {"bowed": 0.28, "warmth": 0.08, "bloom": 0.08},
    "piano": {"mallet": 0.22, "softness": 0.06, "brightness": 0.05},
    "haze": {"air": 0.2, "softness": 0.08, "bloom": 0.08},
    "pulse": {"pulse": 0.35, "texture_bias": 0.05},
    "sub": {"sub": 0.22, "warmth": 0.04},
    "shimmer": {"sparkle": 0.3, "brightness": 0.12},
    "grain": {"grain": 0.26, "texture_bias": 0.16},
    "air": {"air": 0.18, "softness": 0.04},
    "mist": {"air": 0.16, "softness": 0.08, "grain": 0.06},
    "rain": {"rain_bias": 0.32, "air": 0.06},
    "wet": {"rain_bias": 0.2, "grain": 0.08, "texture_bias": 0.08},
    "tape": {"grain": 0.2, "warmth": 0.12},
    "traffic": {"texture_bias": 0.18, "grain": 0.08},
    "room": {"room": 0.2, "warmth": 0.06},
    "distant": {"softness": 0.1, "air": 0.06},
    "night": {"sub": 0.08, "warmth": 0.08},
    "low": {"sub": 0.08, "pulse": 0.08},
    "bloom": {"bloom": 0.22, "warmth": 0.08},
    "dusty": {"grain": 0.18, "softness": 0.1},
    "halo": {"sparkle": 0.08, "bloom": 0.1},
    "wash": {"air": 0.14, "room": 0.06},
    "hush": {"room": 0.08, "softness": 0.08, "texture_bias": 0.08},
    "bell": {"glass": 0.18, "sparkle": 0.2, "mallet": 0.12},
    "choir": {"choir": 0.24, "bloom": 0.12, "air": 0.08},
    "organ": {"warmth": 0.12, "bloom": 0.14, "sub": 0.08},
    "string": {"bowed": 0.18, "warmth": 0.06, "bloom": 0.06},
    "brush": {"brush": 0.24, "softness": 0.06},
    "muted": {"softness": 0.08, "brightness": -0.04},
    "veil": {"air": 0.08, "choir": 0.08, "softness": 0.04},
}


class ProceduralAmbientProvider(BaseProvider):
    name = "procedural_dsp"

    def render_section(self, request: SectionRenderRequest) -> SectionRenderResult:
        section_dir = request.session_paths.sections / f"section_{request.section.index:02d}_{request.section.role}"
        section_dir.mkdir(parents=True, exist_ok=True)
        stem_files = {stem: section_dir / f"{stem}.wav" for stem in request.stem_names}
        writers = {
            stem: sf.SoundFile(
                path,
                mode="w",
                samplerate=request.sample_rate,
                channels=request.channels,
                subtype="PCM_24",
            )
            for stem, path in stem_files.items()
        }

        total_frames = request.section.duration_seconds * request.sample_rate
        block_frames = request.block_seconds * request.sample_rate
        root_freq = _note_to_frequency(
            request.profile.tonal_center.get("root", "D"),
            request.profile.tonal_center.get("octave", 3),
        )
        width = float(request.profile.instrumentation.get("stereo_width", 0.35))
        section_state = _build_section_state(request, root_freq, width)

        try:
            frame_cursor = 0
            while frame_cursor < total_frames:
                frames = min(block_frames, total_frames - frame_cursor)
                t = (np.arange(frames, dtype=np.float32) + frame_cursor) / request.sample_rate
                progress = (frame_cursor + np.arange(frames, dtype=np.float32)) / max(1, total_frames)
                stems = self._generate_block(
                    request=request,
                    section_state=section_state,
                    t=t,
                    progress=progress,
                )
                for stem_name, stem_audio in stems.items():
                    writers[stem_name].write(_sanitize_stem_audio(stem_audio, request.channels))
                frame_cursor += frames
        finally:
            for writer in writers.values():
                writer.close()

        return SectionRenderResult(
            section_index=request.section.index,
            section_role=request.section.role,
            provider_name=self.name,
            transition_policy=request.section.transition_policy,
            stem_files=stem_files,
            duration_seconds=request.section.duration_seconds,
            sample_rate=request.sample_rate,
            notes={
                "emotional_goal": request.section.emotional_goal,
                "density": request.section.density,
                "texture_policy": request.section.texture_policy,
                "movement_bias": section_state["movement_bias"],
                "motif_density": section_state["motif_density"],
                "rhythm_intensity": section_state["rhythm_intensity"],
                "rhythm_variant": section_state["rhythm_variant"],
            },
        )

    def _generate_block(
        self,
        request: SectionRenderRequest,
        section_state: dict[str, Any],
        t: np.ndarray,
        progress: np.ndarray,
    ) -> dict[str, np.ndarray]:
        drift = _harmonic_drift(progress, request.section.harmonic_drift_policy)
        macro_energy = _macro_energy_curve(
            progress=progress,
            role=request.section.role,
            pacing_variant=str(section_state["pacing_variant"]),
            movement_bias=float(section_state["movement_bias"]),
            phases=section_state["energy_phases"],
            breath_depth=float(section_state.get("dynamic_breath_depth", 0.0)),
            reference_dark=bool(section_state.get("reference_dark", False)),
        )
        edge = _section_edge_envelope(progress, request.section.role)
        layer_multiplier = _layer_budget_multiplier(request.section.layer_budget, request.section.role)

        drone = _render_drone(
            t=t,
            progress=progress,
            drift=drift,
            section_state=section_state,
            macro_energy=macro_energy,
            edge=edge,
        )
        motion = _render_motion(
            t=t,
            drift=drift,
            section_state=section_state,
            macro_energy=macro_energy,
            edge=edge,
        )
        rhythm = _render_rhythm(
            t=t,
            drift=drift,
            section_state=section_state,
            macro_energy=macro_energy,
            edge=edge,
        )
        texture = _render_texture(
            t=t,
            progress=progress,
            section_state=section_state,
            macro_energy=macro_energy,
            edge=edge,
            sample_rate=request.sample_rate,
        )
        accents = _render_accents(
            t=t,
            drift=drift,
            section_state=section_state,
            macro_energy=macro_energy,
            edge=edge,
        )

        motion *= layer_multiplier["motion"]
        rhythm *= layer_multiplier["rhythm"]
        texture *= layer_multiplier["texture"]
        accents *= layer_multiplier["accents"]

        reference_dark = bool(section_state.get("reference_dark", False))
        drone_width = float(section_state["width"]) * (0.84 if reference_dark else 0.64)
        drone_stereo_drive = float(section_state["stereo_drive"]) * (1.02 if reference_dark else 0.78)
        available = {
            "drone": _stereoize_tonal(
                drone,
                width=drone_width,
                t=t,
                delay_samples=int(section_state["delay_samples"][0]),
                phase_rate=float(section_state["stereo_rates"][0]),
                phase_offset=float(section_state["stereo_phases"][0]),
                stereo_drive=drone_stereo_drive,
            ),
            "motion": _stereoize_tonal(
                motion,
                width=float(section_state["width"]) * 1.02,
                t=t,
                delay_samples=int(section_state["delay_samples"][1]),
                phase_rate=float(section_state["stereo_rates"][1]),
                phase_offset=float(section_state["stereo_phases"][1]),
                stereo_drive=float(section_state["stereo_drive"]) * 1.02,
            ),
            "texture": texture.astype(np.float32),
            "accents": _stereoize_tonal(
                accents,
                width=min(0.98, float(section_state["width"]) + 0.18),
                t=t,
                delay_samples=int(section_state["delay_samples"][2]),
                phase_rate=float(section_state["stereo_rates"][2]),
                phase_offset=float(section_state["stereo_phases"][2]),
                stereo_drive=min(0.62, float(section_state["stereo_drive"]) * 1.12),
            ),
            "rhythm": _stereoize_tonal(
                rhythm,
                width=float(section_state["width"]) * 0.58,
                t=t,
                delay_samples=int(section_state["delay_samples"][3]),
                phase_rate=float(section_state["stereo_rates"][3]),
                phase_offset=float(section_state["stereo_phases"][3]),
                stereo_drive=float(section_state["stereo_drive"]) * 0.92,
            ),
        }
        stems: dict[str, np.ndarray] = {}
        silent = np.zeros((t.shape[0], request.channels), dtype=np.float32)
        for stem_name in request.stem_names:
            stems[stem_name] = available.get(stem_name, silent)
        return stems


def _build_section_state(request: SectionRenderRequest, root_freq: float, width: float) -> dict[str, Any]:
    rng = np.random.default_rng(request.section.variation_seed)
    role = request.section.role
    role_energy = _role_energy(role)
    scale_intervals = SCALE_INTERVALS.get(request.profile.scale_family, SCALE_INTERVALS["aeolian"])
    scale_ratios = _intervals_to_ratios(scale_intervals)
    palette = _derive_instrument_palette(request.profile.instrumentation, request.variation)
    palette = _apply_profile_artifact_guard(palette, request.profile)
    sleep_safe = _is_sleep_safe_profile(request.profile)
    reference_dark = _is_reference_dark_bass_profile(request.profile)
    drone_ratios, motion_ratios = _select_ratio_sets(
        scale_ratios,
        role,
        str(request.variation.get("harmonic_color", "")),
        palette,
        rng,
    )
    if reference_dark:
        drone_ratios = _reference_dark_drone_ratios()
        motion_ratios = _reference_dark_motion_ratios()
    if sleep_safe and not reference_dark:
        drone_ratios = _sleep_safe_drone_ratios(drone_ratios)
        motion_ratios = _sleep_safe_motion_ratios(motion_ratios)
    texture_factors = _texture_policy_factors(
        request.section.texture_policy,
        request.profile.texture_mix,
        str(request.variation.get("texture_variant", "diffuse-air")),
        palette,
    )
    movement_bias = float(request.variation.get("movement_bias", 0.55))
    motif_density = float(request.variation.get("motif_density", 0.62))
    stereo_profile = str(request.variation.get("stereo_profile", "balanced"))
    if sleep_safe:
        width *= 0.88
    if reference_dark:
        width *= 1.18
    width *= {"narrow-core": 0.9, "balanced": 1.0, "wide-tail": 1.12}.get(stereo_profile, 1.0)
    width *= 0.94 + palette["air"] * 0.16 + palette["sparkle"] * 0.08
    pulse_rate = _pulse_rate(request.profile.pulse_density) * (0.86 + palette["pulse"] * 0.42)
    rhythm_variant = str(request.profile.instrumentation.get("rhythm_style", request.variation.get("rhythm_variant", "tidal")))
    rhythm_intensity = float(np.clip(request.profile.instrumentation.get("rhythm_intensity", 0.18), 0.0, 1.0))
    if sleep_safe:
        pulse_rate *= 0.72
        rhythm_intensity *= 0.42
    if reference_dark:
        pulse_rate = float(request.profile.instrumentation.get("breath_rate_hz", 0.028))
        rhythm_intensity *= 0.12
    texture_spread = float(np.clip(0.14 + width * 0.42 + palette["air"] * 0.08 + palette["sparkle"] * 0.05, 0.12, 0.42))
    stereo_drive = float(np.clip(0.26 + width * 0.48 + palette["glass"] * 0.06 + palette["bloom"] * 0.04, 0.22, 0.58))
    if sleep_safe:
        texture_spread = float(np.clip(texture_spread - 0.04, 0.1, 0.32))
        stereo_drive = float(np.clip(stereo_drive - 0.05, 0.18, 0.44))
    if reference_dark:
        texture_spread = float(np.clip(texture_spread + 0.12, 0.18, 0.46))
        stereo_drive = float(np.clip(stereo_drive + 0.1, 0.28, 0.62))

    motion_events = _schedule_motion_events(
        duration_seconds=request.section.duration_seconds,
        density=float(request.section.density),
        motif_density=motif_density,
        role_energy=role_energy,
        ratios=motion_ratios,
        pulse_strength=palette["pulse"],
        artifact_sensitivity=float(palette.get("artifact_sensitivity", 0.0)),
        sleep_safe=sleep_safe,
        reference_dark=reference_dark,
        rng=rng,
    )
    accent_events = _schedule_accent_events(
        duration_seconds=request.section.duration_seconds,
        density=float(request.section.density),
        role=role,
        ratios=motion_ratios,
        accent_variant=str(request.variation.get("accent_variant", "halo")),
        palette=palette,
        sleep_safe=sleep_safe,
        rng=rng,
    )
    rhythm_events = _schedule_rhythm_events(
        duration_seconds=request.section.duration_seconds,
        density=float(request.section.density),
        role=role,
        pulse_density=request.profile.pulse_density,
        pulse_rate=pulse_rate,
        rhythm_variant=rhythm_variant,
        rhythm_intensity=rhythm_intensity,
        ratios=motion_ratios,
        rng=rng,
    )

    return {
        "root_freq": root_freq,
        "width": min(0.98, max(0.1, width)),
        "texture_spread": texture_spread,
        "stereo_drive": stereo_drive,
        "role_energy": role_energy,
        "movement_bias": movement_bias,
        "motif_density": motif_density,
        "pacing_variant": request.variation.get("pacing_variant", "slow-bloom"),
        "palette": palette,
        "pulse_rate": pulse_rate,
        "rhythm_variant": rhythm_variant,
        "rhythm_intensity": rhythm_intensity,
        "reference_dark": reference_dark,
        "dynamic_breath_depth": float(np.clip(request.profile.instrumentation.get("dynamic_breath_depth", 0.0), 0.0, 0.85)),
        "breath_rate_hz": float(request.profile.instrumentation.get("breath_rate_hz", 0.067)),
        "drone_ratios": drone_ratios,
        "motion_ratios": motion_ratios,
        "texture_factors": texture_factors,
        "motion_events": motion_events,
        "accent_events": accent_events,
        "rhythm_events": rhythm_events,
        "sleep_safe": sleep_safe,
        "drone_weights": _drone_weights(drone_ratios, reference_dark, rng),
        "drone_phases": rng.uniform(0.0, 2.0 * np.pi, size=len(drone_ratios)).astype(np.float32),
        "drone_rates": rng.uniform(0.0025, 0.017, size=len(drone_ratios)).astype(np.float32),
        "energy_phases": rng.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32),
        "stereo_rates": rng.uniform(0.01, 0.08, size=4).astype(np.float32),
        "stereo_phases": rng.uniform(0.0, 2.0 * np.pi, size=4).astype(np.float32),
        "delay_samples": rng.integers(24, 72, size=4) if reference_dark else rng.integers(9, 31, size=4),
        "motion_bed_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        "rhythm_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        "rhythm_noise_seed": int(rng.integers(0, 1_000_000)),
        "texture_motion_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
    }


def _render_drone(
    t: np.ndarray,
    progress: np.ndarray,
    drift: np.ndarray,
    section_state: dict[str, Any],
    macro_energy: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    root_freq = float(section_state["root_freq"])
    drone_ratios = np.asarray(section_state["drone_ratios"], dtype=np.float32)
    weights = np.asarray(section_state["drone_weights"], dtype=np.float32)
    phases = np.asarray(section_state["drone_phases"], dtype=np.float32)
    rates = np.asarray(section_state["drone_rates"], dtype=np.float32)
    palette = dict(section_state["palette"])
    sample_rate = _estimate_sample_rate_from_t(t)
    sleep_safe = bool(section_state.get("sleep_safe", False))
    reference_dark = bool(section_state.get("reference_dark", False))

    drone = np.zeros_like(t, dtype=np.float32)
    color_blend = 0.45 + 0.55 * np.sin(2.0 * np.pi * (progress * (0.7 + 0.5 * float(section_state["movement_bias"])) + 0.17))
    if sleep_safe:
        color_blend = 0.54 + 0.34 * np.sin(2.0 * np.pi * (progress * 0.32 + 0.11))
    if reference_dark:
        color_blend = 0.48 + 0.40 * np.sin(2.0 * np.pi * (progress * 0.22 + 0.09))
    wobble_depth = 0.0025 + 0.0024 * palette["bowed"]
    artifact_sensitivity = float(palette.get("artifact_sensitivity", 0.0))
    if sleep_safe:
        wobble_depth *= 0.72
    if reference_dark:
        wobble_depth *= 0.42
    shimmer_depth = (0.04 + 0.035 * palette["sparkle"]) * (1.0 - 0.72 * artifact_sensitivity)
    if sleep_safe:
        shimmer_depth *= 0.22
    if reference_dark:
        shimmer_depth *= 0.18
    for index, ratio in enumerate(drone_ratios):
        wobble = 1.0 + wobble_depth * np.sin(2.0 * np.pi * rates[index] * t + phases[index])
        shimmer = shimmer_depth * np.sin(2.0 * np.pi * (rates[index] * 3.6) * t + phases[index] * 0.7)
        carrier = np.sin(2.0 * np.pi * root_freq * ratio * wobble * t * drift + phases[index] + shimmer)
        weight_curve = weights[index] * ((0.8 + 0.2 * color_blend) if index % 2 else (1.0 - 0.14 * color_blend))
        drone += carrier.astype(np.float32) * weight_curve.astype(np.float32)

    sub = np.sin(2.0 * np.pi * root_freq * 0.5 * t + float(phases[0]) * 0.5)
    halo = np.sin(2.0 * np.pi * root_freq * 2.0 * t * (1.0 + 0.0018 * np.sin(2.0 * np.pi * 0.022 * t)))
    bowed = np.sin(2.0 * np.pi * root_freq * 1.5 * t * drift + float(phases[1]) * 0.6)
    body = np.sin(2.0 * np.pi * root_freq * 2.378414 * t * drift + float(phases[2]) * 0.33)
    upper_body = np.sin(2.0 * np.pi * root_freq * 2.828427 * t * drift + float(phases[2]) * 0.51)
    if reference_dark:
        drone = (
            (0.052 + 0.014 * palette["warmth"]) * drone
            + (0.01 + 0.012 * palette["sub"]) * sub
            + (0.006 + 0.01 * palette["bloom"]) * halo * 0.42
            + (0.003 + 0.006 * palette["bowed"]) * bowed
            + (0.026 + 0.03 * palette["warmth"]) * body
            + (0.018 + 0.02 * palette["warmth"]) * upper_body
        )
    else:
        drone = (
            (0.064 + 0.02 * palette["warmth"]) * drone
            + (0.02 + 0.022 * palette["sub"]) * sub
            + (0.01 + 0.016 * palette["bloom"]) * halo * (0.58 if sleep_safe else 1.0)
            + (0.004 + 0.011 * palette["bowed"]) * bowed
        )
    soften_amount = 0.08 + 0.14 * palette["softness"] + 0.26 * artifact_sensitivity
    if sleep_safe:
        soften_amount += 0.14
    if reference_dark:
        soften_amount += 0.1
    cutoff_hz = 1150.0 if reference_dark else (3100.0 if sleep_safe else 4200.0)
    drone = _soften_tone(drone, sample_rate=sample_rate, amount=soften_amount, cutoff_hz=cutoff_hz)
    drone = _soft_clip(drone, 1.2)
    if reference_dark:
        breath_depth = float(section_state.get("dynamic_breath_depth", 0.0))
        breath_rate = float(section_state.get("breath_rate_hz", 0.067))
        breath = 1.0 - breath_depth * 0.18 + breath_depth * 0.22 * (
            0.5 + 0.5 * np.sin(2.0 * np.pi * breath_rate * 0.42 * t + float(phases[1]))
        )
        slow_bloom = 0.86 + 0.14 * np.sin(2.0 * np.pi * 0.0032 * t + float(phases[2]))
        drone *= breath.astype(np.float32) * slow_bloom.astype(np.float32) * macro_energy * edge
    else:
        drone *= (0.82 + 0.18 * np.sin(2.0 * np.pi * 0.006 * t + float(phases[1]))) * macro_energy * edge
    return drone.astype(np.float32)


def _render_motion(
    t: np.ndarray,
    drift: np.ndarray,
    section_state: dict[str, Any],
    macro_energy: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    motion = np.zeros_like(t, dtype=np.float32)
    sample_rate = _estimate_sample_rate_from_t(t)
    root_freq = float(section_state["root_freq"])
    block_start = float(t[0])
    block_end = float(t[-1])
    palette = dict(section_state["palette"])
    artifact_sensitivity = float(palette.get("artifact_sensitivity", 0.0))
    sleep_safe = bool(section_state.get("sleep_safe", False))
    reference_dark = bool(section_state.get("reference_dark", False))
    pulse_lattice = _soft_lattice(
        t=t,
        rate=float(section_state["pulse_rate"]),
        phase=float(section_state["rhythm_phase"]),
        sharpness=(1.25 + palette["pulse"] * 0.8) if reference_dark else ((1.8 + palette["pulse"] * 1.4) if sleep_safe else (2.6 + palette["pulse"] * 2.6)),
    )

    for event in section_state["motion_events"]:
        center = float(event["center"])
        span = float(event["duration"]) * 1.9
        if center + span < block_start or center - span > block_end:
            continue
        local = t - center
        sigma = max(0.45, float(event["duration"]) * 0.3)
        env = np.exp(-0.5 * (local / sigma) ** 2).astype(np.float32)
        glide = 1.0 + float(event["glide"]) * np.clip(local, -sigma, sigma)
        freq = root_freq * float(event["ratio"]) * glide
        phase = 2.0 * np.pi * freq * t * drift + float(event["phase"])
        carrier = np.sin(phase)
        if reference_dark:
            carrier += (0.05 + 0.08 * palette["bowed"]) * np.sin(phase * 0.5 + float(event["phase"]) * 0.17)
            carrier += (0.015 + 0.035 * palette["bloom"]) * np.sin(phase * 0.25 + 0.13)
        elif sleep_safe:
            carrier += (0.08 + 0.10 * palette["bowed"]) * np.sin(phase * 0.5 + float(event["phase"]) * 0.21)
            carrier += (0.02 + 0.05 * palette["bloom"]) * np.sin(phase * 0.25 + 0.18)
        else:
            carrier += (0.18 + 0.16 * palette["bowed"]) * np.sin(phase * 0.5 + float(event["phase"]) * 0.37)
            carrier += (0.06 + 0.18 * palette["glass"]) * (1.0 - 0.78 * artifact_sensitivity) * np.sin(phase * (1.45 + 0.4 * palette["brightness"]) + 0.24)
        flutter_depth = (0.08 + 0.06 * palette["pulse"]) * (1.0 - 0.62 * artifact_sensitivity)
        if reference_dark:
            flutter_depth *= 0.22
        elif sleep_safe:
            flutter_depth *= 0.34
        flutter = 1.0 + flutter_depth * np.sin(2.0 * np.pi * float(event["mod_rate"]) * t + float(event["phase"]) * 0.3)
        gate = 0.78 + 0.22 * pulse_lattice * (0.55 + palette["pulse"] * 0.65)
        if reference_dark:
            gate = 0.88 + 0.12 * pulse_lattice * (0.78 + palette["pulse"] * 0.16)
        elif sleep_safe:
            gate = 0.9 + 0.1 * pulse_lattice * (0.48 + palette["pulse"] * 0.32)
        motion += float(event["amplitude"]) * env * carrier.astype(np.float32) * flutter.astype(np.float32) * gate.astype(np.float32)

    if len(section_state["motion_ratios"]):
        bed_ratio = float(section_state["motion_ratios"][0])
        bed = np.sin(2.0 * np.pi * root_freq * bed_ratio * t * drift + float(section_state["motion_bed_phase"]))
        motion += (0.005 + 0.014 * palette["bloom"]) * bed.astype(np.float32) * (0.72 + 0.28 * pulse_lattice) * (0.78 if reference_dark else (0.58 if sleep_safe else 1.0))
        choir_bed = np.sin(2.0 * np.pi * root_freq * (bed_ratio * 0.75) * t * drift + float(section_state["motion_bed_phase"]) * 0.6)
        motion += (0.003 + 0.011 * palette["choir"]) * choir_bed.astype(np.float32) * (0.68 + 0.32 * macro_energy) * (0.72 + 0.28 * pulse_lattice) * (0.66 if reference_dark else (0.52 if sleep_safe else 1.0))
        if reference_dark:
            body_phase = float(section_state["motion_bed_phase"])
            body_pad = (
                0.82 * np.sin(2.0 * np.pi * root_freq * 2.378414 * t * drift + body_phase)
                + 0.58 * np.sin(2.0 * np.pi * root_freq * 2.828427 * t * drift + body_phase * 0.71)
                + 0.36 * np.sin(2.0 * np.pi * root_freq * 3.174802 * t * drift + body_phase * 1.13)
            )
            body_env = 0.78 + 0.22 * np.sin(2.0 * np.pi * 0.0065 * t + body_phase * 0.37)
            motion += (0.028 + 0.026 * palette["warmth"]) * body_pad.astype(np.float32) * body_env.astype(np.float32) * (0.78 + 0.22 * macro_energy)

    soften_amount = 0.12 + 0.20 * palette["softness"] + 0.34 * artifact_sensitivity
    if sleep_safe:
        soften_amount += 0.22
    if reference_dark:
        soften_amount += 0.12
    cutoff_hz = 1250.0 if reference_dark else (2500.0 if sleep_safe else 3600.0)
    motion = _soften_tone(motion, sample_rate=sample_rate, amount=soften_amount, cutoff_hz=cutoff_hz)
    motion = _soft_clip(motion, 1.15)
    motion *= macro_energy * edge
    return motion.astype(np.float32)


def _render_rhythm(
    t: np.ndarray,
    drift: np.ndarray,
    section_state: dict[str, Any],
    macro_energy: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    rhythm = np.zeros_like(t, dtype=np.float32)
    sample_rate = _estimate_sample_rate_from_t(t)
    root_freq = float(section_state["root_freq"])
    block_start = float(t[0])
    block_end = float(t[-1])
    palette = dict(section_state["palette"])
    intensity = float(section_state["rhythm_intensity"])
    if intensity <= 0.0:
        return rhythm
    variant_weights = _rhythm_variant_weights(str(section_state["rhythm_variant"]))
    block_rng = np.random.default_rng(int(section_state["rhythm_noise_seed"]) + int(t[0] * 10.0))
    brush_noise = block_rng.normal(0.0, 1.0, size=t.shape[0]).astype(np.float32)
    brush_noise = _smooth_signal(brush_noise) - 0.72 * _smooth_signal(_smooth_signal(brush_noise))
    brush_noise = _normalize(brush_noise)

    pulse_lattice = _soft_lattice(
        t=t,
        rate=float(section_state["pulse_rate"]),
        phase=float(section_state["rhythm_phase"]),
        sharpness=3.2 + palette["pulse"] * 3.0,
    )
    bed = np.sin(2.0 * np.pi * root_freq * 0.5 * t * drift + float(section_state["rhythm_phase"]))
    undertow = np.sin(2.0 * np.pi * root_freq * 0.25 * t * drift + float(section_state["rhythm_phase"]) * 0.45)
    rhythm += (0.004 + 0.014 * intensity) * bed.astype(np.float32) * pulse_lattice
    rhythm += (0.002 + 0.008 * intensity) * undertow.astype(np.float32) * (0.74 + 0.26 * pulse_lattice)

    for event in section_state["rhythm_events"]:
        center = float(event["center"])
        span = float(event["duration"]) * 3.0
        if center + span < block_start or center - span > block_end:
            continue
        local = t - center
        env = _pulse_envelope(local, float(event["duration"]), softness=palette["softness"])
        phase = 2.0 * np.pi * root_freq * float(event["ratio"]) * t * drift + float(event["phase"])
        tone = np.sin(phase)
        tone += (0.12 + 0.14 * palette["glass"]) * np.sin(phase * 1.95 + 0.21)
        thump = np.sin(phase * 0.5 + 0.18)
        click = np.sin(2.0 * np.pi * root_freq * (8.0 + 4.0 * palette["brightness"]) * t + float(event["phase"]))
        brush = brush_noise * (0.82 + 0.18 * np.sin(phase * 0.25 + 0.3)).astype(np.float32)
        event_signal = env * (
            variant_weights["tone"] * tone.astype(np.float32)
            + variant_weights["thump"] * thump.astype(np.float32)
            + variant_weights["brush"] * brush.astype(np.float32) * (0.42 + palette["brush"] * 0.58)
            + variant_weights["click"] * click.astype(np.float32) * (0.24 + 0.5 * palette["glass"])
        )
        rhythm += float(event["amplitude"]) * event_signal

    rhythm = _smooth_signal(rhythm)
    soften_amount = 0.18 + 0.22 * palette["softness"] + 0.08 * (1.0 - palette["brightness"])
    rhythm = _soften_tone(rhythm, sample_rate=sample_rate, amount=soften_amount, cutoff_hz=5400.0)
    rhythm = _soft_clip(rhythm, 1.05)
    rhythm *= macro_energy * edge * (0.78 + 0.22 * palette["pulse"])
    return rhythm.astype(np.float32)


def _render_texture(
    t: np.ndarray,
    progress: np.ndarray,
    section_state: dict[str, Any],
    macro_energy: np.ndarray,
    edge: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    factors = dict(section_state["texture_factors"])
    palette = dict(section_state["palette"])
    sleep_safe = bool(section_state.get("sleep_safe", False))
    reference_dark = bool(section_state.get("reference_dark", False))
    reference_dark = bool(section_state.get("reference_dark", False))
    n = t.shape[0]
    block_seed = int(section_state["motion_events"][0]["seed"]) if section_state["motion_events"] else 0
    rng_left = np.random.default_rng(block_seed + int(t[0] * sample_rate) + 19)
    rng_right = np.random.default_rng(block_seed + int(t[0] * sample_rate) + 137)
    left = _texture_channel(
        rng_left,
        n,
        sample_rate,
        factors,
        palette,
        progress,
        macro_energy,
        float(section_state["texture_motion_phase"]),
        sleep_safe=sleep_safe,
        reference_dark=reference_dark,
    )
    right = _texture_channel(
        rng_right,
        n,
        sample_rate,
        factors,
        palette,
        progress[::-1],
        macro_energy[::-1],
        float(section_state["texture_motion_phase"]) + 1.2,
        sleep_safe=sleep_safe,
        reference_dark=reference_dark,
    )[::-1]
    spread = float(section_state.get("texture_spread", 0.22))
    if sleep_safe:
        spread *= 0.42
    if reference_dark:
        spread *= 1.18
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    side_cloud = _smooth_signal(side)
    air_swirl = np.sin(2.0 * np.pi * (progress * (0.22 + palette["air"] * (0.18 if sleep_safe else 0.8)) + float(section_state["texture_motion_phase"]) * 0.17))
    halo = _smooth_signal(side_cloud * air_swirl.astype(np.float32))
    decorrelated = spread * (0.8 * side_cloud + (0.08 if sleep_safe else 0.22) * halo + (0.04 if sleep_safe else 0.12) * side * air_swirl.astype(np.float32))
    texture = np.column_stack([mid * (1.0 - 0.04 * spread) + decorrelated, mid * (1.0 - 0.04 * spread) - decorrelated]).astype(np.float32)
    texture *= edge[:, None]
    return texture.astype(np.float32)


def _render_accents(
    t: np.ndarray,
    drift: np.ndarray,
    section_state: dict[str, Any],
    macro_energy: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    accents = np.zeros_like(t, dtype=np.float32)
    sample_rate = _estimate_sample_rate_from_t(t)
    root_freq = float(section_state["root_freq"])
    block_start = float(t[0])
    block_end = float(t[-1])
    palette = dict(section_state["palette"])
    sleep_safe = bool(section_state.get("sleep_safe", False))
    reference_dark = bool(section_state.get("reference_dark", False))
    for event in section_state["accent_events"]:
        center = float(event["center"])
        span = float(event["duration"]) * 2.6
        if center + span < block_start or center - span > block_end:
            continue
        local = t - center
        sigma = max(0.18, float(event["duration"]) * (0.18 + 0.08 * palette["softness"]))
        env = np.exp(-0.5 * (local / sigma) ** 2).astype(np.float32)
        freq = root_freq * float(event["ratio"]) * (1.0 + float(event["glide"]) * np.clip(local, -sigma, sigma))
        phase = 2.0 * np.pi * freq * t * drift + float(event["phase"])
        shimmer = np.sin(phase)
        shimmer += (0.08 + 0.18 * palette["sparkle"]) * np.sin(phase * (1.3 + 0.45 * palette["brightness"]) + 0.3) * (0.14 if reference_dark else (0.24 if sleep_safe else 1.0))
        shimmer += (0.04 + 0.14 * palette["mallet"]) * np.sin(phase * 0.5 + 0.15) * (0.28 if reference_dark else (0.4 if sleep_safe else 1.0))
        accents += float(event["amplitude"]) * (0.55 if reference_dark else 1.0) * env * shimmer.astype(np.float32)

    accents = _smooth_signal(accents)
    soften_amount = 0.2 + 0.26 * palette["softness"] + 0.06 * palette["glass"]
    if sleep_safe:
        soften_amount += 0.16
    if reference_dark:
        soften_amount += 0.18
    cutoff_hz = 1250.0 if reference_dark else (3000.0 if sleep_safe else 5000.0)
    accents = _soften_tone(accents, sample_rate=sample_rate, amount=soften_amount, cutoff_hz=cutoff_hz)
    accents = _soft_clip(accents, 0.95)
    accents *= macro_energy * edge
    return accents.astype(np.float32)


def _texture_channel(
    rng: np.random.Generator,
    frames: int,
    sample_rate: int,
    factors: dict[str, float],
    palette: dict[str, float],
    progress: np.ndarray,
    macro_energy: np.ndarray,
    phase_offset: float,
    sleep_safe: bool = False,
    reference_dark: bool = False,
) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=frames).astype(np.float32)
    if reference_dark:
        white = _one_pole_lowpass(white, sample_rate=sample_rate, cutoff_hz=520.0)
    elif sleep_safe:
        white = _one_pole_lowpass(white, sample_rate=sample_rate, cutoff_hz=1600.0)
    freqs = np.fft.rfftfreq(frames, d=1.0 / sample_rate)
    base_spec = np.fft.rfft(white)
    if reference_dark:
        room = _spectral_color(base_spec, freqs, 1.0 / (1.0 + (freqs / 210.0) ** 2.2))
        low_velvet = _spectral_color(base_spec, freqs, np.exp(-((freqs - 260.0) / 180.0) ** 2))
        tape = _spectral_color(base_spec, freqs, 1.0 / (1.0 + (freqs / 760.0) ** 2.8))
        motion = np.sin(2.0 * np.pi * (progress * (0.18 + palette["texture_bias"] * 0.06) + 0.05 + phase_offset))
        texture = (
            (0.006 + 0.006 * palette["room"]) * factors["room"] * room
            + (0.003 + 0.003 * palette["softness"]) * factors["room"] * low_velvet * (0.84 + 0.16 * motion)
            + (0.001 + 0.002 * palette["grain"]) * factors["tape"] * np.tanh(tape * 0.9)
        )
        rain_amount = 0.0
    elif sleep_safe:
        room = _spectral_color(base_spec, freqs, 1.0 / (1.0 + (freqs / 420.0) ** 1.6))
        hush = _spectral_color(base_spec, freqs, np.exp(-((freqs - 820.0) / 560.0) ** 2))
        veil = _spectral_color(base_spec, freqs, np.exp(-((freqs - 1450.0) / 760.0) ** 2))
        tape = _spectral_color(base_spec, freqs, 1.0 / (1.0 + (freqs / 1450.0) ** 2.4))
        motion = np.sin(2.0 * np.pi * (progress * (0.32 + palette["texture_bias"] * 0.18) + 0.07 + phase_offset))
        texture = (
            (0.02 + 0.012 * palette["room"]) * factors["room"] * room
            + (0.005 + 0.004 * palette["softness"]) * factors["room"] * hush * (0.9 + 0.1 * motion)
            + (0.003 + 0.004 * palette["bloom"]) * min(0.35, factors["air"]) * veil
            + (0.001 + 0.003 * palette["grain"]) * factors["tape"] * np.tanh(tape * 1.15)
        )
        rain_amount = 0.0
    else:
        room = _spectral_color(base_spec, freqs, 1.0 / (1.0 + (freqs / 540.0) ** 1.8))
        air = _spectral_color(base_spec, freqs, (freqs / 2400.0) / (1.0 + (freqs / 5200.0) ** 2.2))
        silk = _spectral_color(base_spec, freqs, np.exp(-((freqs - 3200.0) / 1900.0) ** 2))
        hush = _spectral_color(base_spec, freqs, np.exp(-((freqs - 1150.0) / 920.0) ** 2))
        tape = _spectral_color(base_spec, freqs, 1.0 / np.sqrt(np.maximum(freqs, 55.0)))
        motion = np.sin(2.0 * np.pi * (progress * (1.4 + palette["texture_bias"]) + 0.11 + phase_offset))
        shimmer_motion = np.sin(2.0 * np.pi * (progress * (2.2 + palette["sparkle"]) + phase_offset * 0.7))
        texture = (
            (0.018 + 0.01 * palette["room"]) * factors["room"] * room
            + (0.008 + 0.007 * palette["air"]) * factors["air"] * air * (0.72 + 0.24 * motion)
            + (0.003 + 0.006 * palette["sparkle"]) * factors["air"] * silk * (0.8 + 0.2 * shimmer_motion)
            + (0.004 + 0.006 * palette["texture_bias"]) * factors["room"] * hush
            + (0.007 + 0.007 * palette["grain"]) * factors["tape"] * np.tanh(tape * 1.6)
        )
        rain_amount = float(np.clip(factors["rain"] + palette["rain_bias"] * 0.35, 0.0, 1.0))
    if rain_amount > 0.02:
        rain = np.zeros(frames, dtype=np.float32)
        hit_probability = 0.00018 + rain_amount * 0.00062
        positions = np.where(rng.random(frames) < hit_probability)[0][: max(2, frames // 3800)]
        for position in positions:
            duration = int(min(frames - position, rng.integers(700, 3000)))
            if duration <= 0:
                continue
            decay = np.exp(-np.linspace(0.0, 7.4, duration)).astype(np.float32)
            burst = rng.uniform(0.16, 0.46) * decay * rng.uniform(0.68, 1.0, size=duration).astype(np.float32)
            rain[position : position + duration] += burst
        texture += (0.026 + 0.015 * palette["air"]) * rain_amount * _normalize(rain)

    texture *= (0.65 + 0.35 * macro_energy).astype(np.float32)
    soften_amount = 0.12 + 0.18 * palette["softness"] + 0.12 * factors["air"] + 0.08 * rain_amount
    if sleep_safe:
        soften_amount += 0.18
    if reference_dark:
        soften_amount += 0.14
    cutoff_hz = 780.0 if reference_dark else (2600.0 if sleep_safe else 6200.0)
    texture = _soften_tone(texture, sample_rate=sample_rate, amount=soften_amount, cutoff_hz=cutoff_hz)
    return _soft_clip(texture.astype(np.float32), 0.92)


def _pulse_rate(pulse_density: str) -> float:
    mapping = {
        "none": 0.0,
        "minimal": 0.03,
        "low": 0.045,
        "soft": 0.06,
        "medium": 0.08,
    }
    return mapping.get(pulse_density, 0.045)


def _rhythm_variant_weights(rhythm_variant: str) -> dict[str, float]:
    return {
        "tidal": {"tone": 0.48, "thump": 0.28, "brush": 0.18, "click": 0.012},
        "muted-step": {"tone": 0.44, "thump": 0.24, "brush": 0.24, "click": 0.008},
        "glass-pulse": {"tone": 0.42, "thump": 0.18, "brush": 0.1, "click": 0.028},
        "heartbeat": {"tone": 0.34, "thump": 0.4, "brush": 0.14, "click": 0.004},
        "rainwalk": {"tone": 0.38, "thump": 0.22, "brush": 0.26, "click": 0.01},
        "breathing-swell": {"tone": 0.62, "thump": 0.18, "brush": 0.04, "click": 0.0},
    }.get(rhythm_variant, {"tone": 0.46, "thump": 0.24, "brush": 0.16, "click": 0.01})


def _role_energy(role: str) -> float:
    mapping = {
        "intro": 0.52,
        "settle": 0.61,
        "drift_a": 0.75,
        "drift_b": 0.82,
        "sparse_break": 0.34,
        "return": 0.69,
        "low_energy_tail": 0.24,
    }
    return mapping.get(role, 0.65)


def _derive_instrument_palette(instrumentation: dict[str, Any], variation: dict[str, Any]) -> dict[str, float]:
    palette = BASE_PALETTE.copy()
    names = list(instrumentation.get("primary", []))
    names += list(instrumentation.get("secondary", []))
    names += list(variation.get("primary_instruments", []))
    combined = " ".join(name.lower() for name in names)
    for keyword, deltas in KEYWORD_PALETTE.items():
        if keyword not in combined:
            continue
        for field, delta in deltas.items():
            palette[field] = palette.get(field, 0.0) + delta
    if str(variation.get("texture_variant", "")) == "mist-rain":
        palette["rain_bias"] += 0.1
        palette["air"] += 0.06
    if str(variation.get("rhythm_variant", "")) in {"muted-step", "heartbeat"}:
        palette["pulse"] += 0.08
    if str(variation.get("rhythm_variant", "")) == "glass-pulse":
        palette["glass"] += 0.1
        palette["sparkle"] += 0.08
    for field in list(palette):
        palette[field] = float(np.clip(palette[field], 0.0, 1.0))
    return palette


def _apply_profile_artifact_guard(palette: dict[str, float], profile: Any) -> dict[str, float]:
    guarded = palette.copy()
    mood_text = " ".join(
        [
            str(getattr(profile, "mood", "")),
            " ".join(str(item) for item in getattr(profile, "forbidden_artifacts", [])),
            " ".join(str(item) for item in getattr(profile, "instrumentation", {}).get("primary", [])),
            " ".join(str(item) for item in getattr(profile, "instrumentation", {}).get("secondary", [])),
        ]
    ).lower()
    sensitivity = 0.0
    if any(token in mood_text for token in ["sleep", "rest", "insomnia", "solitude"]):
        sensitivity += 0.28
    if any(token in mood_text for token in ["harsh", "brittle", "bright"]):
        sensitivity += 0.24
    if any(token in mood_text for token in ["percussion", "rhythmic", "click", "shimmer"]):
        sensitivity += 0.16
    sensitivity = float(np.clip(sensitivity, 0.0, 0.8))
    guarded["brightness"] = float(np.clip(guarded["brightness"] * (1.0 - 0.48 * sensitivity), 0.0, 1.0))
    guarded["sparkle"] = float(np.clip(guarded["sparkle"] * (1.0 - 0.72 * sensitivity), 0.0, 1.0))
    guarded["glass"] = float(np.clip(guarded["glass"] * (1.0 - 0.78 * sensitivity), 0.0, 1.0))
    guarded["pulse"] = float(np.clip(guarded["pulse"] * (1.0 - 0.44 * sensitivity), 0.0, 1.0))
    guarded["brush"] = float(np.clip(guarded["brush"] * (1.0 - 0.52 * sensitivity), 0.0, 1.0))
    guarded["air"] = float(np.clip(guarded["air"] * (1.0 - 0.58 * sensitivity), 0.0, 1.0))
    guarded["grain"] = float(np.clip(guarded["grain"] * (1.0 - 0.70 * sensitivity), 0.0, 1.0))
    guarded["texture_bias"] = float(np.clip(guarded["texture_bias"] * (1.0 - 0.60 * sensitivity), 0.0, 1.0))
    guarded["softness"] = float(np.clip(guarded["softness"] + 0.22 * sensitivity, 0.0, 1.0))
    guarded["warmth"] = float(np.clip(guarded["warmth"] + 0.12 * sensitivity, 0.0, 1.0))
    guarded["bloom"] = float(np.clip(guarded["bloom"] + 0.10 * sensitivity, 0.0, 1.0))
    guarded["artifact_sensitivity"] = sensitivity
    return guarded


def _is_sleep_safe_profile(profile: Any) -> bool:
    mood_text = " ".join(
        [
            str(getattr(profile, "mood", "")),
            " ".join(str(item) for item in getattr(profile, "forbidden_artifacts", [])),
            " ".join(str(item) for item in getattr(profile, "instrumentation", {}).get("primary", [])),
            " ".join(str(item) for item in getattr(profile, "instrumentation", {}).get("secondary", [])),
        ]
    ).lower()
    return any(token in mood_text for token in ["sleep", "rest", "insomnia", "solitude", "calm mind"])


def _is_reference_dark_bass_profile(profile: Any) -> bool:
    instrumentation = getattr(profile, "instrumentation", {})
    return str(instrumentation.get("production_dna", "")).lower() == "reference_dark_bass"


def _reference_dark_drone_ratios() -> np.ndarray:
    # C#3 profile anchor yields roughly 69, 82, 98, 110, 124, 139, 165 Hz.
    return np.asarray([0.5, 0.594604, 0.707107, 0.793701, 0.890899, 1.0, 1.189207], dtype=np.float32)


def _reference_dark_motion_ratios() -> np.ndarray:
    return np.asarray([0.594604, 0.66742, 0.707107, 0.793701, 0.890899, 1.0, 1.189207], dtype=np.float32)


def _drone_weights(ratios: np.ndarray, reference_dark: bool, rng: np.random.Generator) -> np.ndarray:
    if not reference_dark:
        return rng.uniform(0.75, 1.15, size=len(ratios)).astype(np.float32)
    weights = []
    for ratio in ratios:
        value = float(ratio)
        if value < 0.62:
            weight = 0.48
        elif value < 0.82:
            weight = 0.64
        elif value < 0.94:
            weight = 0.98
        elif value < 1.08:
            weight = 1.04
        else:
            weight = 1.18
        weights.append(weight * float(rng.uniform(0.92, 1.08)))
    return np.asarray(weights, dtype=np.float32)


def _sleep_safe_motion_ratios(ratios: np.ndarray) -> np.ndarray:
    safe = sorted(float(ratio) for ratio in ratios if 0.45 <= float(ratio) <= 1.52)
    if len(safe) < 4:
        safe = sorted(float(ratio) for ratio in ratios if 0.45 <= float(ratio) <= 2.0)
    unique: list[float] = []
    for ratio in safe:
        if any(abs(ratio - existing) < 0.04 for existing in unique):
            continue
        unique.append(ratio)
    return np.asarray(unique[:6] if unique else ratios[:6], dtype=np.float32)


def _sleep_safe_drone_ratios(ratios: np.ndarray) -> np.ndarray:
    safe = [float(ratio) for ratio in ratios if 0.48 <= float(ratio) <= 1.52]
    ordered: list[float] = []
    for ratio in sorted(safe):
        if any(abs(ratio - existing) < 0.04 for existing in ordered):
            continue
        ordered.append(ratio)
    chosen = ordered[:4] if ordered else [float(ratio) for ratio in ratios[:4]]
    return np.asarray(chosen, dtype=np.float32)


def _select_ratio_sets(
    scale_ratios: list[float],
    role: str,
    harmonic_color: str,
    palette: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ratio_pool = np.asarray(scale_ratios, dtype=np.float32)
    if len(ratio_pool) < 4:
        ratio_pool = np.asarray([1.0, 1.2, 1.5, 1.8], dtype=np.float32)

    color_index = {
        "root-fifth": [0, 4, 6],
        "modal-third": [0, 2, 4],
        "minor-seventh": [0, 4, 5],
        "suspended-second": [0, 1, 4],
    }.get(harmonic_color, [0, 2, 4])
    role_index = {
        "intro": [0, 2, 4],
        "settle": [0, 2, 4],
        "drift_a": [0, 3, 4],
        "drift_b": [0, 4, 5],
        "sparse_break": [0, 2, 3],
        "return": [0, 2, 5],
        "low_energy_tail": [0, 2, 4],
    }.get(role, [0, 2, 4])

    drone: list[float] = []
    for index in role_index:
        drone.append(float(ratio_pool[min(index, len(ratio_pool) - 1)]))
    octave_weights = np.array([0.24, 0.54, 0.22], dtype=np.float32)
    if palette["sub"] > 0.45:
        octave_weights = np.array([0.34, 0.5, 0.16], dtype=np.float32)
    for index in color_index[:2]:
        drone.append(float(ratio_pool[min(index, len(ratio_pool) - 1)] * rng.choice([0.5, 1.0, 2.0], p=octave_weights)))

    motion: list[float] = []
    for ratio in ratio_pool:
        motion.append(float(ratio))
        motion.append(float(ratio * 2.0))
    if palette["glass"] > 0.28:
        motion.extend(float(ratio * 1.5) for ratio in ratio_pool[:3])
    rng.shuffle(motion)
    return np.asarray(drone[:4], dtype=np.float32), np.asarray(motion[:8], dtype=np.float32)


def _texture_policy_factors(
    policy: str,
    profile_mix: dict[str, float],
    variant: str,
    palette: dict[str, float],
) -> dict[str, float]:
    factors = {"air": 0.45, "room": 0.35, "rain": 0.0, "tape": 0.18}
    factors.update(TEXTURE_PRESETS.get(policy, {}))
    room_value = float(profile_mix.get("room_tone", profile_mix.get("room", factors["room"])))
    mix_alias = {
        "air": float(profile_mix.get("air", factors["air"])),
        "room": room_value,
        "rain": float(profile_mix.get("rain", factors["rain"])),
        "tape": float(profile_mix.get("tape", factors["tape"])),
    }
    for name, mix_value in mix_alias.items():
        factors[name] = float(np.clip((factors[name] + mix_value) * 0.5, 0.0, 1.0))

    variant_delta = {
        "diffuse-air": {"air": 0.1, "room": 0.04},
        "room-hush": {"room": 0.12, "tape": 0.05},
        "velvet-room": {"room": 0.16, "air": 0.02},
        "sub-room": {"room": 0.18, "air": -0.08, "tape": 0.02},
        "black-room": {"room": 0.12, "air": -0.12, "tape": -0.02},
        "mist-rain": {"air": 0.06, "rain": 0.12},
        "tape-halo": {"tape": 0.12, "air": 0.04},
    }.get(variant, {})
    for name, delta in variant_delta.items():
        factors[name] = float(np.clip(factors[name] + delta, 0.0, 1.0))
    factors["air"] = float(np.clip(factors["air"] + palette["air"] * 0.08, 0.0, 1.0))
    factors["tape"] = float(np.clip(factors["tape"] + palette["grain"] * 0.06, 0.0, 1.0))
    return factors


def _schedule_motion_events(
    duration_seconds: int,
    density: float,
    motif_density: float,
    role_energy: float,
    ratios: np.ndarray,
    pulse_strength: float,
    artifact_sensitivity: float,
    sleep_safe: bool,
    reference_dark: bool,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    cursor = float(rng.uniform(0.0, 7.0))
    mean_gap = max(4.8, 16.0 - density * 9.0 - motif_density * 4.5 - role_energy * 2.5 - pulse_strength * 1.6)
    if sleep_safe:
        mean_gap += 5.5 + artifact_sensitivity * 4.5
    if reference_dark:
        mean_gap = max(9.0, mean_gap - 4.0)
    while cursor < duration_seconds + 10.0:
        if reference_dark:
            duration = float(rng.uniform(13.0, 28.0) * (1.08 - density * 0.12))
        elif sleep_safe:
            duration = float(rng.uniform(8.0, 18.0) * (1.06 - density * 0.16))
        else:
            duration = float(rng.uniform(4.0, 12.0) * (1.18 - density * 0.32 - pulse_strength * 0.08))
        ratio = float(rng.choice(ratios))
        if reference_dark:
            amplitude = float(
                rng.uniform(0.016, 0.038)
                * (0.78 + density * 0.46 + pulse_strength * 0.08)
                * (1.0 - 0.32 * artifact_sensitivity)
            )
            glide = float(rng.uniform(-0.0025, 0.0035))
            mod_rate = float(rng.uniform(0.008, 0.026))
        elif sleep_safe:
            amplitude = float(
                rng.uniform(0.009, 0.024)
                * (0.58 + density * 0.42 + pulse_strength * 0.08)
                * (1.0 - 0.38 * artifact_sensitivity)
            )
            glide = float(rng.uniform(-0.004, 0.006))
            mod_rate = float(rng.uniform(0.01, 0.045))
        else:
            amplitude = float(rng.uniform(0.018, 0.052) * (0.74 + density * 0.82 + pulse_strength * 0.16))
            glide = float(rng.uniform(-0.012, 0.015))
            mod_rate = float(rng.uniform(0.025, 0.11))
        events.append(
            {
                "center": cursor,
                "duration": duration,
                "ratio": ratio,
                "amplitude": amplitude,
                "glide": glide,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "mod_rate": mod_rate,
                "seed": float(rng.integers(0, 1_000_000)),
            }
        )
        cursor += mean_gap * float(rng.uniform(0.62, 1.42))
    return events


def _schedule_accent_events(
    duration_seconds: int,
    density: float,
    role: str,
    ratios: np.ndarray,
    accent_variant: str,
    palette: dict[str, float],
    sleep_safe: bool,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    base_gap = {
        "intro": 32.0,
        "settle": 28.0,
        "drift_a": 21.0,
        "drift_b": 18.0,
        "sparse_break": 34.0,
        "return": 24.0,
        "low_energy_tail": 38.0,
    }.get(role, 24.0)
    variant_gain = {"soft-grid": 0.95, "rain-lines": 0.7, "halo": 1.1, "orbital": 1.2}.get(accent_variant, 1.0)
    cursor = float(rng.uniform(6.0, 18.0))
    events: list[dict[str, float]] = []
    high_ratios = np.asarray([ratio for ratio in ratios if ratio >= 1.5], dtype=np.float32)
    pool = high_ratios if len(high_ratios) else ratios
    if sleep_safe:
        lower_pool = np.asarray([ratio for ratio in ratios if 0.5 <= ratio <= 1.5], dtype=np.float32)
        if len(lower_pool):
            pool = lower_pool
        base_gap *= 1.35
    while cursor < duration_seconds + 8.0:
        amplitude = float(rng.uniform(0.008, 0.022) * variant_gain * (0.78 + density * 0.38 + palette["mallet"] * 0.18))
        if sleep_safe:
            amplitude *= 0.42
        events.append(
            {
                "center": cursor,
                "duration": float(rng.uniform(0.9, 3.4) * (1.0 + 0.18 * palette["softness"])),
                "ratio": float(rng.choice(pool)),
                "amplitude": amplitude,
                "glide": float(rng.uniform(-0.012, 0.015) if sleep_safe else rng.uniform(-0.035, 0.042)),
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )
        cursor += base_gap * float(rng.uniform(0.68, 1.52))
    return events


def _schedule_rhythm_events(
    duration_seconds: int,
    density: float,
    role: str,
    pulse_density: str,
    pulse_rate: float,
    rhythm_variant: str,
    rhythm_intensity: float,
    ratios: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    if pulse_rate <= 0.0 or rhythm_intensity <= 0.01 or pulse_density == "none":
        return []
    interval_base = {
        "minimal": 8.2,
        "low": 5.6,
        "soft": 4.8,
        "medium": 4.0,
    }.get(pulse_density, 6.0)
    interval_base *= {"intro": 1.15, "settle": 1.0, "drift_a": 0.96, "drift_b": 0.9, "sparse_break": 1.45, "return": 0.95, "low_energy_tail": 1.35}.get(role, 1.0)
    interval_base *= 1.06 - rhythm_intensity * 0.22

    pair_chance = {"tidal": 0.22, "muted-step": 0.12, "glass-pulse": 0.18, "heartbeat": 0.34, "rainwalk": 0.28, "breathing-swell": 0.06}.get(rhythm_variant, 0.18)
    jitter = {"tidal": 0.24, "muted-step": 0.14, "glass-pulse": 0.16, "heartbeat": 0.12, "rainwalk": 0.3, "breathing-swell": 0.32}.get(rhythm_variant, 0.18)
    event_pool = {
        "tidal": [0.5, 1.0, 1.5],
        "muted-step": [0.5, 1.0, 2.0],
        "glass-pulse": [1.0, 1.5, 2.0],
        "heartbeat": [0.5, 1.0],
        "rainwalk": [0.5, 1.0, 1.5],
        "breathing-swell": [0.5, 0.707, 1.0],
    }.get(rhythm_variant, [0.5, 1.0, 1.5])

    available = [float(r) for r in ratios if 0.45 <= float(r) <= 2.2]
    if not available:
        available = event_pool

    cursor = float(rng.uniform(0.5, interval_base))
    events: list[dict[str, float]] = []
    while cursor < duration_seconds + interval_base:
        ratio_candidates = [ratio for ratio in available if any(abs(ratio - target) < 0.55 for target in event_pool)]
        ratio = float(rng.choice(ratio_candidates if ratio_candidates else available))
        duration = float(rng.uniform(0.28, 0.92) * (1.04 - rhythm_intensity * 0.16))
        amplitude = float(rng.uniform(0.012, 0.028) * (0.58 + density * 0.66 + rhythm_intensity * 0.4))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        events.append(
            {
                "center": cursor,
                "duration": duration,
                "ratio": ratio,
                "amplitude": amplitude,
                "phase": phase,
            }
        )
        if rng.random() < pair_chance and cursor + interval_base * 0.32 < duration_seconds:
            events.append(
                {
                    "center": cursor + interval_base * rng.uniform(0.26, 0.38),
                    "duration": duration * rng.uniform(0.7, 0.92),
                    "ratio": ratio * rng.choice([1.0, 1.5], p=[0.72, 0.28]),
                    "amplitude": amplitude * rng.uniform(0.58, 0.82),
                    "phase": phase + rng.uniform(-0.4, 0.4),
                }
            )
        cursor += interval_base * float(rng.uniform(1.0 - jitter, 1.0 + jitter))
    return events


def _macro_energy_curve(
    progress: np.ndarray,
    role: str,
    pacing_variant: str,
    movement_bias: float,
    phases: np.ndarray,
    breath_depth: float = 0.0,
    reference_dark: bool = False,
) -> np.ndarray:
    role_curve = {
        "intro": 0.38 + 0.52 * np.power(progress, 0.8),
        "settle": 0.62 + 0.14 * np.sin(np.pi * progress),
        "drift_a": 0.68 + 0.12 * np.sin(np.pi * (progress + 0.08)),
        "drift_b": 0.72 + 0.16 * np.sin(np.pi * (progress * 0.88 + 0.12)),
        "sparse_break": 0.44 + 0.08 * np.sin(2.0 * np.pi * progress),
        "return": 0.55 + 0.18 * np.sin(np.pi * progress),
        "low_energy_tail": 0.66 - 0.42 * progress,
    }.get(role, 0.6 + 0.1 * np.sin(np.pi * progress))
    macro_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * (progress * (0.65 + movement_bias * 0.55)) + phases[0])
    micro_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * (progress * (1.4 + movement_bias)) + phases[1])
    pulse_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * (progress * (2.3 + movement_bias * 0.8)) + phases[2])
    pacing = {
        "glacial": 0.92 + 0.10 * macro_wave,
        "slow-bloom": 0.86 + 0.14 * macro_wave + 0.05 * micro_wave,
        "steady-fall": 0.96 - 0.08 * progress + 0.05 * pulse_wave,
        "long-breath": 0.78 + 0.26 * macro_wave + 0.08 * micro_wave,
    }.get(pacing_variant, 0.9 + 0.1 * macro_wave)
    energy = role_curve * pacing * (0.94 + 0.08 * micro_wave) * (0.96 + 0.05 * pulse_wave)
    if reference_dark:
        breath_depth = float(np.clip(breath_depth, 0.0, 0.85))
        slow_breath = 1.0 - breath_depth * 0.38 + breath_depth * (0.62 * macro_wave + 0.38 * pulse_wave)
        energy *= slow_breath
    ceiling = 1.26 if reference_dark else 1.08
    floor = 0.07 if reference_dark else 0.16
    return np.clip(energy, floor, ceiling).astype(np.float32)


def _harmonic_drift(progress: np.ndarray, policy: str) -> np.ndarray:
    if policy == "anchored":
        return np.ones_like(progress, dtype=np.float32)
    if policy == "slow_modal_shift":
        return 1.0 + 0.015 * np.sin(2.0 * np.pi * progress) + 0.006 * np.sin(4.0 * np.pi * progress + 0.2)
    if policy == "descending":
        return 1.0 - 0.026 * progress + 0.004 * np.sin(2.0 * np.pi * progress)
    if policy == "rising_return":
        return 1.0 + 0.02 * progress + 0.004 * np.sin(3.0 * np.pi * progress)
    return np.ones_like(progress, dtype=np.float32)


def _layer_budget_multiplier(layer_budget: int, role: str) -> dict[str, float]:
    multipliers = {
        "motion": 1.0,
        "texture": 1.0,
        "accents": 1.0,
        "rhythm": 1.0,
    }
    if layer_budget <= 2:
        multipliers["motion"] *= 0.74
        multipliers["accents"] *= 0.45
        multipliers["rhythm"] *= 0.62
    elif layer_budget == 3:
        multipliers["accents"] *= 0.78
        multipliers["rhythm"] *= 0.84
    if role == "sparse_break":
        multipliers["motion"] *= 0.66
        multipliers["texture"] *= 0.72
        multipliers["accents"] *= 0.35
        multipliers["rhythm"] *= 0.24
    if role == "low_energy_tail":
        multipliers["motion"] *= 0.52
        multipliers["accents"] *= 0.2
        multipliers["rhythm"] *= 0.18
    if role == "intro":
        multipliers["rhythm"] *= 0.58
    return multipliers


def _section_edge_envelope(progress: np.ndarray, role: str) -> np.ndarray:
    attack_share = 0.12 if role == "intro" else 0.07
    release_share = 0.16 if role == "low_energy_tail" else 0.08
    attack = np.clip(progress / attack_share, 0.0, 1.0)
    release = np.clip((1.0 - progress) / release_share, 0.0, 1.0)
    fade = np.sqrt(np.minimum(attack, release))
    return (0.14 + 0.86 * fade).astype(np.float32)


def _spectral_color(spectrum: np.ndarray, freqs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    del freqs
    colored = np.fft.irfft(spectrum * weights, n=(len(spectrum) - 1) * 2).astype(np.float32)
    return _normalize(colored)


def _intervals_to_ratios(intervals: list[int]) -> list[float]:
    return [float(math.pow(2.0, interval / 12.0)) for interval in intervals]


def _note_to_frequency(note: str, octave: int) -> float:
    offset = NOTE_OFFSETS.get(note, -7)
    semitones = offset + (octave - 4) * 12
    return float(440.0 * math.pow(2.0, semitones / 12.0))


def _stereoize_tonal(
    signal: np.ndarray,
    width: float,
    t: np.ndarray,
    delay_samples: int,
    phase_rate: float,
    phase_offset: float,
    stereo_drive: float = 0.36,
) -> np.ndarray:
    if delay_samples <= 0 or signal.shape[0] <= delay_samples + 1:
        delay_samples = 1
    delayed_left = np.empty_like(signal)
    delayed_left[:delay_samples] = signal[:1]
    delayed_left[delay_samples:] = signal[:-delay_samples]
    delayed_right = np.empty_like(signal)
    delayed_right[-delay_samples:] = signal[-1:]
    delayed_right[:-delay_samples] = signal[delay_samples:]
    motion = np.sin(2.0 * np.pi * phase_rate * t + phase_offset).astype(np.float32)
    swirl = np.sin(2.0 * np.pi * (phase_rate * 0.63 + 0.008) * t + phase_offset * 1.43).astype(np.float32)
    mid = signal * (1.0 - width * 0.12)
    diff = delayed_left - delayed_right
    diff_smooth = _smooth_signal(diff)
    bloom = _smooth_signal(signal * motion)
    side = width * stereo_drive * (0.28 * diff + 0.22 * diff_smooth + 0.14 * signal * motion + 0.08 * bloom * swirl)
    left = mid + side
    right = mid - side
    return np.column_stack([left, right]).astype(np.float32)


def _soft_lattice(t: np.ndarray, rate: float, phase: float, sharpness: float) -> np.ndarray:
    if rate <= 0.0:
        return np.zeros_like(t, dtype=np.float32)
    wave = np.clip(0.5 + 0.5 * np.sin(2.0 * np.pi * rate * t + phase), 0.0, 1.0)
    return np.power(wave, sharpness).astype(np.float32)


def _pulse_envelope(local: np.ndarray, duration: float, softness: float) -> np.ndarray:
    attack = max(0.03, duration * (0.16 + 0.08 * softness))
    release = max(0.08, duration * (0.62 + 0.18 * softness))
    attack_env = np.clip((local + attack) / attack, 0.0, 1.0)
    release_env = np.exp(-np.clip(local, 0.0, None) / release)
    env = attack_env * release_env
    env *= (local >= -attack).astype(np.float32)
    return env.astype(np.float32)


def _smooth_signal(signal: np.ndarray) -> np.ndarray:
    kernel = np.array([0.06, 0.12, 0.18, 0.28, 0.18, 0.12, 0.06], dtype=np.float32)
    return np.convolve(signal, kernel, mode="same").astype(np.float32)


def _soft_clip(signal: np.ndarray, drive: float) -> np.ndarray:
    norm = max(1e-6, math.tanh(drive))
    return (np.tanh(signal * drive) / norm).astype(np.float32)


def _soften_tone(signal: np.ndarray, sample_rate: int, amount: float, cutoff_hz: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.65))
    if amount <= 0.0 or len(signal) <= 8:
        return signal.astype(np.float32)
    low = _one_pole_lowpass(signal, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    high = signal - low
    return (low + high * (1.0 - amount)).astype(np.float32)


def _sanitize_stem_audio(audio: np.ndarray, channels: int) -> np.ndarray:
    block = np.asarray(audio, dtype=np.float32)
    if block.ndim == 1:
        block = np.repeat(block[:, None], channels, axis=1)
    elif block.shape[1] != channels:
        if block.shape[1] == 1:
            block = np.repeat(block, channels, axis=1)
        else:
            block = block[:, :channels]
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    block = np.clip(block, -1.25, 1.25)
    return np.ascontiguousarray(block, dtype=np.float32)


def _estimate_sample_rate_from_t(t: np.ndarray) -> int:
    if len(t) < 2:
        return 48000
    step = float(t[1] - t[0])
    if step <= 0.0:
        return 48000
    return int(round(1.0 / step))


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


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < 1e-6:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)
