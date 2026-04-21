from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ambient_engine.arrangement.layering import resolve_stem_levels
from ambient_engine.arrangement.spectral_balance import db_to_linear, stabilize_master_block
from ambient_engine.arrangement.transitions import assemble_stem_sequence
from ambient_engine.generation.contracts import SectionRenderResult
from ambient_engine.profiles.schema import Profile


@dataclass
class AssemblyResult:
    stem_exports: dict[str, Path]
    master_wav: Path
    boundary_frames: list[int]
    integrated_loudness_estimate: float
    true_peak: float


class LongformAssembler:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        crossfade_seconds: float,
        block_frames: int = 65536,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.crossfade_seconds = crossfade_seconds
        self.block_frames = block_frames

    def assemble(
        self,
        profile: Profile,
        section_results: list[SectionRenderResult],
        session_stems_dir: Path,
        master_output_path: Path,
        target_lufs: float,
    ) -> AssemblyResult:
        stem_names = sorted(section_results[0].stem_files)
        stem_exports: dict[str, Path] = {}
        boundary_frames: list[int] = []
        for stem_name in stem_names:
            stem_output = session_stems_dir / f"{stem_name}.wav"
            input_paths = [result.stem_files[stem_name] for result in section_results]
            transition_policies = [result.transition_policy for result in section_results]
            stem_boundaries = assemble_stem_sequence(
                input_paths=input_paths,
                output_path=stem_output,
                sample_rate=self.sample_rate,
                channels=self.channels,
                crossfade_seconds=self.crossfade_seconds,
                transition_policies=transition_policies,
                stem_name=stem_name,
                block_frames=self.block_frames,
            )
            stem_exports[stem_name] = stem_output
            if not boundary_frames:
                boundary_frames = stem_boundaries

        stem_levels_db = resolve_stem_levels(profile)
        linear_levels = {name: db_to_linear(level) for name, level in stem_levels_db.items()}
        stereo_width = float(profile.instrumentation.get("stereo_width", 0.35))
        target_width = float(profile.instrumentation.get("target_stereo_side_mid", np.clip(0.06 + stereo_width * 0.12, 0.065, 0.14)))
        max_width = float(profile.instrumentation.get("max_stereo_side_mid", np.clip(target_width * 2.1, 0.18, 0.28)))
        target_hf_ratio = self._target_hf_ratio(profile)
        target_presence_ratio = self._target_presence_ratio(profile)
        stats = self._estimate_mix(stem_exports, linear_levels, target_width, max_width, target_hf_ratio, target_presence_ratio)
        gain_db = target_lufs - stats["integrated_loudness_estimate"]
        if stats["true_peak"] > 0:
            max_gain_db = 20.0 * np.log10(0.8912509381337456 / max(stats["true_peak"], 1e-9))
            gain_db = min(gain_db, max_gain_db)
        final_stats = self._write_mix(
            stem_exports,
            linear_levels,
            master_output_path,
            gain_db,
            target_width,
            max_width,
            target_hf_ratio,
            target_presence_ratio,
        )
        return AssemblyResult(
            stem_exports=stem_exports,
            master_wav=master_output_path,
            boundary_frames=boundary_frames,
            integrated_loudness_estimate=final_stats["integrated_loudness_estimate"],
            true_peak=final_stats["true_peak"],
        )

    def _estimate_mix(
        self,
        stem_exports: dict[str, Path],
        linear_levels: dict[str, float],
        target_width: float,
        max_width: float,
        target_hf_ratio: float,
        target_presence_ratio: float,
    ) -> dict[str, float]:
        readers = {name: sf.SoundFile(path, mode="r") for name, path in stem_exports.items()}
        try:
            sumsq = 0.0
            peak = 0.0
            frames = 0
            while True:
                block = self._read_mix_block(readers, linear_levels)
                if block is None:
                    break
                block = stabilize_master_block(
                    block,
                    target_width=target_width,
                    max_width=max_width,
                    sample_rate=self.sample_rate,
                    target_hf_ratio=target_hf_ratio,
                    target_presence_ratio=target_presence_ratio,
                )
                sumsq += float(np.sum(block ** 2))
                peak = max(peak, float(np.max(np.abs(block))))
                frames += len(block)
            rms = np.sqrt(sumsq / max(1, frames * self.channels))
            loudness = 20.0 * np.log10(max(rms, 1e-9))
            return {"integrated_loudness_estimate": loudness, "true_peak": peak}
        finally:
            for reader in readers.values():
                reader.close()

    def _write_mix(
        self,
        stem_exports: dict[str, Path],
        linear_levels: dict[str, float],
        output_path: Path,
        gain_db: float,
        target_width: float,
        max_width: float,
        target_hf_ratio: float,
        target_presence_ratio: float,
    ) -> dict[str, float]:
        linear_gain = db_to_linear(gain_db)
        readers = {name: sf.SoundFile(path, mode="r") for name, path in stem_exports.items()}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sumsq = 0.0
        peak = 0.0
        frames = 0
        try:
            with sf.SoundFile(
                output_path,
                mode="w",
                samplerate=self.sample_rate,
                channels=self.channels,
                subtype="PCM_24",
            ) as writer:
                while True:
                    block = self._read_mix_block(readers, linear_levels)
                    if block is None:
                        break
                    mixed = stabilize_master_block(
                        block * linear_gain,
                        target_width=target_width,
                        max_width=max_width,
                        sample_rate=self.sample_rate,
                        target_hf_ratio=target_hf_ratio,
                        target_presence_ratio=target_presence_ratio,
                    )
                    writer.write(mixed)
                    sumsq += float(np.sum(mixed ** 2))
                    peak = max(peak, float(np.max(np.abs(mixed))))
                    frames += len(mixed)
            rms = np.sqrt(sumsq / max(1, frames * self.channels))
            loudness = 20.0 * np.log10(max(rms, 1e-9))
            return {"integrated_loudness_estimate": loudness, "true_peak": peak}
        finally:
            for reader in readers.values():
                reader.close()

    def _read_mix_block(
        self,
        readers: dict[str, sf.SoundFile],
        linear_levels: dict[str, float],
    ) -> np.ndarray | None:
        any_audio = False
        mixed = np.zeros((self.block_frames, self.channels), dtype=np.float32)
        max_len = 0
        for name, reader in readers.items():
            block = reader.read(self.block_frames, dtype="float32", always_2d=True)
            if len(block):
                any_audio = True
                max_len = max(max_len, len(block))
                mixed[: len(block)] += block * linear_levels.get(name, 1.0)
        if not any_audio:
            return None
        return mixed[:max_len]

    def _target_hf_ratio(self, profile: Profile) -> float:
        mood_text = " ".join(
            [
                profile.mood,
                " ".join(profile.forbidden_artifacts),
                " ".join(str(item) for item in profile.instrumentation.get("primary", [])),
                " ".join(str(item) for item in profile.instrumentation.get("secondary", [])),
            ]
        ).lower()
        target = 0.09 if profile.loudness_target_lufs > -18.0 else 0.082
        if any(token in mood_text for token in ["sleep", "rest", "solitude", "insomnia"]):
            target -= 0.01
        if "focus" in mood_text:
            target += 0.004
        if "rain" in mood_text or float(profile.texture_mix.get("rain", 0.0)) > 0.4:
            target += 0.006
        if any(token in mood_text for token in ["harsh", "brittle", "bright"]):
            target -= 0.006
        reference_dark = str(profile.instrumentation.get("production_dna", "")).lower() == "reference_dark_bass"
        if reference_dark:
            target = min(target, 0.035)
        return float(np.clip(target, 0.012 if reference_dark else 0.068, 0.1))

    def _target_presence_ratio(self, profile: Profile) -> float:
        mood_text = " ".join(
            [
                profile.mood,
                " ".join(profile.forbidden_artifacts),
                " ".join(str(item) for item in profile.instrumentation.get("primary", [])),
                " ".join(str(item) for item in profile.instrumentation.get("secondary", [])),
            ]
        ).lower()
        target = 0.14 if profile.loudness_target_lufs > -18.0 else 0.125
        if any(token in mood_text for token in ["sleep", "rest", "solitude", "insomnia"]):
            target -= 0.04
        if "focus" in mood_text:
            target += 0.015
        if any(token in mood_text for token in ["harsh", "brittle", "bright"]):
            target -= 0.01
        if str(profile.instrumentation.get("production_dna", "")).lower() == "reference_dark_bass":
            target = min(target, 0.018)
        return float(np.clip(target, 0.01, 0.18))
