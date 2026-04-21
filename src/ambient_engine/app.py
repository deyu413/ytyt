from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ambient_engine.arrangement.assembler import LongformAssembler
from ambient_engine.core.durations import parse_duration
from ambient_engine.core.logging import configure_logging
from ambient_engine.core.manifest import SessionManifest
from ambient_engine.core.paths import ProjectPaths
from ambient_engine.core.runtime import detect_runtime
from ambient_engine.core.seeds import seed_everything
from ambient_engine.generation.contracts import SectionRenderRequest
from ambient_engine.generation.registry import ProviderRegistry
from ambient_engine.planning.macrostructure import MacrostructurePlanner
from ambient_engine.planning.variation import VariationPlanner
from ambient_engine.profiles.loader import load_profile_by_id
from ambient_engine.qc.analyzers import analyze_audio
from ambient_engine.qc.gating import gate_render
from ambient_engine.qc.regeneration import RegenerationPlanner
from ambient_engine.qc.scoring import score_metrics
from ambient_engine.publish.youtube import YouTubePublisher
from ambient_engine.render.exports import export_mp3, export_preview
from ambient_engine.render.hud_video import render_hud_video
from ambient_engine.render.metadata import MetadataBuilder
from ambient_engine.render.reports import write_qc_report, write_session_report
from ambient_engine.render.shorts import render_session_shorts
from ambient_engine.render.static_frame import render_static_frame
from ambient_engine.render.thumbnails import render_thumbnail


class AmbientEngine:
    def __init__(self, project_root: Path) -> None:
        self.project_paths = ProjectPaths(project_root)
        defaults_path = self.project_paths.config_dir / "defaults.yaml"
        self.defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))

    def render(
        self,
        profile_id: str,
        target_length: str | None,
        runtime_mode: str,
        seed: int,
        dry_run: bool = False,
        session_id: str | None = None,
        background_image_path: Path | None = None,
        with_shorts: bool = False,
    ) -> dict[str, Any]:
        profile = load_profile_by_id(self.project_paths.profiles_dir, profile_id)
        target_seconds = parse_duration(target_length).seconds if target_length else profile.default_target_length_seconds
        session_id = session_id or self._build_session_id(profile_id, seed)
        session_paths = self.project_paths.create_session(session_id)
        logger = configure_logging(session_paths.logs / "session.log")
        seed_everything(seed)

        runtime = detect_runtime(runtime_mode, self.project_paths)
        variation = VariationPlanner(profile, seed).build()
        if background_image_path is not None:
            variation["subject_alignment"] = "left"
            variation["background_image"] = str(background_image_path.resolve())
        planner = MacrostructurePlanner(profile, seed)
        crossfade_seconds = float(self.defaults["audio"]["crossfade_seconds"])
        overlap_compensation_seconds = int(round(crossfade_seconds * max(0, len(profile.section_schema) - 1)))
        section_plan = planner.build(target_seconds + overlap_compensation_seconds)
        registry = ProviderRegistry(runtime)

        manifest = SessionManifest(
            session_id=session_id,
            profile_id=profile.profile_id,
            runtime_mode=runtime_mode,
            seed=seed,
            provider_capabilities={
                name: {"available": cap.available, "reason": cap.reason}
                for name, cap in runtime.provider_capabilities.items()
            },
            variation=variation,
        )

        manifest.section_plan = []
        routing_rows = []
        for section in section_plan:
            selection = registry.select(task=f"section:{section.role}", preferred_chain=section.generator_chain)
            manifest.section_plan.append(
                {
                    "index": section.index,
                    "role": section.role,
                    "duration_seconds": section.duration_seconds,
                    "density": section.density,
                    "provider": selection.provider_name,
                    "transition_policy": section.transition_policy,
                }
            )
            routing_rows.append(
                {
                    "task": f"section:{section.role}",
                    "provider": selection.provider_name,
                    "fallback_chain": selection.fallback_chain,
                    "reason": selection.reason,
                }
            )
        manifest.model_routing = routing_rows
        manifest.save(session_paths.manifests / "session_manifest.json")

        metadata = MetadataBuilder(profile, seed).build(target_seconds, variation)
        metadata_path = session_paths.manifests / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        if dry_run:
            outputs = {
                "manifest": str(session_paths.manifests / "session_manifest.json"),
                "metadata": str(metadata_path),
            }
            write_session_report(manifest, metadata, outputs, session_paths.reports / "session_report.md")
            return {
                "session_id": session_id,
                "session_root": str(session_paths.root),
                "dry_run": True,
                "outputs": outputs,
            }

        logger.info("Rendering %s sections for profile %s", len(section_plan), profile.profile_id)
        section_results = []
        stem_names = list(profile.instrumentation.get("stems", ["drone", "motion", "texture", "accents"]))
        for section in section_plan:
            selection = registry.select(task=f"section:{section.role}", preferred_chain=section.generator_chain)
            provider = registry.get(selection.provider_name)
            logger.info(
                "Rendering section %02d %s (%ss) via %s",
                section.index,
                section.role,
                section.duration_seconds,
                selection.provider_name,
            )
            request = SectionRenderRequest(
                session_paths=session_paths,
                profile=profile,
                section=section,
                sample_rate=int(self.defaults["audio"]["sample_rate"]),
                block_seconds=int(self.defaults["audio"]["block_seconds"]),
                channels=int(self.defaults["audio"]["channels"]),
                stem_names=stem_names,
                session_seed=seed,
                variation=variation,
            )
            result = provider.render_section(request)
            section_results.append(result)

        assembler = LongformAssembler(
            sample_rate=int(self.defaults["audio"]["sample_rate"]),
            channels=int(self.defaults["audio"]["channels"]),
            crossfade_seconds=crossfade_seconds,
            block_frames=int(self.defaults["audio"]["mix_block_frames"]),
        )
        master_wav = session_paths.exports / "master.wav"
        assembly = assembler.assemble(
            profile=profile,
            section_results=section_results,
            session_stems_dir=session_paths.stems,
            master_output_path=master_wav,
            target_lufs=profile.loudness_target_lufs,
        )

        qc_metrics, score_card, gate = self._run_qc(master_wav, profile.loudness_target_lufs, assembly.boundary_frames)
        regeneration_plan = RegenerationPlanner().plan(qc_metrics, section_results)
        if not gate["accepted"] and regeneration_plan["sections_to_regenerate"]:
            logger.info("QC failed; regenerating sections %s", regeneration_plan["sections_to_regenerate"])
            for section_index in regeneration_plan["sections_to_regenerate"]:
                section = section_plan[section_index]
                section.variation_seed += 1009
                selection = registry.select(task=f"section:{section.role}", preferred_chain=section.generator_chain)
                provider = registry.get(selection.provider_name)
                request = SectionRenderRequest(
                    session_paths=session_paths,
                    profile=profile,
                    section=section,
                    sample_rate=int(self.defaults["audio"]["sample_rate"]),
                    block_seconds=int(self.defaults["audio"]["block_seconds"]),
                    channels=int(self.defaults["audio"]["channels"]),
                    stem_names=stem_names,
                    session_seed=seed,
                    variation=variation,
                )
                section_results[section_index] = provider.render_section(request)
            assembly = assembler.assemble(
                profile=profile,
                section_results=section_results,
                session_stems_dir=session_paths.stems,
                master_output_path=master_wav,
                target_lufs=profile.loudness_target_lufs,
            )
            qc_metrics, score_card, gate = self._run_qc(master_wav, profile.loudness_target_lufs, assembly.boundary_frames)

        preview_wav = export_preview(
            master_path=master_wav,
            output_path=session_paths.exports / "preview_60s.wav",
            seconds=int(self.defaults["render"]["preview_seconds"]),
            sample_rate=int(self.defaults["audio"]["sample_rate"]),
        )
        master_mp3 = export_mp3(runtime.ffmpeg_executable, master_wav, session_paths.exports / "master.mp3")
        preview_mp3 = export_mp3(runtime.ffmpeg_executable, preview_wav, session_paths.exports / "preview_60s.mp3")
        static_frame = render_static_frame(
            profile,
            variation,
            session_paths.exports / "static_frame.png",
            background_image_path=background_image_path,
        )
        thumbnail = render_thumbnail(
            profile=profile,
            metadata=metadata,
            target_seconds=target_seconds,
            static_frame_path=static_frame,
            output_path=session_paths.exports / "thumbnail.png",
            variation=variation,
        )
        hud_video = render_hud_video(
            ffmpeg_executable=runtime.ffmpeg_executable,
            static_frame_path=static_frame,
            audio_path=master_wav,
            output_path=session_paths.exports / "hud_video.mp4",
            hud_label=str(metadata["hud_label"]),
        )

        qc_report_path = write_qc_report(qc_metrics, score_card, gate, session_paths.manifests / "qc_report.json")
        manifest.asset_lineage = {
            "master_wav": str(master_wav),
            "master_mp3": str(master_mp3),
            "preview_wav": str(preview_wav),
            "preview_mp3": str(preview_mp3),
            "thumbnail": str(thumbnail),
            "static_frame": str(static_frame),
            "hud_video": str(hud_video),
            "background_image": str(background_image_path.resolve()) if background_image_path else None,
            "section_count": len(section_results),
            "boundary_frames": list(assembly.boundary_frames),
            "stems": {name: str(path) for name, path in assembly.stem_exports.items()},
        }
        manifest.qc = {
            "accepted": gate["accepted"],
            "failures": gate["failures"],
            "global_score": score_card["global_score"],
            "true_peak": qc_metrics["true_peak"],
            "integrated_loudness": qc_metrics["integrated_loudness"],
            "fatigue_risk": qc_metrics["fatigue_risk"],
            "reference_dna_score": qc_metrics.get("reference_dna_score"),
            "bass_anchor_score": qc_metrics.get("bass_anchor_score"),
            "dynamic_breath_score": qc_metrics.get("dynamic_breath_score"),
            "lowmid_boxiness_risk": qc_metrics.get("lowmid_boxiness_risk"),
        }
        manifest.save(session_paths.manifests / "session_manifest.json")
        outputs = {
            "manifest": str(session_paths.manifests / "session_manifest.json"),
            "metadata": str(metadata_path),
            "qc_report": str(qc_report_path),
            "master_wav": str(master_wav),
            "master_mp3": str(master_mp3),
            "preview_wav": str(preview_wav),
            "preview_mp3": str(preview_mp3),
            "thumbnail": str(thumbnail),
            "static_frame": str(static_frame),
            "hud_video": str(hud_video),
        }
        if with_shorts:
            shorts_result = render_session_shorts(
                profile=profile,
                metadata=metadata,
                manifest=manifest,
                session_dir=session_paths.root,
                ffmpeg_executable=runtime.ffmpeg_executable,
            )
            manifest.asset_lineage["shorts"] = shorts_result["shorts"]
            outputs["shorts_manifest"] = shorts_result["shorts_manifest"]
            for short in shorts_result["shorts"]:
                outputs[f"{short['slug']}_video"] = short["video"]
        write_session_report(manifest, metadata, outputs, session_paths.reports / "session_report.md")
        manifest.save(session_paths.manifests / "session_manifest.json")
        return {
            "session_id": session_id,
            "session_root": str(session_paths.root),
            "outputs": outputs,
            "qc": manifest.qc,
        }

    def run_qc(self, session_dir: Path) -> dict[str, Any]:
        session_dir = session_dir.resolve()
        manifest = SessionManifest.load(session_dir / "manifests" / "session_manifest.json")
        metrics, score_card, gate = self._run_qc(
            master_path=session_dir / "exports" / "master.wav",
            target_lufs=self._profile_loudness(manifest.profile_id),
            boundary_frames=list(manifest.asset_lineage.get("boundary_frames", [])),
        )
        report_path = write_qc_report(metrics, score_card, gate, session_dir / "manifests" / "qc_report.json")
        return {"metrics": metrics, "score_card": score_card, "gate": gate, "qc_report": str(report_path)}

    def package(self, session_dir: Path) -> dict[str, Any]:
        session_dir = session_dir.resolve()
        manifest = SessionManifest.load(session_dir / "manifests" / "session_manifest.json")
        metadata = json.loads((session_dir / "manifests" / "metadata.json").read_text(encoding="utf-8"))
        outputs = {
            str(path.relative_to(session_dir / "exports")): str(path)
            for path in (session_dir / "exports").rglob("*")
            if path.is_file()
        }
        shorts_manifest = session_dir / "manifests" / "shorts_manifest.json"
        if shorts_manifest.exists():
            outputs["../manifests/shorts_manifest.json"] = str(shorts_manifest)
        report = write_session_report(manifest, metadata, outputs, session_dir / "reports" / "session_report.md")
        return {"session_id": manifest.session_id, "outputs": outputs, "session_report": str(report)}

    def publish(self, session_dir: Path, dry_run: bool) -> dict[str, Any]:
        session_dir = session_dir.resolve()
        metadata = json.loads((session_dir / "manifests" / "metadata.json").read_text(encoding="utf-8"))
        publisher = YouTubePublisher(
            client_secrets=self.project_paths.config_dir / "client_secret.json",
            token_file=self.project_paths.config_dir / "youtube_token.json",
        )
        video_path = session_dir / "exports" / "hud_video.mp4"
        thumbnail_path = session_dir / "exports" / "thumbnail.png"
        if dry_run:
            return publisher.dry_run(video_path, metadata, thumbnail_path)
        return publisher.upload(video_path, metadata, thumbnail_path)

    def generate_shorts(self, session_dir: Path) -> dict[str, Any]:
        session_dir = session_dir.resolve()
        manifest = SessionManifest.load(session_dir / "manifests" / "session_manifest.json")
        metadata = json.loads((session_dir / "manifests" / "metadata.json").read_text(encoding="utf-8"))
        profile = load_profile_by_id(self.project_paths.profiles_dir, manifest.profile_id)
        runtime = detect_runtime(manifest.runtime_mode, self.project_paths)
        result = render_session_shorts(
            profile=profile,
            metadata=metadata,
            manifest=manifest,
            session_dir=session_dir,
            ffmpeg_executable=runtime.ffmpeg_executable,
        )
        manifest.asset_lineage["shorts"] = result["shorts"]
        manifest.save(session_dir / "manifests" / "session_manifest.json")
        outputs = {"shorts_manifest": result["shorts_manifest"]}
        for short in result["shorts"]:
            outputs[f"{short['slug']}_video"] = short["video"]
        write_session_report(manifest, metadata, outputs, session_dir / "reports" / "session_report.md")
        return {"session_id": manifest.session_id, **result}

    def _run_qc(self, master_path: Path, target_lufs: float, boundary_frames: list[int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        metrics = analyze_audio(
            master_path=master_path,
            sample_rate=int(self.defaults["audio"]["sample_rate"]),
            boundary_frames=boundary_frames,
            block_seconds=int(self.defaults["qc"]["analysis_block_seconds"]),
        )
        score_card = score_metrics(
            metrics=metrics,
            target_lufs=target_lufs,
            true_peak_ceiling=float(self.defaults["audio"]["true_peak_ceiling"]),
        )
        gate = gate_render(
            metrics=metrics,
            score_card=score_card,
            minimum_score=float(self.defaults["qc"]["minimum_global_score"]),
            true_peak_ceiling=float(self.defaults["audio"]["true_peak_ceiling"]),
        )
        return metrics, score_card, gate

    def _build_session_id(self, profile_id: str, seed: int) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{profile_id}_{seed}"

    def _profile_loudness(self, profile_id: str) -> float:
        profile = load_profile_by_id(self.project_paths.profiles_dir, profile_id)
        return profile.loudness_target_lufs
