# Audit

## What Exists

- A monolithic pipeline centered on a trading niche and YouTube upload flow.
- Audio generation built around short segments, then extension to long duration by concatenation and repeated crossfading.
- Separate modules for audio generation, textures, post-processing, metadata, thumbnails, loop-video assembly, and upload.
- Runtime state mixed directly into the repo via `assets/`, `output/`, and `logs/`.

## What Is Broken

- Long-form composition quality is structurally weak. The old system builds duration by repetition and sliding-window style concatenation instead of macrostructure.
- The overlap logic in the previous extender duplicated material at seams instead of replacing the overlap region cleanly, increasing audible repetition.
- The repo hardcoded Windows-specific paths and local assumptions, including model locations and video files.
- Trading-specific metadata and editorial logic were welded into the audio engine core.
- Optional models had no robust capability detection or fallback policy, so the pipeline failed hard on dependency issues instead of degrading gracefully.
- Runtime prerequisites were under-specified. The old code assumed `python`, `ffmpeg`, and optional model dependencies would already exist and work on PATH.
- There was no objective QC gate before export or publish.
- Tests and smoke checks were missing.

## What Is Salvageable

- Some of the old uploader OAuth logic was worth reusing as a decoupled publish adapter.
- Pillow-based image generation patterns were reusable for thumbnails and static frames.
- Basic FFmpeg orchestration ideas were reusable, but only after removing assumptions about system-wide binaries.

## What Must Be Replaced

- The old audio core based on naive extension and late-stage texturing.
- Trading-first metadata, template, and prompt systems.
- Veo3 prompt generation as a required part of the product.
- Top-level ad hoc entrypoints and single-flow orchestrator design.
- Hardcoded paths and environment assumptions.
- Lack of manifests, seeds, lineage, and QC enforcement.

