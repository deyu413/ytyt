# Ambient Engine

Free/open ambient music production engine for long-form YouTube output.

This repository no longer targets trading loops, Veo3 prompts, or naive repetition-based extension. It now focuses on a portable, profile-driven ambient pipeline that can generate long-form audio with section planning, stem assembly, QC gating, reproducible manifests, and a simple final video format: static frame + minimal HUD.

## What It Ships

- Free/open core path with no paid API dependency
- CPU-safe demo path that runs without ACE-Step, YuE, or Stable Audio Open
- Optional adapters for `ACE-Step 1.5`, `YuE`, and `Stable Audio Open`
- Profile-driven long-form planning
- Stem-based assembly with overlap-replace transitions
- QC scoring and gating before publish
- Output packaging for audio, preview, visuals, reports, and lineage

## Commercial-Safe Model Policy

- Default commercial-safe core: `procedural_dsp`
- Intended preferred music engine when available: `ACE-Step 1.5`
- Optional structured experiment path: `YuE`
- Optional texture enhancer: `Stable Audio Open`
- Not used in the monetized production path: `MusicGen`

See [docs/model_routing.md](/C:/Users/nerod/OneDrive/Desktop/allprojects/musciayt/docs/model_routing.md) for the decision table.

## Repository Layout

```text
config/
  defaults.yaml
  model_routing.yaml
docs/
  audit.md
  model_routing.md
profiles/
  afterblue_sleep.yaml
  afterblue_reference_sleep.yaml
  rainy_city_calm.yaml
  blue_hour_rest.yaml
  quiet_night_focus.yaml
  rainy_city_sleep.yaml
  quiet_midnight_solitude.yaml
src/ambient_engine/
  core/
  profiles/
  planning/
  generation/
  arrangement/
  qc/
  render/
  publish/
  app.py
  cli.py
tests/
sessions/   # generated at runtime, gitignored
```

## Install

```bash
pip install -e .
```

If you prefer a plain requirements install:

```bash
pip install -r requirements.txt
```

The project uses `imageio-ffmpeg` to avoid relying on a system-wide FFmpeg installation. Optional YouTube publishing still requires Google API credentials under `config/`.

## Quickstart

Dry-run the full planning flow:

```bash
ambient render --profile blue_hour_rest --target-length 60m --runtime cpu-safe --seed 42 --dry-run
```

Run the canonical demo:

```bash
ambient demo --profile afterblue_sleep --target-length 2h --runtime cpu-safe --seed 42
```

Render the reference-guided Afterblue profile:

```bash
ambient render --profile afterblue_reference_sleep --target-length 10m --runtime cpu-safe --seed 77
```

Run QC again on an existing session:

```bash
ambient qc --session sessions/<session_id>
```

Regenerate package/report artifacts:

```bash
ambient package --session sessions/<session_id>
```

Dry-run YouTube publishing:

```bash
ambient publish youtube --session sessions/<session_id> --dry-run
```

## Session Outputs

Each render writes to `sessions/<session_id>/`:

- `exports/master.wav`
- `exports/master.mp3`
- `exports/preview_60s.wav`
- `exports/preview_60s.mp3`
- `exports/thumbnail.png`
- `exports/static_frame.png`
- `exports/hud_video.mp4`
- `stems/*.wav`
- `manifests/session_manifest.json`
- `manifests/metadata.json`
- `manifests/qc_report.json`
- `reports/session_report.md`

## Profiles

Primary shipped profiles:

- `afterblue_sleep`
- `afterblue_reference_sleep`
- `rainy_city_calm`
- `blue_hour_rest`
- `quiet_night_focus`

Editorial example profiles:

- `rainy_city_sleep`
- `quiet_midnight_solitude`

Each profile defines mood, pulse density, tonal center, scale family, instrumentation, texture mix, section schema, target length, loudness target, thumbnail style, title families, forbidden artifacts, and branding metadata.

## Runtime Modes

- `cpu-safe`: guaranteed free/open path using the bundled procedural DSP provider
- `gpu`: same engine plus optional provider routing to ACE-Step, YuE, and Stable Audio Open when installed

The engine never assumes local model paths, `python`, `ffmpeg`, or `ffprobe` on PATH. Missing optional providers degrade cleanly to the procedural provider and are recorded in the session manifest.

## QC Gate

Every real render is scored before the session is accepted:

- clipping / true peak safety
- integrated loudness estimate
- silence ratio
- repetition score
- section boundary smoothness
- spectral monotony
- harshness
- stereo / mono collapse risk
- dynamic flatness
- bass-anchor score
- low-mid boxiness risk
- dark-balance score
- reference-DNA score for reference-guided profiles
- artifact spike ratio
- long-form fatigue risk

If the score fails, the engine can regenerate selected high-exposure sections instead of rerunning the full session.

## Troubleshooting

- If `ambient` is not found after install, use `python -m ambient_engine ...`.
- If `hud_video.mp4` or MP3 export fails, verify `imageio-ffmpeg` is installed.
- If optional providers do not engage in `gpu` mode, set:
  - `AMBIENT_ACESTEP_PATH`
  - `AMBIENT_YUE_PATH`
  - `AMBIENT_STABLE_AUDIO_PATH`
- If YouTube upload fails, add OAuth desktop credentials to:
  - `config/client_secret.json`

## Known Limits

- The CPU-safe provider is a production-grade fallback engine, not a substitute for a tuned ACE-Step stack.
- Integrated loudness is estimated heuristically in the current implementation rather than a full BS.1770 offline pass.
- The current optional provider adapters detect availability and reserve the architecture, but they do not yet ship full inference wiring for every third-party model.

## Documentation

- [Audit](/C:/Users/nerod/OneDrive/Desktop/allprojects/musciayt/docs/audit.md)
- [Model Routing](/C:/Users/nerod/OneDrive/Desktop/allprojects/musciayt/docs/model_routing.md)
- [Afterblue Reference DNA](/C:/Users/nerod/OneDrive/Desktop/allprojects/musciayt/docs/afterblue_reference_dna.md)
- [Migration Notes](/C:/Users/nerod/OneDrive/Desktop/allprojects/musciayt/MIGRATION_NOTES.md)
- [TODO](/C:/Users/nerod/OneDrive/Desktop/allprojects/musciayt/TODO.md)

## Roadmap

- Full ACE-Step 1.5 inference adapter
- Optional Stable Audio Open texture enhancement pass
- Stronger loudness measurement
- Additional profile families and editorial variants
- Better spectral fatigue heuristics over very long sessions
