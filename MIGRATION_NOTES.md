# Migration Notes

## Product Direction Shift

Before:

- Trading-centric ambient pipeline
- Video loop generation and Veo3 prompts inside the main flow
- Metadata and monetization logic coupled to audio generation

Now:

- Ambient production engine for sleep, calm, rest, focus, solitude, and late-night moods
- Final visual target is static frame + minimal HUD
- Branding, metadata, and publish steps are decoupled from the audio engine

## Removed

- Legacy top-level scripts
- Trading prompt library
- Trading description templates
- Old audio extension pipeline
- Old video prompt generation

## Added

- `src/ambient_engine/`
- `profiles/`
- `config/defaults.yaml`
- `config/model_routing.yaml`
- `sessions/` output model
- QC gate with regeneration support
- Reproducible manifests and session reports

## Operational Difference

- `ambient render` is now the main entrypoint.
- `ambient demo --profile afterblue_sleep --target-length 2h --runtime cpu-safe` is the canonical end-to-end example.
- Optional providers are detected and recorded; they are never assumed.

