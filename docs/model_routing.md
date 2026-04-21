# Model Routing

The engine is built around a commercial-safe free/open core.

## Decisions

- `ACE-Step 1.5` is the intended preferred musical generator when a local GPU stack is available.
- `procedural_dsp` is the guaranteed runnable fallback and the required CPU-safe demo path.
- `YuE` is optional for more structured experiments, not the default ambient engine.
- `Stable Audio Open` is optional for short texture enhancement only.
- `MusicGen` is excluded from the monetized production path because the official AudioCraft repository publishes model weights under `CC-BY-NC 4.0`.

## Source Notes

- ACE-Step 1.5 MIT: [GitHub](https://github.com/ace-step/ACE-Step-1.5)
- AudioCraft code MIT, weights CC-BY-NC 4.0: [GitHub](https://github.com/facebookresearch/audiocraft)
- Stable Audio Open community terms: [License](https://stability.ai/license), [Model Card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- YuE Apache-2.0: [GitHub](https://github.com/multimodal-art-projection/YuE)

## Routing Table

| Task | Preferred Model | Fallback Model | Why | Cost | Latency | Quality Tradeoff |
| --- | --- | --- | --- | --- | --- | --- |
| Long-form section seed | `ace_step_1_5` | `procedural_dsp` | Stronger musical realism if the local GPU stack exists | Free/open local | Medium | Better realism, heavier setup |
| CPU-safe long-form | `procedural_dsp` | `procedural_dsp` | Guaranteed runnable baseline with no external model dependency | Free/open local | Low to medium | More synthetic, highly portable |
| Structured song experiment | `yue` | `ace_step_1_5` | Useful when explicit multi-minute structure matters more than lightweight ambient rendering | Free/open local | High | Richer structure, more overhead |
| Texture enhancement | `stable_audio_open` | `procedural_dsp` | Optional short texture realism pass | Free under community terms for eligible users | Medium to high | Better realism, optional by policy |

