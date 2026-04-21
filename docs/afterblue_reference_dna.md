# Afterblue Reference DNA

This document records the production target extracted from the user-supplied reference file `forget about it and relax.mp3`.

The goal is not to clone the recording. The goal is to make the engine follow the same production direction: dark low-frequency sleep ambient, slow breathing movement, no harsh air layer, and enough stereo depth to feel immersive.

## Reference Metrics

- Duration: `02:56:57`
- Median loudness: `-18.887 LUFS`
- Median spectral centroid: `153.645 Hz`
- Median 85% rolloff: `199.425 Hz`
- Median short-term range: `17.7835 dB`
- Median stereo side/mid: `0.2792`
- Dominant modulation: `0.067 Hz`
- Bass share `60-120 Hz`: `0.5304`
- Low-mid share `120-300 Hz`: `0.3451`
- Body share `300-800 Hz`: `0.0688`
- Presence, air, hiss: effectively absent

## Engine Target

- Profile: `afterblue_reference_sleep`
- Tonal anchor: `C#3`, with sub-octave energy around C#2.
- Scale family: `afterblue_minor_cluster`, matching the dark C# / E / G / A / B color without forcing a copied progression.
- Loudness target: `-19 LUFS`.
- Breath rate: `0.067 Hz`.
- Core texture policy: `sub_room` and `black_room`.
- Forbidden artifacts: broadband hiss, TV-static noise, bright air layer, busy percussion, harsh upper mids.

## Validated 10-Minute Test

Session:

`sessions/20260421_141604_afterblue_reference_sleep_77`

QC:

- Accepted: `true`
- Global score: `90.43`
- Reference DNA score: `0.8646`
- Bass anchor score: `0.8681`
- Dynamic breath score: `0.7772`
- Low-mid boxiness risk: `0.0`
- Integrated loudness: `-19.006 LUFS`

External comparison against the reference:

- Bass share: reference `0.5304`, engine `0.5125`
- Low-mid share: reference `0.3451`, engine `0.4587`
- Body share: reference `0.0688`, engine `0.0277`
- Stereo side/mid: reference `0.2792`, engine `0.2386`
- Dominant modulation: reference `0.067 Hz`, engine `0.0671 Hz`
- Hiss/presence: effectively zero in both

## Remaining Gap

The engine now matches the reference direction much more closely than the earlier `afterblue_sleep` renders, especially in bass anchor, absence of hiss, and slow modulation. It still has less body in `300-800 Hz` and less long-range dynamic contrast than the full 3-hour reference. Those gaps should be addressed with better generated motifs or curated/user-owned warm body stems, not by adding noise or bright texture.
