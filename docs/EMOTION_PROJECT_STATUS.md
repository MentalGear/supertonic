# Emotion Project Status

This file is a handoff for a fresh session. It summarizes the decisions,
implementation, assets, experiments, and next steps from the emotion-control
work.

## Goal

Add reusable `surprised` and `angry` controls to Supertonic voices, with
continuous strength and a path toward inline phrase-level emotion tags.

## Key References

- [supertonic.embed](https://github.com/kdrkdrkdr/supertonic.embed): extracts
  Supertonic-compatible `style_ttl` and `style_dp` JSONs from WAV recordings.
- [supertonic3-voice-clone](https://github.com/saurabhv749/supertonic3-voice-clone):
  Supertonic 3 style optimization and cloning reference.
- [voice-builder-for-supertonic-3](https://github.com/Fawzan09/voice-builder-for-supertonic-3):
  community Colab/T4 workflow for custom styles.
- [EmoShift](https://arxiv.org/abs/2601.22873): emotion-specific activation
  steering vectors.
- [EmoSphere-TTS](https://arxiv.org/abs/2406.07803): continuous normalized
  emotion intensity.
- [RAVDESS](https://zenodo.org/records/1188976): matched neutral, angry, and
  surprised recordings. License: CC BY-NC-SA 4.0.

## Implemented Code

Python:

- `py/helper.py`: `Style.with_deltas()` is now the primary API — it accumulates
  any number of `(delta, weight)` pairs in pre-normalization space and restores
  per-row norms once at the end, so composition is order-independent and a
  zero weight is an exact no-op. `Style.with_emotion()` remains as a
  backward-compatible single-delta wrapper keeping the `[0, 1]` intensity
  constraint. DP stays neutral unless `include_duration` is set, in which case
  it now receives the same per-row projection as TTL.
- `py/example_onnx.py`: supports `--emotion`, `--emotion-style`,
  `--emotion-intensity`, and `--emotion-include-duration`.
- `py/create_emotion_style.py`: creates a style-difference JSON from neutral and
  emotional full styles.

Web:

- `web/helper.js`: browser tensor blending and TTL normalization.
- `web/main.js` and `web/index.html`: emotion selector and intensity slider.
- `web/style.css`: controls styling.

Documentation/notebooks:

- `docs/EMOTION_CALIBRATION.md`: full extraction/calibration guide.
- `docs/EMOTION_ROADMAP.md`: phased roadmap and naming conventions.
- `docs/emotion_calibration_colab.ipynb`: resumable Colab workflow.

## Runtime Contract

The current blend is conceptually:

$$
s' = \operatorname{normalize}(s_{voice} + \alpha \Delta_{emotion})
$$

where `alpha` is intensity from `0.0` to `1.0`. DP is unchanged unless
`--emotion-include-duration` is supplied. The ONNX graph is unchanged.

A first-class gain parameter is planned but not implemented yet. Current gain
experiments were made by multiplying the delta JSON offline, so do not use
those files as the long-term API design.

## Calibrated Assets

These ignored local assets were generated in Colab from matched RAVDESS actor
01 recordings:

```text
assets/emotion_styles/angry.json
assets/emotion_styles/surprised.json
```

Shapes:

```text
style_ttl: [1, 50, 256]
style_dp:  [1, 8, 16]
```

They are a one-speaker proof of concept, not general speaker-independent
emotion vectors. General vectors require multiple speakers and neutral
subtraction per speaker before averaging.

## Audio Outputs

Canonical comparison set:

```text
py/results/listening_sets/m1_meeting/
```

It contains neutral, angry, and surprised outputs at strengths `0.25`, `0.5`,
`0.75`, and `1.0`, plus gain `2.0` and `3.0` experiments. `manifest.json`
records base voice, text, emotion, strength, gain, and experimental status.

Legacy one-off files are preserved under:

```text
py/results/legacy/
```

No generated audio was deleted during cleanup.

## Verification Already Completed

- Official Supertonic 3 ONNX assets downloaded locally under ignored `assets/`.
- Base M1 synthesis works on CPU.
- Angry and surprised assets load successfully.
- Neutral, angry, and surprised audio generated from identical M1 text.
- Strength range generated and renamed with explicit gain metadata.
- Gain 2.0 and 3.0 variants generated; angry gain variants reached peak 1.0,
  so clipping/naturalness evaluation is required.
- Python compilation, JavaScript syntax, web build, notebook JSON, and editor
  diagnostics passed at various checkpoints.

## Colab Recovery

Use `docs/emotion_calibration_colab.ipynb` with a T4 or better. It mounts
Google Drive and stores recordings, Hugging Face cache, extracted styles, and
`extraction_manifest.json` under:

```text
MyDrive/supertonic-emotion-calibration/
```

If Colab disconnects, reconnect the same Drive, rerun setup, and rerun the
extraction cell. Existing completed emotion JSONs are skipped.

## Session Handoff (current state)

**No audio exists in a fresh clone.** `assets/`, `py/assets/` and `py/results/`
are gitignored and are not present after cloning. Every calibrated style,
model file, and WAV referenced below lives only on the machine that generated
it. A new session must download the ONNX assets and preset voices before it can
synthesize anything at all.

**No audio was generated in the session that produced `new-plan.md` and the
`with_deltas()` refactor.** That work was static analysis, documentation, and
numeric tests against synthetic tensors. Nothing has been run through the real
ONNX graph since the blending change.

**Any audio produced by the browser path before that change is invalid.**
`web/helper.js` normalized `style_ttl` across the whole 50x256 block instead of
per row, scaling every row by 1/sqrt(50) — roughly 7.07x too small — at every
intensity setting including neutral. The web demo's output never matched
Python's and its "neutral" was never the base voice. Discard any web-generated
comparison audio.

**Python-generated listening sets are probably still valid, but verify.** The
old Python code normalized rows to unit norm; the new code restores each row to
its original pre-blend norm. Those are identical if and only if the preset rows
are already unit-norm, which `kdrkdrkdr/supertonic.embed` reports is true of the
released presets. Confirm before trusting existing sets, and regenerate if not:

```python
np.linalg.norm(style.ttl, axis=-1)   # expect all ~1.0
```

**Next action is Phase 0 of [new-plan.md](../new-plan.md)**, the linearity gate:
interpolate two preset voices at 0.25/0.5/0.75 and listen. It gates the entire
parametric approach and it is the first thing that needs real assets. The
sequencing that follows it is in that document, not in this one — this
roadmap's phase order is superseded.

## Next Best Steps

1. ~~Add `gain` as a real parameter~~ — superseded. `with_deltas()` accepts any
   finite weight, including values above 1, so a separate gain parameter is no
   longer needed. Done.
2. Generate audio directly with canonical names and update `manifest.json`
   automatically.
3. Add clipping, RMS/loudness, duration, and finite-audio checks.
4. Evaluate gain `1.0`, `2.0`, and `3.0`; choose safe defaults based on
   listening and measurements.
5. Extract matched recordings for at least five speakers, preferably 8-20.
6. Compute speaker-normalized deltas and replace the one-speaker assets.
7. Add extraction-step progress callbacks or `tqdm` to the optimizer.
8. Implement non-nested inline tags through segment synthesis and 10-30 ms
   crossfades.
9. ~~Add automated tests for intensity zero, normalization, gain,
   broadcasting, and neutral preservation~~ — done for the blending layer:
   `py/test_style.py` (21 tests) and `web/helper.test.js` (17 tests), both
   passing, cross-checked against each other to float32 rounding. Tests for
   malformed inline tags remain outstanding, with tags themselves deferred.
10. Expand the backlog with `happy`, `sad`, `calm`, `fearful`, and `excited`.

## Important Cautions

- Do not commit ignored model, recording, style, or generated WAV assets.
- RAVDESS is CC BY-NC-SA 4.0; preserve attribution and license restrictions.
- High gain can clip or produce unnatural speech.
- A single-speaker delta should not be presented as a general emotion model.
- Tags must be parsed before the existing text normalizer, which strips square
  brackets into spaces.
