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
- Phase 0 of [new-plan.md](../new-plan.md), the linearity gate, run on the real
  ONNX graph on 2026-09-08 via `py/phase0_linearity_gate.py`. Passed. Details
  in the handoff section below.
- Phase 0's companion row-structure probe run the same day via
  `py/phase0_row_locality.py`. `style_ttl` is localized at the row-group level,
  confirmed by ear. Details in the handoff section below.

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

**Phase 0 has been run and it passed (2026-09-08).**
`py/phase0_linearity_gate.py` interpolated `style_ttl` between presets M1 and
F1 at weights 0.00 / 0.25 / 0.50 / 0.75 / 1.00 through
`with_deltas([(delta, w)], include_duration=False)`, holding `style_dp` at M1's,
and synthesized "The quick brown fox jumps over the lazy dog." (en,
`total_step=8`, `speed=1.05`, 44.1 kHz, about 3.10 s per clip) at each point.
Outputs and `manifest.json` are under the ignored
`py/results/listening_sets/phase0_linearity/`. Endpoints round-trip: `w=0.00`
recovers M1's TTL bit-for-bit (max abs diff 0.0) and `w=1.00` recovers F1's to
float32 rounding (max abs diff 2.98e-07). Intermediate weights stay unit-norm
(`w=0.50` row norms 0.99999976 to 1.0000002), with no clipping, no NaNs, and
peaks well under 1.0. The set was listened to: the midpoints are clean,
plausible voices. **Verdict: the gate passes. Linear travel through style space
is viable on this evidence, so the parametric approach proceeds rather than
being re-scoped to a learned manifold.** Scope of that evidence is one voice
pair, one sentence, and one set of inference settings — it does not validate the
style space generally.

**The companion row-structure probe has also been run (2026-09-08).**
`py/phase0_row_locality.py` built 66 hybrids of M1's `style_ttl` with named rows
replaced verbatim by F1's — the 50 single-row swaps, contiguous band swaps,
top-k sets by share of the M1->F1 delta, and an active/inactive split derived
from per-row spread across all ten shipped presets — on the same text and
settings as the linearity gate, with the RNG seeded so clips differ only by the
style tensor. **`style_ttl` is localized at the row-group level, not diffuse.**
Across the ten presets, per-row spread from the centroid is sharply bimodal, and
a 24-row active set

```text
[0, 2, 5, 6, 7, 8, 9, 13, 15, 16, 18, 19, 20, 22, 23, 27, 31, 32, 38, 42, 45,
 47, 48, 49]
```

carries 99.8% of the M1->F1 delta; the other 26 rows are near-constant in every
released voice. The user listened to the pair and confirmed the split: the
24-active-row hybrid reads as F1, the 26-frozen-row hybrid still reads as M1.
No single row is the gender switch — the strongest, row 15, reaches only ~0.30
travel against 1.00 for the full swap — so future axes should be fitted on the
~24 active rows (6,144 parameters instead of 12,800) with the rest held fixed. Caveats: swapping only
the 26 inactive rows still moved the audio a little, so do not hard-zero them
without listening; the distance metric (mean |Δ log-STFT|) saturates, so its
`travel` numbers are ordinal, not fractions; and the delta profile itself is one
voice pair, only the active/inactive split spans all ten presets. Full numbers
are in [new-plan.md](../new-plan.md) under Phase 0.

Those two runs are the first things to touch the real ONNX graph since the
`with_deltas()` refactor. The rest of that session's work was static analysis,
documentation, and numeric tests against synthetic tensors.

The model assets it needed were fetched with
`git clone https://huggingface.co/Supertone/supertonic-3 assets` (about 392 MB,
requires git-lfs) and remain gitignored.

**Any audio produced by the browser path before the `with_deltas()` refactor
is invalid.**
`web/helper.js` normalized `style_ttl` across the whole 50x256 block instead of
per row, scaling every row by 1/sqrt(50) — roughly 7.07x too small — at every
intensity setting including neutral. The web demo's output never matched
Python's and its "neutral" was never the base voice. Discard any web-generated
comparison audio.

**Preset TTL rows are unit-norm — now measured, not assumed.** The Phase 0 run
reports per-row TTL norms of 1.0000000 for both M1 and F1 (min
0.9999998211860657, max 1.000000238418579), matching what
`kdrkdrkdr/supertonic.embed` claims of the released presets. The old code's
normalize-to-unit and the new code's restore-original-norm therefore agree on
these presets, so existing Python-generated listening sets are not invalidated
by the blending change. Only M1 and F1 were measured, and `style_dp` row norms
were not measured at all. Check any other preset before relying on it:

```python
np.linalg.norm(style.ttl, axis=-1)   # expect all ~1.0
```

**Next action is Phase 2 of [new-plan.md](../new-plan.md): amortized style
inversion.** Phase 0 has passed, its companion probe is done, and Phase 1's
composition work already landed with `with_deltas()` (1a is verified above), so
nothing remains at the front of that plan and its sequencing puts Phase 2 next —
train `audio -> style_ttl` on optimizer-generated pairs (2a) and run the
ECAPA/WavLM linear probe with its control task and speaker-disjoint splits (2b).
Report the probe's per-row R^2 over the 24 active rows separately from the
aggregate; the near-constant rows would flatter a whole-tensor number. The
ordering is in that document, not in this one; this roadmap's phase order is
superseded.

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
