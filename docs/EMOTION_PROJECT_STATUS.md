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
- `docs/LISTENING_BENCHES.md`: index of published Artifact listening benches,
  their generators, and the verdicts they produced.

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
- Phase 2b of [new-plan.md](../new-plan.md), the speaker-embedding probe, run on
  2026-09-09 with the `py/phase2b_*.py` scripts and closed the same day. Neither
  ECAPA nor WavLM predicts `style_ttl` for unseen voices; WavLM is about twice
  as good as ECAPA and still recovers none of the perturbation. Details in the
  handoff section below.

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

**Phase 2b has been run for both ECAPA and WavLM, came back negative, and is
closed (2026-09-09).**
Phase 2 as written was blocked here — its optimizer is external and there is no
corpus or GPU on this machine — but the engine is its own paired-data generator,
so synthesizing from known styles gives ground-truth `(audio, style)` pairs. The
`py/phase2b_*.py` scripts built 1,920 such pairs: six conditions of 320 (preset
blends, then random per-row perturbations at eps 0.05 through 0.80) over 8 texts
and 7 training presets, with M5, F4 and F5 held out, seeded per sample, with
`style_dp` fixed at M1's. **A linear probe from ECAPA embeddings to `style_ttl`
reaches family-disjoint R^2 of only 0.192 on preset blends and 0.007 at the
widest perturbation**, against a linear ceiling of 1.000 and 0.746; against the
held-out set's own mean every one of those figures is negative. Random-split
R^2 was 0.925-0.962 and tracks "which base preset was this" almost exactly, so
the naive experiment would have reported ~0.93 and meant nothing. Every control
agrees: shuffled-target R^2 -0.000 to -0.020 (nothing to fit), an RBF kernel
probe worse than linear with real control-task leakage, ECAPA itself fine on
this audio (100% 1-NN speaker ID, chance 10%), multi-utterance averaging worth
0.011, and a 120-feature MFCC-moment baseline matching or beating ECAPA
everywhere. All 24 active rows are negative under the family-disjoint split.

The mechanism is the more useful half, and **the mechanism first recorded was
wrong.** The perturbation sweep found the audio never stops being voice-like —
F0, voiced fraction, spectral flatness, modulation and WER flat across the whole
ladder, and a controlled ray at a 72-degree per-row rotation (twice the M1->F1
angle) still had WER 0.00 — and that was read as "most of style space is nearly
inaudible, so the audio-to-style inverse is ill-posed in nearly every
direction." **A listener heard the ray and contradicted it**: the clips clearly
differ, increasingly with magnitude, and what changes is which words are
emphasised (M1) and per-word loudness (F1). The flat metrics were all
utterance-level aggregates, which collapse the time axis the effect lives on. A
frame-level follow-up (`py/phase2b_prosody_*.py`, reports under the ignored
`py/results/phase2b_prosody/`) confirms the listener: on M1 from eps 0.20 to
3.20, utterance level goes +0.4 -> +4.9 dB and per-word energy spreads over 1.6
-> 10.1 dB while median F0 moves 115 -> 119 Hz and WER stays 0.00; at eps 0.80 a
random direction moves the log-mel spectrogram 8.4 dB rms against 16.6 dB for a
full M1->F1 swap. The "3-5x more audible" figure was measured with delta-ECAPA,
a speaker-verification embedding trained to be prosody-invariant; on
prosody-sensitive measures the same comparison gives 1.1-3.2x.

**Corrected: style space is anisotropic in what a direction changes, not in
whether it changes anything.** At matched per-row angle a random direction moves
median F0 by 40 cents where a preset-aligned one moves it 790, and delta-ECAPA
3.5-6.5x less, but the log-mel spectrogram only ~1.7x less. So the null means
**ECAPA cannot see most of what a style direction does**, not that most style
directions do nothing — there is no information ceiling on an encoder; the
information is in the audio, it was not in ECAPA. That still kills the probe
shortcut and still points Phase 2 at the direct encoder (2a), but for a
different reason: the *encoder input* was the wrong representation, so a
prosody-bearing input (WavLM, or explicit prosodic features) is worth testing
before concluding the inverse is ill-posed. The strong two-subspace reading —
a clean "identity" subspace orthogonal to a "prosody" one — is **not**
supported: the broadband-versus-spectral-shape split of the diff is essentially
the same across direction families, and at matched total acoustic change
the localisation advantage disappears. A graded version is: directions differ in
what they move and in how much travel it costs, and random directions do reach
F0 and timbre, just far along the ray. What survives untouched is that inside
the preset-spanned subspace embedding distance tracks style distance (Spearman
0.41 against 0.00 for random directions), which still helps Phases 3 and 4.
Scope: synthetic audio, engine-generated styles, three held-out identities; the
ray is one seeded random direction per base, word boundaries came from
faster-whisper tiny.en with a hand-stated repair, F0 above eps 1.60 is
pyin-unreliable, and `style_dp` was pinned throughout.

**The WavLM half closes the phase, and the positive in it is real.**
`py/phase2b_wavlm_embed.py` and `py/phase2b_wavlm.py` ran WavLM-large
(torchaudio's `WAVLM_LARGE`) over all 1,920 existing clips — no subsetting,
nothing re-synthesized, no audio rendered — reusing the same styles and
manifest, 0.55 s/clip for 18 minutes, importing `phase2b_probe`'s functions so
splits, control task and reporting are identical, and refitting ECAPA on the
same indices for an exact comparison. Best representation is layer 3, mean+std
pooled, 2048-dim. Family-disjoint R^2 against the train-mean baseline runs
**0.320 / 0.192 / 0.162 / 0.142 / 0.116 / 0.042** down the ladder from preset
blends to eps 0.80, against ECAPA's
0.192 / 0.089 / 0.086 / 0.067 / 0.057 / 0.007 —
**roughly 2x everywhere, with shuffled-target selectivity ~0.000, so the raw
score is the selectivity.** The layer sweep over all 24 layers decays
monotonically from early to late (L1 .323, L3 .320, L4 .303, L5 .288, L12 .260,
L24 .189 on preset blends), with L3/L4/L5 indistinguishable at the top — which
**independently corroborates `supertonic.embed`'s choice of layer 4**; the last
layer would have cost more than half the signal. WavLM also works on this audio:
1-NN preset ID 0.86-0.95 for layers 1-5 against 10% chance (L3 = 0.887, ECAPA
1.000), so domain mismatch is not the explanation.

**It does not change the conclusion, and the capacity hypothesis is falsified.**
The pooled condition is the cleanest statement: raising the reachable ceiling
from 0.758 to 0.999 — 32 points of extra headroom — bought 5.4 points of actual
R^2 (0.088 -> 0.142). The probe is information-limited, not ceiling-limited.
Concatenating four layers to 8,192 dims changed nothing (0.136 vs 0.142) and RBF
kernel ridge was worse than linear (0.105 vs 0.142 at eps 0.20). The ceiling
arithmetic in the earlier record was also wrong and is corrected here: the ratio
is min(d, n_train)/6120, set by the rank of the ridge fit rather than the width
of its input, so 240 training clips give 240/6120 = 3.9% for WavLM against
ECAPA's 192/6120 = 3.1% — not 2048/6120 = 33%. The decisive control is the
**within-family residual**, base preset subtracted so the target is the
perturbation itself: ECAPA -0.0000 / +0.0000 / +0.0003 and WavLM
-0.0000 / +0.0000 / +0.0014 at eps 0.05 / 0.20 / 0.80, against a residual
ceiling of ~0.82. Both collapse to a constant predictor. **WavLM's entire gain
is on "which base voice was this" and none of it is on the perturbation** —
78.8% of eps 0.20's target variance is between-base-preset, the probe recovers
a slice of that and none of the remaining 21.2%.

**The synthesis first recorded here — "audible does not mean identifiable" —
has since been refuted by the direct test it named as unrun.** What stood at
this point read the two results above (audible, but WavLM still zero on the
residual) as: a 6,120-dimensional perturbation collapses onto a low-dimensional
audible readout — roughly an utterance level plus a per-word emphasis pattern,
on the order of ten numbers for this sentence — so many different directions
produce nearly the same audible consequence, and the audio-to-style inverse is
many-to-one for that reason. It was flagged at the time as inference, with the
direct test named and explicitly not run: whether distinct random directions at
matched magnitude produce *similar* audible readouts from the same base.

**That test has now been run (`py/phase2b_direction_collapse.py`,
2026-09-09) and refutes it.** K=24 mutually near-orthogonal random directions
from base M1, at the same eps 0.20/0.80 magnitudes, are *distinguishable*, not
collapsed: pairwise per-word-emphasis correlation between directions is
weak (mean r +0.158 / +0.060, and 55% of all pairs fall inside the null band
for unrelated 9-word profiles), two directions typically differ from each
other in log-mel spectrogram *more* than either differs from the base
(5.5-10.4 dB between-direction vs. 4.8-9.6 dB from-base), and direction
identity survives an independent vocoder-latent redraw: 1-NN against the
other 23 directions picks the true one 70.8% (eps 0.20) / 66.7% (eps 0.80) of
the time on the log-mel diff map (chance 4.2%, p < 1e-16 both). **Specifically
refuted:** "many different directions produce nearly the same audible
consequence," and the claim it licensed that well-chosen audio metrics might
agree between far-apart styles — measured directly, they disagree, sharply.

**The Phase 2b null survives, for a different reason: the inverse is narrow,
not many-to-one.** The audible readout genuinely is low-dimensional
(participation ratio 3.7 of 9 for emphasis, 4.6-7.0 of 23 for the log-mel diff
map), so perfectly inverting it recovers at most ~7/6,120 = 0.1% of the
6,120-dimensional target — but those few dimensions are cleanly
direction-specific rather than shared across directions. The energy contour is
the one readout that genuinely is shared (pairwise r up to +0.50, and it fails
cross-latent identification); the emphasis pattern and spectral detail are not.
**Admission carried forward:** the within-family residual control
(R^2 +0.0000 to +0.0014, cited above as decisive) is exactly what *both* the
collapse account and the narrow-inverse account predict, so it never had the
power to distinguish them — it is not evidence for collapse, nor against
recoverability.

**The instruction for Phase 2a changes as a result.** The limit is dimensional
and *additive across utterances*, not a property of the audio representation —
a different sentence exposes a different 5-10 dimensions, not a better view of
the same ones. **The lever for an encoder is many utterances per style, not a
better single-utterance representation.** A companion check
(`py/phase2b_wavlm_render.py`, `probe_recovery_triples.json`) reinforces this
from the other side: refitting both probes and adding a constant train-mean
baseline, both probes' rendered predictions sit 14-21 dB (log-mel) from the
true style.

**Correction (2026-09-09): the follow-on claim built on that figure — "as far
as, or further than, a full identity swap (16.6 dB)," with the train-mean
baseline trailing WavLM by only 0.017-0.028 cosine — is false and withdrawn.**
The 16.6 dB threshold came from one pair (M1-F1); calibrated across all 80
held-out samples, same-speaker pairs (different vocoder seed) average 17.42 dB
and different-speaker pairs average 17.82 dB — indistinguishable
distributions, so log-mel distance at this scale cannot separate same- from
different-speaker and the "as far as a different speaker" framing never held.
Recalibrated on ECAPA cosine (the metric actually built to measure speaker
identity): WavLM's reconstructions average 0.432 against true style, inside
the different-speaker range (anchor 0.225) and short of the same-speaker floor
(0.669-0.879) — so the reconstructions still fall short of true identity, but
for a valid reason this time, not the invalid log-mel one. And the margin over
train-mean is real, not the near-zero 0.017-0.028 first reported: WavLM beats
train-mean on 80/80 samples and a same-gender impostor on 95%, including
29/29 on the one held-out identity (M5) where a gender confound cannot
explain it. **This narrows, not reopens, the closure**: it is evidence about
which base voice the probes recover (predicted below, in the between-preset
share of target variance), not about the within-family residual that is the
closure's load-bearing claim. Full numbers and derivation are in
[new-plan.md](../new-plan.md) under Phase 2b.

Caveats: one base, one sentence, so the dimensionality figures are
per-(base, sentence); at eps 0.20 the emphasis profile sits near the
vocoder-nuisance floor and is weakly identifiable, but the log-mel diff map is
unambiguous at both magnitudes. Scope on the WavLM half is the scope above: the
same synthetic audio, the same engine-generated styles from base presets, the
same three held-out identities. Full numbers, including the direction-collapse
and probe-reconstruction detail, are in [new-plan.md](../new-plan.md) under
Phase 2.

**What this means for emotion.** `style_ttl` demonstrably has prosodic reach:
random directions produce monotone, magnitude-scaled changes in utterance level
and per-word emphasis while leaving intelligibility (WER 0.00) and pitch
register intact, and the emphasis effect replicates on two further sentences
(8.1 and 10.0 dB at eps 0.80). **Emphasis and loudness therefore look reachable
as emotion axes** without touching the duration tensor. Two caveats go with
that. First, a random direction produces *unstructured* emphasis jitter, not a
coherent emotional contour — this shows the lever exists; an actual axis still
has to be derived. Second, **`style_dp` was pinned throughout the entire
analysis**, so speech rate and timing — a first-order emotion cue — lie outside
everything measured here.

Results live under the ignored `py/results/phase2b/` and are not present in a
fresh clone — 343 MB before the WavLM run, plus `wavlm_feats.npz` at 377 MB
(all 24 layers, mean+std, for all 1,920 clips). The environment now has `torch`
2.11.0+cpu, `torchaudio` 2.11.0, `speechbrain` 1.1.1 (for ECAPA) and
`faster-whisper` 1.2.1 (for WER) installed; the probe scripts need them. The
torchaudio `WAVLM_LARGE` weights are cached at
`/root/.cache/torch/hub/checkpoints/wavlm_large.pth` (1.18 GB) and will be
re-downloaded on a fresh machine.

The two Phase 0 runs were the first things to touch the real ONNX graph since
the `with_deltas()` refactor, and Phase 2b's 1,920 renders are the largest use
of it so far. The rest of the 2026-09-08 session's work was static analysis,
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

**Next action: Phase 2a — it is the only remaining route.** Phase 0 passed, its
companion probe is done, Phase 1's composition work already landed with
`with_deltas()` (1a is verified above), and Phase 2b is closed negative on both
ECAPA and WavLM. What is left is 2a, the direct `audio -> style_ttl` encoder on
generated pairs. Two things bound it, and they pull in opposite directions.
Nothing bounds it from the audio side: the perturbation is audible, and a
prosody-bearing input does read more of it, so feed 2a WavLM-class features
(layers 3-5, mean+std, which is also what `supertonic.embed` uses) rather than a
speaker-verification embedding. But no probe of this class recovered the
perturbation *direction* from either input, so the encoder is genuinely
unproven, and **2b's instruction to evaluate it against the optimizer's
converged style rather than against downstream audio matters more, not less —
now for a corrected reason.** It is not, as first thought, that a well-chosen
audio metric might falsely agree between two styles that are far apart in
style space — the direction-collapse test above refutes that; distinct
directions disagree in the audio, sharply. It is that the audio readout is
narrow (on the order of 5-10 dimensions out of 6,120 per utterance), so an
encoder can match the audio closely — as the probe-reconstruction check shows
directly, both probes' predictions still landing inside the different-speaker
range on calibrated ECAPA cosine despite genuine R^2 (0.432 WavLM / 0.383
ECAPA against a same-speaker floor of 0.669; see the correction above) —
while leaving most of the target unconstrained. Style-space evaluation
catches that; audio proximity alone does not. **And because the limit is
additive across utterances, not representational, train and evaluate 2a
against many renders per style, not a single-utterance objective — that is the
lever, not a better per-utterance feature.** Axis work in Phases 3-4 should
still be fitted inside the preset-spanned subspace where 2b found the
conditioning is good. The ordering is in [new-plan.md](../new-plan.md), not in
this one; this roadmap's phase order is superseded.

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
