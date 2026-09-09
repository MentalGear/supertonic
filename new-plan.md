# Parametric Voice Space — Research Plan

Status: proposal. Supersedes the `new-plan` sketch. Phase 0 was run on
2026-09-08 and passed; Phase 2b was run on 2026-09-09 with both ECAPA and
WavLM, came back negative on both, and is closed. See the results recorded
under those phases.

## Goal

Replace fixed preset voices with a small set of **continuous, orthogonalized
attribute axes** over Supertonic's style space, so perceived age, vocal
presentation, and emotion can be dialed at (or before) runtime.

Explicit non-goal: composing an arbitrary *specific* person's voice from
scalars. Speaker identity is high-dimensional and idiosyncratic — glottal
source characteristics, formant fine structure, idiolectal timing. Age and
vocal presentation are low-dimensional marginals of it. What this plan can
deliver is a controllable voice space: shift a base voice along an attribute,
and sample plausible unseen voices by perturbing non-attribute directions.

Framing the goal as "no specific voices needed" overstates what is reachable
and would set the project up to be judged a failure at something it was never
going to do.

Secondary motivation worth stating: fully synthetic parametric voices sidestep
the cloning-consent problem that zero-shot cloning creates.

**Scope risk to name up front:** the highest axis count demonstrated in
published work is three (VoiceShop: age + gender + accent). This plan implies
roughly five once emotion axes are included. Nothing surveyed tests that many,
and the documented failure mode as axes accumulate is cross-attribute leakage.
Treat anything beyond three simultaneous axes as unproven territory, and stage
the axis count upward with the leakage protocol in Phase 5 rather than
assuming composition scales.

## Correction to the Premise

The original sketch cited `docs/` as having "good results doing dynamic emotion
adaptation." The docs say the opposite. Per
[EMOTION_PROJECT_STATUS.md](docs/EMOTION_PROJECT_STATUS.md), the calibrated
assets are a **one-speaker proof of concept** from RAVDESS actor 01. Gain 2.0
and 3.0 experiments reached peak 1.0 (clipping risk). No speaker-similarity,
emotion-classifier, or intelligibility evaluation has been run. Roadmap phases
1-3 are unstarted.

This matters because the parametric plan inherits the same unverified
assumption — that style deltas are linear, additive, and transferable across
speakers. Nothing has tested that yet. Phase 0 below tests it before any
further investment.

## What "Parametric" Can Actually Mean Here

The ONNX graph is frozen. Its only conditioning inputs are:

```text
style_ttl: [1, 50, 256]
style_dp:  [1, 8, 16]
```

So a parametric *engine* is not on the table. What is on the table is a
**parametric control layer over the style space**: a learned map from
attributes to those two tensors. That is a strictly additive layer — no graph
surgery, no retraining of the acoustic model.

### Runtime cost: not a constraint

Style composition is one matrix add plus a row normalize over 12,800 floats,
performed once per utterance, before the DP -> text-encoder -> vector-estimator
-> vocoder chain runs. Even replacing the add with a small MLP mapping
attributes to a style costs microseconds against the `total_steps` sampling
loop.

The only place attribute control costs real time is **per-segment switching**,
which forces segment-wise synthesis (see emotion roadmap phase 4). That cost is
`O(segments)` and is independent of how many attribute parameters exist.

Conclusion: do not let compute budget shape this design. The binding costs are
data collection and evaluation, and both are offline.

## Phase 0: Linearity Gate (do this first)

Everything downstream assumes the style manifold is locally linear and
well-behaved between speakers. Test it in about an hour.

Take two preset voices (M1, F1). Interpolate `style_ttl` at 0.25 / 0.5 / 0.75.
Synthesize identical text at each point.

- **Midpoints sound like clean, plausible, distinct voices** -> the space is
  linear enough. Proceed.
- **Midpoints sound smeared or artifacty** -> a linear delta model will not
  carry this. Re-scope to a learned manifold over style space (VAE or
  normalizing flow), which is a materially larger project.

**Kill criterion:** if interpolation fails, stop and re-scope. Do not proceed
to phases 2+ on the assumption it will work out.

Record the outputs under a listening set with the existing manifest convention.

### Result: passed (2026-09-08)

Run with `py/phase0_linearity_gate.py`. `style_ttl` was interpolated between
presets M1 and F1 at 0.00 / 0.25 / 0.50 / 0.75 / 1.00 via
`with_deltas([(delta, w)], include_duration=False)`, with `style_dp` held at
M1's; text "The quick brown fox jumps over the lazy dog." (en, `total_step=8`,
`speed=1.05`, 44.1 kHz, about 3.10 s per clip). Outputs and `manifest.json` are
under the ignored `py/results/listening_sets/phase0_linearity/`.

Numerically: `w=0.00` recovers M1's TTL bit-for-bit (max abs diff 0.0) and
`w=1.00` recovers F1's to float32 rounding (max abs diff 2.98e-07);
intermediate weights stay unit-norm (`w=0.50` row norms 0.99999976 to
1.0000002); no clipping, no NaNs, peaks well under 1.0. By ear, the midpoints
are clean, plausible voices.

**Verdict: the gate passes. Linear travel through style space is viable on this
evidence, so the parametric approach proceeds rather than being re-scoped to a
learned manifold.** The evidence is one voice pair, one sentence, and one set of
inference settings — it clears the gate, it does not validate the style space
generally.

The companion row-structure experiment below has since been run too; its result
is recorded after it.

### Companion experiment: is the 50x256 grid structured?

The architecture is now partly known, and it changes the rationale for this
experiment without removing the need for it.

The SupertonicTTS paper ([arXiv:2503.23108](https://arxiv.org/abs/2503.23108),
Appendix A.2) states that "50 learnable vectors ... are used in the first
attention block" of the reference encoder, and that "these 50 vectors are
reused as keys in the VF estimator." So the 50 rows are **learnable attention
queries in a Perceiver-style bottleneck** — not time steps, not a layer stack,
and not slots designed to carry separable human-interpretable attributes. Our
local code agrees on the consumption pattern: `style_ttl` feeds the text
encoder and the vector estimator, `style_dp` feeds only the duration
predictor (`py/helper.py:228-247`).

Two caveats on that, both material:

- **The paper describes a predecessor.** It covers SupertonicTTS (44M params,
  English-only) and gives the learnable vectors dimension **128**. This repo
  runs Supertonic 3 (~99M, 31 languages) with `style_ttl` at 50 x **256**. The
  row count carries over; the width does not. No architecture paper for v3 was
  found, so treat the query interpretation as strongly suggested, not
  confirmed for our weights.
- **Per-row L2 normalization is not stated in the paper.** It is observed in
  the released presets and enforced by community extractors, but the reason is
  inferred. Unit-norming attention keys is ordinary magnitude stabilization,
  so it is **not** evidence that rows specialize semantically — which is the
  opposite of the inference an earlier draft of this plan drew from it.

That weakens the prior for a localized outcome but raises the value of
measuring, because nobody has. All three community extraction tools
(`kdrkdrkdr/supertonic.embed`, `saurabhv749/supertonic3-voice-clone`,
`Fawzan09/voice-builder-for-supertonic-3`) optimize `style_ttl` as a single
monolithic 12,800-parameter block against one global loss, none treats rows
independently, and none reports which rows moved. No public per-row ablation of
this tensor exists.

Cheap probe: take voices A and B, and synthesize 50 hybrids where hybrid `i`
uses A's TTL with row `i` replaced by B's. Measure and listen.

- **Diffuse** (every row shifts the voice slightly) -> axes must span all rows;
  deltas stay dense. Proceed as planned.
- **Localized** (a handful of rows dominate identity, others prosody or
  timbre) -> derive axes per row group instead. Much better conditioned, far
  fewer parameters to fit, and it makes "change age without changing identity"
  a structural property rather than something orthogonalization has to
  enforce after the fact.

Diffuse is the likelier outcome given the query interpretation above, but there
is a real precedent for the other side: *Eigenvoice Synthesis based on Model
Editing* ([arXiv:2507.03377](https://arxiv.org/abs/2507.03377)) builds an
orthogonal SVD basis over speaker parameter deltas and finds gender loading
almost entirely on the sign of one component. (Measured otherwise here — see the
result below.)

A few hours of work, it could substantially simplify Phase 3, and either way it
would be the first documented probe of this tensor's structure.

### Companion result: localized (2026-09-08)

Run with `py/phase0_row_locality.py`. M1's `style_ttl` with named rows replaced
verbatim by F1's, 66 hybrids: the 50 single-row swaps proposed above, contiguous
band swaps (halves and fifths), top-k sets by share of the M1->F1 squared delta
(k = 3/5/10/15/20), and an active/inactive split derived from per-row spread
across all ten shipped presets (M1-M5, F1-F5). Same text and settings as the
linearity gate, `style_dp` held at M1's, `include_duration=False`. RNG seeded
before every synthesis, so clips differ only by the style tensor; two renders of
the same tensor were confirmed bit-identical.

The M1->F1 delta is concentrated. Per-row cos(M1, F1) runs 0.796 min / 0.949
mean / 0.9999 max, and ten rows carry 61% of the total squared delta against 20%
for uniform: rows 48, 15, 32, 38, 2, 23, 45, 16, 49, 47. A row's share of the
delta predicts how far its single-row swap moves the audio (correlation 0.916),
and 21 of the 50 single-row swaps move it by less than 2% of the M1->F1
distance.

The activity split is not specific to that pair. Across all ten presets, per-row
spread from the centroid is sharply bimodal — 21 rows above 0.34, a cliff, then
24 rows below 0.06. Taking 0.1 as the threshold gives a **24-row active set**:

```text
[0, 2, 5, 6, 7, 8, 9, 13, 15, 16, 18, 19, 20, 22, 23, 27, 31, 32, 38, 42, 45,
 47, 48, 49]
```

The complementary 26 rows are near-constant in every released voice; rows 3, 12,
17, 24, 33, 34, 36, 37, 40 and 46 are the deadest (pairwise cos >= 0.996 across
presets). Sparse edits nearly suffice: the top-20 rows (98.8% of the delta)
reach 0.78 travel and the 24-row active set (99.8%) reaches 0.79, against 1.00
for the full swap. Contiguous band swaps only track their delta content
(0.24-0.51 travel), so the active rows are **scattered, not contiguous** — group
by measured activity, never by index range.

No single row is the gender switch. The strongest, row 15, reaches only ~0.30
travel. That cuts against the Eigenvoice precedent cited above: gender here is
roughly twenty scattered components, not one sign bit.

The split was confirmed by ear. Swapping the 24 active rows produces a hybrid
that reads as F1; freezing them and swapping the other 26 produces one that
still reads as M1.

**Verdict: localized at the row-group level. Derive future axes on the ~24
active rows and hold the rest fixed — a 6,144-parameter fit instead of 12,800,
better conditioned, and "change age without changing identity" gets a structural
handle rather than needing post-hoc orthogonalization.**

Caveats:

- **The inactive rows are not free to zero.** Swapping only the 26 inactive rows
  still moved the audio by 0.14 travel. Freeze them, but do not hard-zero them
  without a listening check.
- **The distance metric saturates.** Distances are mean |Δ log-STFT|; the 50
  single-row distances sum to 4.80 against an endpoint distance of 0.97. Treat
  `travel` as ordinal, not as a fraction of the way to the target.
- **Scope.** The delta profile — the top-ten rows, the 0.916 correlation, the
  travel figures — is one voice pair, M1/F1, on one sentence. Only the
  active/inactive split is drawn from all ten presets. Expect the active set to
  hold; do not assume another pair's delta lands on the same ten rows.

Row norms were incidentally re-checked across all 66 tensors: worst deviation
from unit norm was 2.4e-07, and that worst case was the unmodified M1 endpoint,
i.e. the float32 precision of the shipped presets rather than an artifact of
swapping.

## Phase 1: Fix the Composition Algebra

Two defects in the current blending code block reliable multi-axis work.

### 1a. `intensity=0.0` may not be a no-op

`py/helper.py:172-173`:

```python
ttl = self.ttl + intensity * emotion_ttl
ttl = ttl / np.linalg.norm(ttl, axis=-1, keepdims=True).clip(min=1e-8)
```

At `intensity=0.0` this returns `normalize(self.ttl)`, not `self.ttl`. If the
preset voice TTL rows are not already unit-norm, **intensity zero silently
changes the voice**, and every "neutral" reference recorded so far is a
different voice than the base.

Verify directly:

```python
np.linalg.norm(style.ttl, axis=-1)   # all approximately 1.0?
```

If not unit-norm, either rescale to the original per-row norm rather than to
unit, or skip the normalize entirely when no delta is applied. Add a regression
test asserting `base.with_deltas([]) == base` exactly.

Prior art suggests this is probably benign but still worth confirming.
`kdrkdrkdr/supertonic.embed` states that every row of *both* tensors is a unit
vector in the released presets, and its own optimizer re-projects all 50 rows
onto the unit sphere after each step. If that holds for our assets, the
normalize at zero weight is a no-op rather than a corruption — but the whole
emotion listening set rests on it, so measure rather than assume. Note also
that `style_dp`'s 8 rows are unit-normalized too, which the current blending
code does not account for: **DP deltas will need the same row projection once
Phase 3 starts moving them.**

**Verified (2026-09-08).** The Phase 0 run measured per-row TTL norms for M1
and F1: all 1.0000000 (min 0.9999998211860657, max 1.000000238418579). Zero
weight is an exact no-op on these presets, so no recorded "neutral" was
silently a different voice. Only M1 and F1 were measured, and `style_dp` row
norms were not, so the DP caveat above still stands.

### 1b. Normalize-after-add is non-commutative

Because the row normalize happens inside each blend, applying age then emotion
does not equal applying emotion then age, and `intensity` is not linear in
perceptual effect. With three to five axes this becomes unmanageable.

Fix: accumulate every delta in pre-normalization space and normalize **once**
at the end.

### 1c. Generalize the API before adding axes

Attribute axes are the same machinery as emotion deltas. Do not build a second
parallel system. Refactor:

```python
style = base_style.with_deltas([
    (age_axis,     +0.4),
    (presentation, -0.2),
    (emotion_angry, 0.7),
])
```

`with_emotion()` becomes a thin wrapper for backward compatibility. Carry over
the planned first-class `gain` parameter from emotion roadmap phase 1 as a
per-axis scale.

Also complete the outstanding items from that roadmap that this plan depends
on: output metadata recording every axis and weight, and automated tests for
zero-weight, broadcasting, and row-norm preservation.

## Phase 2: Amortized Style Inversion (the enabler)

Today, extracting a style from audio means gradient optimization through the
frozen graph (`supertonic.embed`). At minutes per speaker, scaling from 20
speakers to hundreds is bottlenecked on GPU time forever. Every later phase
depends on removing that bottleneck.

**Do not use vec2vec for this.** vec2vec solves *unpaired* embedding-space
translation. That is not the problem here — the existing optimizer generates
unlimited `(audio, style)` pairs on demand, so this is supervised distillation
of the optimizer, not translation. Paired regression is simpler, better
conditioned, and directly yields what a "recovered embedder" would give.

Two variants, run both:

**2a. Direct encoder, then refinement.** Train `audio -> style_ttl` on
optimizer-generated pairs, turning per-speaker extraction into a forward pass.

Do not stop there. The GAN-inversion literature's consistent finding is not
"encoder replaces optimizer" but that **hybrid wins**: the survey
([arXiv:2101.05278](https://arxiv.org/abs/2101.05278)) splits the field into
optimization (accurate, slow), encoder (fast, loses fidelity), and hybrid, with
HyperStyle- and PTI-style refinement beating pure encoders on fidelity while
staying far cheaper than pure optimization. Apply the same shape: run a handful
of gradient steps through the frozen graph starting from the encoder's output.
Because the init is already near-optimal these should take seconds rather than
minutes, giving a quality dial between "instant" and "optimizer-exact" instead
of one fixed operating point.

**Warm-start note (added after 2b's speaker-similarity correction below):**
the WavLM linear probe from 2b, despite failing on the perturbation direction,
lands at 0.432 calibrated ECAPA speaker-similarity against true style —
consistently ahead of a train-mean or random start (80/80 held-out samples).
That is exactly what a refinement loop's starting point needs to be good,
even though it is not good enough to be the answer on its own — use the 2b
probe (or a lightweight WavLM-features-in encoder trained the same way) as
2a's init before running the gradient steps below, and see the "What the
recovered signal is worth" note under the 2b correction for the numbers and
its limits.

Evaluate the encoder against **the optimizer's own converged style**, not only
against downstream audio quality. Audio metrics masked systematic encoder bias
in the early image-inversion work, and the same failure is available here. The
2b result below sharpens this twice, and neither time in the way first
recorded. First: the constraint is not an information ceiling in the audio.
ECAPA cannot see most of what a style direction does — it is trained to be
prosody-invariant, and most of what a random direction does is prosodic — so
feed this encoder something that carries prosody. Second, against the obvious
hope: WavLM, which does carry prosody, is twice as good as ECAPA at the full
target and still recovers exactly none of the perturbation direction. So judge
the encoder against the optimizer's converged style — that instruction matters
*more* after 2b, not less. No probe of this class recovered a style direction
from audio, and an audio metric chosen as badly as delta-ECAPA was would report
the same false negative as success.

**2b. Probe from an off-the-shelf speaker encoder.** Fit a map
`ECAPA/WavLM embedding -> style_ttl`. Run this explicitly as a **linear probe
first**, and report R^2 (per-row and overall), not just downstream audio
quality. The R^2 is the result:

A bare R^2 here would be meaningless. Two controls are mandatory:

- **Control-task selectivity** (Hewitt & Liang,
  [arXiv:1909.03368](https://arxiv.org/abs/1909.03368)): fit the identical
  probe against shuffled targets and report real-minus-control. The target is
  12,800-dimensional and VCTK offers ~110 speakers, so probe capacity alone can
  fit noise to a flattering number.
- **Speaker-disjoint splits.** Speaker-overlapping splits are documented to
  inflate these scores substantially. Enforce disjointness before reporting.

Report per-row R^2 as well as overall — a flat aggregate can hide that only a
few of the 50 rows are linearly predictable, which feeds directly into the
row-structure question in Phase 0. Phase 0's companion probe has since answered
that question — 24 of the 50 rows carry essentially all preset-to-preset
variation — so report R^2 over the active set separately from the aggregate;
the 26 near-constant rows are trivially predictable and will inflate a
whole-tensor number.

- **High linear R^2** -> the two spaces are related by an affine map, and
  directions established in ECAPA space transfer into style space by
  construction. **But this shortcut is gender-only.** Gender is near-perfectly
  linearly decodable from ECAPA/x-vector/WavLM (~99-100%), whereas age
  consistently requires nonlinear probing (best reported MAE ~4.8-5.3 years
  with MLP/ResNet heads). So even the optimistic branch buys the presentation
  axis and not the age axis — the one with no prior art. Plan for age to need
  native derivation in Phase 3 regardless of how the probe lands.
- **Low linear R^2, high MLP R^2** -> the spaces are related but nonlinearly.
  Usable as an encoder, but attribute directions will not transfer directly and
  must be derived natively in style space (phase 3).
- **Low R^2 either way** -> style space encodes something meaningfully
  different from speaker-verification space. Informative on its own, and a
  signal that the direct encoder (2a) is the only viable route.

Expect the middle branch. The closest published analog
([arXiv:2607.26742](https://arxiv.org/abs/2607.26742)) maps a foreign embedding
space into a frozen style-diffusion TTS latent, needs a nonlinear MLP adapter
to do it, and still reaches only ~0.40-0.42 cosine similarity to native style
prototypes. No published R^2 exists for ECAPA -> TTS-style-latent specifically,
so this is a genuinely open measurement with no baseline to check against.

Report the probe result before building on either branch. It changes how much
of phase 3 is needed.

### Result: negative for ECAPA and WavLM (2026-09-09); phase closed

Phase 2 as written is blocked on this machine — the optimizer is external, and
there is no corpus and no GPU — but 2b's question needed none of that. The
engine is its own paired-data generator: synthesizing from a known style yields
ground-truth `(audio, style)` pairs, which is the same argument this phase opens
with, run in the other direction.

Run with `py/phase2b_generate.py`, `phase2b_embed.py` and `phase2b_probe.py`,
with `phase2b_audio_sanity.py`, `phase2b_embedding_controls.py`,
`phase2b_direction_audibility.py`, `phase2b_perturbation_sweep.py`,
`phase2b_multi_utterance.py`, `phase2b_render_predictions.py` and
`phase2b_summarize.py` as controls and follow-ups. 1,920 pairs: six conditions
of 320 — preset blends, then perturbations at eps 0.05 / 0.10 / 0.20 / 0.40 /
0.80 — over 8 texts, 7 training presets with M5, F4 and F5 held out, per-sample
seeded rendering, `style_dp` fixed at M1's. The perturbation model is, per
active row, `normalize(P_r + eps*u_r)` with `u_r` uniform on the unit sphere in
R^256; the 26 inactive rows stay at the base preset. Row norms held throughout
(worst deviation 2.4e-07) and one clip of the 1,920 clipped. Outputs are under
the ignored `py/results/phase2b/`.

Family-disjoint R^2 (the three unseen voices), against the linear ceiling on the
same data, with the effective rank of the sampled style set:

| Condition | Family-disjoint R^2 | Linear ceiling | Effective rank |
|---|---|---|---|
| preset blends | 0.192 | 1.000 | 9.0 |
| eps 0.05 | 0.089 | 0.995 | 5.1 |
| eps 0.10 | 0.086 | 0.979 | 5.7 |
| eps 0.20 | 0.067 | 0.933 | 7.8 |
| eps 0.40 | 0.057 | 0.834 | 22.2 |
| eps 0.80 | 0.007 | 0.746 | 104.7 |

The ranks are censored by n=320 and so are lower bounds; the designed
dimensionality is 24 rows x 255 tangent dimensions = 6,120 continuous, plus the
discrete choice of base preset.

**The methodological trap was sprung deliberately.** Random-split R^2 runs 0.925
(preset blends) to 0.962 (eps 0.05), decaying to 0.109 at eps 0.80, and a
variance decomposition shows it tracking "which base preset was this" almost
exactly — that factor's share of target variance is 0.983 / 0.936 / 0.788 /
0.478 / 0.202 down the eps ladder. Sampling only preset blends and splitting
randomly would have reported R^2 ~0.93 and meant nothing. Measured instead
against the held-out set's own mean, every family-disjoint R^2 above is negative
(-0.18 to -5.4); the positive figures are variance explained relative to
predicting one fixed style, which is the fairer framing of the same result.

Controls:

- **Control task** (Hewitt & Liang): shuffled-target R^2 is -0.000 to -0.020, so
  selectivity equals the full number. The probe is not fitting noise — there is
  nothing there to fit.
- **Nonlinearity.** An RBF kernel probe was *worse* on unseen voices and showed
  real control-task leakage (-0.05 to -4.7).
- **Not domain mismatch.** ECAPA works fine on this audio: 100% 1-NN speaker ID
  across the ten presets against 10% chance, same-preset cosine 0.734 vs
  different-preset 0.186.
- **Not text nuisance.** Averaging ECAPA over 4 utterances moved family-disjoint
  R^2 only 0.067 to 0.078.
- **Not speaker-verification structure either.** A 120-feature MFCC-moment
  baseline matches ECAPA everywhere and beats it on preset blends (0.286 vs
  0.192). What little signal exists is coarse spectral statistics.
- **Trivial baselines** at eps 0.20: probe cosine 0.896, constant train mean
  0.887, 1-NN 0.847.
- **Within-family residual** (base preset removed — can it recover the
  perturbation direction alone): R^2 -0.013 to -0.031, and exactly +0.0000
  against the constant predictor, LOO-CV having selected the maximum ridge
  penalty and collapsed the probe to a constant. Honest caveat: that residual is
  isotropic in 6,120 dimensions, so a ridge fit here has a structural ceiling
  of min(d, n_train)/6120 — set by the *rank* of the fit, not by the width of
  the input. With 192 ECAPA dimensions and 240 training clips that is
  192/6120 = 3.1%. (The WavLM run below is why the rank form is the right one:
  2048 inputs do not buy 33%, they buy the same 240/6120 = 3.9%.) It is a weak
  test by construction; the full-target numbers are the load-bearing evidence.
- **Per row**, as this phase asks: all 24 active rows are negative under the
  family-disjoint split at every condition. Least bad at eps 0.20 are rows 6 and
  45 (-0.40), 23 (-0.48) and 22 (-0.54); worst is row 8 (-2.79). The
  whole-tensor number is uniformly *worse* than the active-row number rather
  than flattered by it — a held-out preset's 26 near-constant rows are unseen
  constants, not free wins.

**The mechanism matters more than the null.** The perturbation sweep found the
audio never stops being voice-like: voiced fraction, median F0, spectral
flatness, 2-8 Hz modulation and WER are flat across the whole eps ladder (WER
0.054-0.083 against 0.066 for the unperturbed presets), and a controlled ray
reached eps 3.20 — a 72-degree per-row rotation, twice the largest per-row angle
between M1 and F1 — with WER still 0.00 and F0 moved 5 Hz. Direction audibility
at matched per-row angle from M1, measured as delta-ECAPA: a random direction
gives 0.028 / 0.059 / 0.354 at eps 0.1 / 0.2 / 0.8, a direction inside the
preset-PCA subspace gives 0.113 / 0.207 / 0.685, and a direction toward another
preset gives 0.128 / 0.301 / 0.953 (a full M1->F1 swap is 0.814). Preset-aligned
directions move delta-ECAPA 3-5x further per unit of travel through style space
than random ones — recorded at the time as "3-5x more audible", which the
correction below overturns — and at eps 0.2 the style signal (0.059) is 4.5x
*smaller* than the text nuisance between two renders of the same style (0.266).

**Correction to the mechanism (2026-09-09): audibility was misread.** What
stood here read the anisotropy as an audibility ceiling — the audible subspace
is small, most of style space is close to inaudible, and the audio-to-style
inverse is ill-posed in nearly every other direction, so most of the target is
not in the audio at all. **That mechanism is wrong; the null it explained is
not.** Everything above stands, and nothing in this correction weakens it.

**A listener overturned it.** Played the M1 and F1 random rays, the user
reported the clips clearly differ, increasingly with magnitude, and that what
changes is which words are emphasised (M1) and per-word loudness (F1), with the
effect weaker on F1. Every measurement that said "flat" was an utterance-level
aggregate, which collapses precisely the time axis the effect lives on. The
follow-up is `py/phase2b_prosody_analysis.py`, `phase2b_prosody_mel.py` and
`phase2b_prosody_render.py`, reporting to the ignored
`py/results/phase2b_prosody/` (`prosody_report.json`, `mel_report.json`). It
confirms the listener on every point.

Frame alignment first, since frame-by-frame comparison is only valid if
durations are pinned: every ray clip is exactly 136,696 samples, re-rendering
reproduces the on-disk ray to 3.05e-05 (16-bit quantisation), and envelope
cross-correlation r(lag 0)/r(peak) stays >= 0.976 across M1's random ray
(>= 0.91 on F1's).

- **The random ray is loud and prosodic, not quiet.** From M1, eps 0.20 ->
  3.20: utterance level +0.4 -> +4.9 dB; per-word energy with that level change
  removed, sd 0.47 -> 2.88 dB and range 1.6 -> 10.1 dB. Median F0 moves only
  115 -> 119 Hz (+65 cents) and WER stays 0.00. The old battery saw the F0 and
  the WER and called the ladder flat.
- **It is far above audibility threshold.** At eps 0.80 — the top of the
  sampled training range — a random direction moves the log-mel spectrogram by
  8.4 dB rms, against 16.6 dB for a full M1->F1 identity swap.
- **The "3-5x more audible" figure was an artifact of the instrument.** It was
  measured with delta-ECAPA, a speaker-verification embedding trained to be
  prosody-invariant. On prosody-sensitive measures the same preset-aligned-over-
  random comparison gives roughly 1.1-3.2x — emphasis contour 1.1-1.8x, per-word
  emphasis 1.5-3.2x, per-frame spectral shape 1.1-1.8x, LTAS shape 1.5-2.4x —
  against delta-ECAPA's 3.5-6.5x.
- **The listener's M1/F1 asymmetry is real, and was recovered independently.**
  Both bases gain almost the same level (+4.89 vs +4.81 dB at eps 3.20). What
  differs is whether the emphasis *pattern* holds: M1's per-word profile
  correlation with its own eps-0.20 shape falls 1.00 -> 0.58 -> 0.56 -> 0.06
  across eps 0.20 / 0.80 / 1.60 / 3.20 — the ranking reorders, "dog" going from
  -0.7 to +6.5 dB relative — while F1's holds at 1.00 -> 0.90 -> 0.78 -> 0.67
  and saturates after eps 1.60, the same pattern scaled. Per-word range 1.6 ->
  10.1 dB on M1 against 2.4 -> 5.0 dB on F1.

**The two-subspace hypothesis: the strong form is not supported, a graded form
is.** For it: at matched per-row angle (eps 0.80) a random direction moves F0
register 20x less than a preset-aligned one (+40 vs +790 cents) and delta-ECAPA
3.5x less, but emphasis only 1.7x less; and the diff map is more time-localized
the more preset-aligned the direction (time-concentration at eps 0.20: 0.078
random, 0.170 toward F1, 0.370 full swap). Against it: the broadband-versus-
spectral-shape split of the diff is essentially the same across direction
families (broadband share 0.43-0.62, the lone outlier being F1's smallest
step), so there is no clean orthogonal
decomposition into an "identity" and a "prosody" subspace; and at matched
*total* acoustic change the localisation advantage disappears (random eps 3.20
at 0.223 against toward-F1 eps 0.20 at 0.170). Random directions do reach F0 and
timbre — just far along the ray. So: directions differ in what they move and in
how much travel that costs, not in whether they move anything.

The corrected mechanism, replacing the paragraph this section overturns:

> **Style space is anisotropic in what a direction changes, not in whether it
> changes anything.** At matched per-row angle from M1, a random direction moves
> median F0 by 40 cents where a preset-aligned one moves it 790, and moves
> delta-ECAPA 3.5-6.5x less — but it moves the log-mel spectrogram only ~1.7x
> less, lifts the utterance level 2.2 dB, and redistributes per-word energy over
> a 3.7 dB range at eps 0.80. A human listener hears these clips as clearly
> different, and describes the difference as changing word emphasis rather than
> changing speaker. The "3-5x more audible" figure was measured with
> delta-ECAPA, a speaker-verification embedding trained to be prosody-invariant;
> on prosody-sensitive measures the same comparison gives 1.1-3.2x. So the
> probe's null means **ECAPA cannot see most of what a style direction does**,
> not that most style directions do nothing. That still blocks the ECAPA
> shortcut and still points at the direct encoder (2a), but for a different
> reason: the target is audible and the *encoder input* was the wrong
> representation, so a prosody-bearing input (WavLM, or explicit prosodic
> features) is worth testing before concluding the inverse is ill-posed.

**The consequence that matters: there is no information ceiling on an encoder.**
The previous wording implied one — that no encoder could recover more than a
small audible quotient, because the rest was not in the audio. It is in the
audio. It was not in ECAPA.

**For emotion, the first axis in this plan, this is a positive result.**
`style_ttl` has demonstrable prosodic reach: random directions give monotone,
magnitude-scaled changes in utterance level and per-word emphasis while leaving
intelligibility (WER 0.00) and pitch register intact, and the emphasis effect
replicates on two further sentences (8.1 and 10.0 dB at eps 0.80). Emphasis and
loudness axes therefore look reachable without touching the duration tensor.
Two caveats travel with that: a random direction produces *unstructured*
emphasis jitter rather than a coherent emotional contour, so this shows the
lever exists and not that an axis exists; and `style_dp` was pinned throughout,
so speech rate and timing — a first-order emotion cue — lie outside everything
measured here.

Caveats on this correction, carried honestly: the ray is one seeded random
direction per base, so n=1 in direction space (though the four independent
random directions in `direction_audibility.json` agree on the ECAPA side); word
boundaries came from faster-whisper tiny.en with a hand-stated repair for one
collapsed boundary; M1's "dog" window sits at -26 dB in the base, so its large
dB deltas are partly a small-denominator effect; F0 contours above eps 1.60 are
unreliable (pyin octave errors), so only the medians are quoted there; and
`style_dp` was pinned for the whole analysis, so nothing here measures timing.

The general lesson is now in [CLAUDE.md](CLAUDE.md): compare spectrograms before
aggregates, and level-match before listening.

**The WavLM half (2026-09-09): twice as good, and the same answer.** Run with
`py/phase2b_wavlm_embed.py` and `py/phase2b_wavlm.py`. WavLM-large through
torchaudio's `WAVLM_LARGE` bundle over all 1,920 existing clips — no subsetting,
nothing re-synthesized, no audio rendered — reusing the same `styles.npz` and
manifest, at 0.55 s/clip for 18 minutes total, with all 24 layers mean+std
pooled and cached to the ignored `py/results/phase2b/wavlm_feats.npz` (377 MB).
It imports `phase2b_probe`'s functions directly, so splits, control task and
reporting are identical, and it refits ECAPA on the same indices for an exact
comparison.

Best representation: layer 3, mean+std pooled, 2048-dim. Family-disjoint R^2
against the train-mean baseline — the metric the table above quotes:

| Condition | ECAPA | WavLM L3 | ECAPA ceiling | WavLM ceiling | MFCC-moment |
|---|---|---|---|---|---|
| preset blends | 0.192 | 0.320 | 1.000 | 1.000 | 0.286 |
| eps 0.05 | 0.089 | 0.192 | 0.995 | 0.997 | 0.085 |
| eps 0.10 | 0.086 | 0.162 | 0.979 | 0.988 | 0.080 |
| eps 0.20 | 0.067 | 0.142 | 0.933 | 0.961 | 0.062 |
| eps 0.40 | 0.057 | 0.116 | 0.834 | 0.904 | 0.057 |
| eps 0.80 | 0.007 | 0.042 | 0.746 | 0.853 | 0.002 |
| pooled | 0.088 | 0.142 | 0.758 | 0.999 | 0.066 |

WavLM roughly doubles ECAPA in every condition, with selectivity intact —
shuffled-target R^2 is ~0.000 everywhere, so the raw score *is* the selectivity.
That is a real improvement and it is worth stating plainly before saying why it
does not reopen the phase.

- **The capacity hypothesis was tested and falsified.** The pooled row states it
  cleanest: raising the reachable ceiling from 0.758 to 0.999 — 32 points of
  fresh headroom — bought 5.4 points of actual R^2. The probe is
  information-limited, not ceiling-limited. Concatenating four layers to 8,192
  dimensions changed nothing (0.136 pooled against 0.142 for the single layer),
  and an RBF kernel ridge was *worse* than linear (0.105 against 0.142 at
  eps 0.20), so the middle branch is ruled out for WavLM as it was for ECAPA.
- **The ceiling arithmetic, corrected.** The earlier reasoning — including the
  framing in the residual control above — divided input width by target
  dimension. That is wrong. Ridge fitted on 240 training clips has rank at most
  240 whatever the input width, so the per-condition ratio is
  min(d, n_train)/6120 = 240/6120 = 3.9%, barely wider than ECAPA's
  192/6120 = 3.1%, not 2048/6120 = 33%. WavLM's dimensional advantage only
  becomes real in the pooled condition, where n_train = 1,440 — and that is
  precisely the condition where it converts least.
- **The decisive control is the within-family residual**, with the base preset
  subtracted so the target is the perturbation itself rather than a 7-way
  speaker ID. ECAPA gives -0.0000 / +0.0000 / +0.0003 and WavLM
  -0.0000 / +0.0000 / +0.0014 at eps 0.05 / 0.20 / 0.80, against a residual
  ceiling of ~0.82. Both collapse to a constant predictor, identically.
  **WavLM's entire gain is on "which base voice was this" and none of it is on
  the perturbation.** The variance decomposition agrees: 78.8% of eps 0.20's
  target variance is between-base-preset, the probe recovers a slice of that
  share and none of the remaining 21.2%.
- **Remaining controls.** Random-split inflation reproduces ECAPA's almost
  exactly (0.949 / 0.967 / 0.910 / 0.740 / 0.414 / 0.124), confirming that the
  inflation was "which preset was this" rather than an artifact of one
  representation. Raw-cosine 1-NN is meaningless on WavLM — every pair sits at
  cos ~0.98 off a shared offset — so a z-scored form was added, standardized on
  the training split like the ridge probe's scaler; it is negative in every
  condition except preset blends. Per-row R^2 is negative at all 24 active rows
  (eps 0.20: min -3.89, median -0.53, max -0.23), and whole-tensor R^2 is -0.73
  at eps 0.20.
- **WavLM works on this audio.** 1-NN preset ID over 10 presets x 8 sentences
  scores 0.86-0.95 for layers 1-5 against 10% chance; L3 is 0.887 where ECAPA is
  1.000. Slightly below ECAPA, as expected of a model not trained for speaker
  discrimination, and far above chance. Domain mismatch is not the explanation.
- **Layer sweep over all 24 layers: monotone decay from early to late.**
  Family-disjoint R^2 on preset blends / eps 0.05 / eps 0.20 — L1 .323/.141/.106,
  L3 .320/.192/.142, L4 .303/.184/.134, L5 .288/.188/.140, L12 .260/.125/.089,
  L18 .216/.128/.103, L24 .189/.086/.021. L3, L4 and L5 are indistinguishable at
  the top, which **independently corroborates `supertonic.embed`'s choice of
  layer 4**; taking the last layer would have cost more than half the signal.
- **Pooling.** mean+std beats mean-only slightly (0.192 against 0.167 at
  eps 0.05), while mean-only is the better speaker-ID feature (0.89 against
  0.81). Richer frame-level pooling was deliberately not built, for a reason
  worth recording: the 50 rows of `style_ttl` come from learned query vectors
  attending over the reference, so they have no time alignment for frames to be
  aligned to, and the only temporal variation in this corpus is the 8 fixed
  texts, which is nuisance. The std term *is* the frame-level probe, and it buys
  about 0.01.

**The synthesis (2026-09-09): audible does not mean identifiable — later
refuted, kept here because the correction only makes sense against it.** The
correction above established that random `style_ttl` perturbations are plainly
audible — they move the log-mel spectrogram, lift utterance level by up to
+4.9 dB, and redistribute per-word emphasis over 10 dB — and that ECAPA's
apparent "inaudibility" was an artifact of using a prosody-invariant embedding
as an audibility meter. It named a prosody-bearing input as the thing to test
"before concluding the inverse is ill-posed." That test had just been run, and
WavLM — which does carry prosody, where ECAPA is trained not to — was twice as
good at the full target and still returned *exactly zero* on the
within-family residual. Read together, the two results were taken to say that
a 6,120-dimensional perturbation collapses onto a low-dimensional audible
readout — roughly an utterance level plus a per-word emphasis pattern, on the
order of ten numbers for this sentence — so the map from that audible
consequence back to the direction that caused it is many-to-one, and not
invertible by a probe of this class: not "the audio does not contain the
change," but "many different high-dimensional directions produce nearly the
same low-dimensional audible consequence."

**This was flagged at the time as inference, not a third measurement, and the
direct test was named and left explicitly unrun:** whether distinct random
directions at matched magnitude produce *similar* audible readouts — energy
contour and per-word emphasis — from the same base. It has now been run, and
it refutes the synthesis above.

**Correction to the synthesis (2026-09-09): the collapse account is refuted;
the null survives for a different reason.** `py/phase2b_direction_collapse.py`
renders K=24 independent random directions from base M1 — mutually
near-orthogonal in style space, mean cosine -0.0006, max |0.040| — at the two
magnitudes already characterized above (eps 0.20 / 0.80, per-row angle
11.28 / 38.52 degrees), one sentence, `style_dp` pinned at M1's, seeded
rendering (row-norm deviation 2.4e-07, all clips frame-aligned at 136,696
samples), and asks the question the paragraph above named and left open: do
distinct directions produce *similar* audible readouts, or distinguishable
ones?

They are distinguishable, sharply and reproducibly:

- **Pairwise similarity between directions is low, and mostly inside the null
  band.** Per-word emphasis Pearson r across the 276 direction pairs: mean
  +0.158 / median +0.161 at eps 0.20, falling to +0.060 / +0.091 at eps 0.80 —
  against a null band, for two unrelated 9-word profiles, of r = 0 +/- 0.354,
  which **55% of all pairs fall inside**. Log-mel diff-map cosine: +0.331 /
  +0.404. Shared fraction, on a scale where 1.0 is identical and 1/24 = 0.042
  is unrelated: emphasis 0.205 / 0.090, log-mel diff map 0.380 / 0.437. Two
  directions typically move the log-mel spectrogram *from each other*
  (5.52 / 10.44 dB rms) by more than either moves it from the base
  (4.80 / 9.56 dB).
- **Direction identity survives an independent nuisance draw.** Each direction
  was re-rendered under a second vocoder latent (seed 5151) and matched 1-NN
  against its own latent's base, across all 24 directions (chance
  1/24 = 4.2%). The log-mel diff map picks the right direction 70.8% of the
  time at eps 0.20 (17/24, p = 9.0e-19) and 66.7% at eps 0.80 (16/24,
  p = 4.4e-17). The per-word emphasis profile does the same at eps 0.80
  (45.8%, p = 9.9e-10) but is weak at eps 0.20 (16.7%, p = 1.6e-02) — at that
  magnitude the 9-number profile sits near the vocoder-nuisance floor
  (same-direction cross-latent r only 0.386). The energy contour, consistent
  with being the *shared* part of the readout (below), does not identify
  direction at all (16.7% / 12.5%, the latter not significant).

**Specifically refuted:** "many different high-dimensional directions produce
nearly the same low-dimensional audible consequence," and the forward-looking
claim it licensed — that well-chosen audio metrics might agree between two
styles that are far apart in style space. Measured from a common base at
eps 0.20-0.80, well-chosen audio metrics **disagree**, sharply and
reproducibly, between distinct directions.

**The Phase 2b null survives — but the reason changes.** The readout is
genuinely low-dimensional: participation ratio 3.7 of 9 possible dimensions
for the emphasis profile, 4.6-7.0 of 23 for the log-mel diff map, measured the
same way across all 24 directions. Perfectly inverting a ~5-7 dimensional
readout recovers at most ~7/6,120 = 0.1% of a 6,120-dimensional target —
nowhere near enough to determine a direction by itself. The corrected
statement: **the inverse is narrow, not many-to-one.** Per utterance and per
base, the audio exposes on the order of 5-10 direction-specific dimensions out
of 6,120, and — this is what changed from the refuted account — they are
cleanly direction-specific rather than collapsed onto a readout shared enough
to erase which direction produced it.

**The grain of truth that survives from the refuted account.** The energy
contour genuinely is the shared part: pairwise r reaches +0.50 at eps 0.80,
and it is the one readout that fails cross-latent identification outright.
It is the per-word emphasis pattern and the spectral detail (the log-mel diff
map) that are direction-specific. "Utterance level plus a per-word emphasis
pattern" was the right list of readouts to name; "the same for every
direction" was the wrong claim to make about them.

**An admission, stated plainly because it changes how much weight the earlier
record deserves.** The within-family residual control (R^2 +0.0000 to
+0.0014 for both ECAPA and WavLM), cited above as the decisive evidence for
collapse, is exactly what *both* accounts predict — a genuinely low-dimensional
readout is just as invisible to a full-rank linear probe as a genuinely
collapsed one would be. **It never had the power to distinguish collapse from
narrow-but-real recoverability, so it cannot be cited as evidence for
collapse — nor is it evidence against recoverability.** Every other control in
this section stands; this one specific citation does not carry the weight it
was given.

**The materially different instruction for Phase 2a.** The limit measured
here is dimensional and **additive across utterances**, not a property of the
audio representation — a second sentence exposes a different 5-10 dimensions,
not a better view of the same ones. **The lever for an encoder is many
utterances per style, not a better single-utterance representation.** This
differs from what stood here before, which implied representation quality was
the binding constraint: feeding a richer per-utterance feature does not relax
a limit set by how many independent readouts exist, not by how well any one of
them is measured.

**Caveats.** One base and one sentence, so the dimensionality figures (3.7 /
4.6-7.0) are per-(base, sentence) — a different base or a longer sentence
could sit at a different point. At eps 0.20 the emphasis profile sits near the
vocoder-nuisance floor, so identification there is weak by construction; the
log-mel diff map is the strong instrument and is unambiguous at both
magnitudes.

**Also measured: what the probes actually reconstructed.**
`py/phase2b_wavlm_render.py` (report `probe_recovery_triples.json`) refit both
probes family-disjoint on the identical indices used above — reproducing the
record exactly, ECAPA R^2 +0.0672 and WavLM L3 +0.1421 at eps 0.20 — and added
a third, deciding arm: a constant train-mean predictor that reads no audio at
all. Active-row cosine to the true style and level-matched log-mel distance:
idx1200 (F5) ECAPA 0.8747 / 15.88 dB, WavLM 0.9000 / 15.68, train-mean
0.8731 / 19.19; idx1201 (F5) ECAPA 0.8842 / 18.58, WavLM 0.9012 / 14.08,
train-mean 0.8730 / 19.71; idx1202 (M5) ECAPA 0.9021 / 21.47, WavLM
0.9012 / 19.01, train-mean 0.8871 / 22.32.

For scale, a full M1->F1 identity swap was quoted above as 16.6 dB rms
log-mel, from which the record concluded **both probes' predictions sit
14-21 dB from the true style — as far as, or further than, a different
speaker.**

**Correction (`py/phase2b_speaker_similarity.py`, `py/phase2b_gender_confound.py`,
2026-09-09): that claim is false and is withdrawn — the 16.6 dB figure was
never a valid identity threshold.** It was calibrated from a single pair.
Re-derived from many pairs — all 80 held-out test samples, condition
eps0.20, same ridge probes and protocol, reproducing ECAPA R^2 +0.067 and
WavLM R^2 +0.142 exactly — frame-aligned log-mel rms distance for
same-speaker pairs at a different vocoder seed averages **17.42 dB** (10
pairs, range 10.5-24.6), and for different-speaker pairs across all 45
preset combinations averages **17.82 dB** (range 11.2-23.9). The two
distributions are statistically indistinguishable (0.4 dB apart against a
same-speaker sd of 3.9, full range overlap), and M1-F1 itself reproduces at
17.49 dB, consistent with the 16.6 dB quoted above. **Same-speaker pairs
routinely exceed 16.6 dB from vocoder-seed noise alone, so log-mel distance
at this scale cannot separate same-speaker from different-speaker pairs and
never supported an identity claim.** Active-row style cosine fares no
better: within a single predictor type — the fair test, correlating with
true ECAPA speaker similarity sample-by-sample rather than pooling across
predictor types — it correlates -0.02 (ECAPA probe) / +0.40 (WavLM) / +0.55
(train-mean) with identity, and log-mel distance -0.33 / -0.13 / -0.15.
Pooling all three predictors together inflates this to r ~+0.67 / -0.29, but
that is between-group separation (which predictor produced this sample) —
the same random-split trap this record warns about elsewhere — not a
within-condition correlation, and not evidence either metric tracks
identity.

Calibrated instead on ECAPA cosine between rendered clips — the metric ECAPA
is actually built to measure — same-speaker/different-seed pairs anchor at
0.879 (0.784-0.927), same-speaker/different-sentence at 0.759
(0.669-0.826), different-speaker across the same 45 preset pairs at 0.225
(-0.012-0.569). WavLM's predictions average **0.432** (0.288-0.656) against
the true style — inside the different-speaker range, with only its single
best sample (F4, 0.656) approaching the same-speaker floor of 0.669 — and
never reach same-speaker territory; ECAPA-pred averages 0.383, train-mean
0.240. So on this calibration the reconstructions still fall short of true
identity, as the withdrawn framing claimed by a different, invalid route.
But **the recovered signal is real and consistent, not the near-zero margin
the withdrawn active-row-cosine comparison (WavLM trailed train-mean by only
0.017-0.028) implied**: WavLM beats the train-mean baseline on **80/80
samples** (mean margin +0.19 ECAPA cosine) and beats a same-gender
wrong-speaker impostor on **95%** of samples (100% on M5), margin +0.22. A
gender confound explains part but not all of this: train-mean is decisively
male-leaning (median F0 920-980 cents against a male-preset mean of 998 and
female mean of 1934; its four nearest presets by ECAPA cosine are M4, M3,
M1, M2), so beating it on female-true samples could be gender alone — but on
**M5**, where baseline and impostor are both male and gender explains
nothing, WavLM still beats the same-gender impostor on 29/29 samples with
margin +0.220, *larger* than its +0.145 margin over train-mean there. Per
identity: WavLM 0.483 (F4) / 0.449 (F5) / 0.374 (M5), matching an
independent listener's ranking of the same three reconstructions (F5 clearly
right, M5 only close).

**This does not reopen the closure below — it corrects how far the
reconstructions sit from the true style, not what they recover it from.**
WavLM's gain is on base-voice identity, exactly what the record already
predicts probes would recover: 78.8% of eps 0.20's target variance is
between-base-preset (the within-family residual control below), and this
measurement is squarely inside that slice. The within-family residual
itself — the perturbation with base voice subtracted, the load-bearing claim
for the closure — was not retested here and remains at R^2 ~0.

**What the recovered signal is worth, stated positively.** The framing above
is deliberately about what WavLM's gain is *not* — read alone it can sound
like the probe recovers nothing usable. It recovers something real: a
consistent, ordered voice-identity signal, not just a mean shift. WavLM beats
the train-mean baseline on **80/80** held-out samples (mean margin +0.19
ECAPA cosine) and a same-gender wrong-speaker impostor on **95%** (100% on
M5, margin +0.22), and its per-identity means — F4 0.483, F5 0.449, M5
0.374 — independently match a listener's ranking of the same three
reconstructions. That has a concrete use in 2a: **as a warm start for the
refinement loop, not as a replacement for it.** 2a is written as a direct
encoder *plus* gradient refinement — the GAN-inversion hybrid pattern above,
run a handful of steps through the frozen graph from the encoder's output —
and the entire point of Phase 2 is amortizing away the per-speaker optimizer
run that costs minutes of GPU time per speaker today (Phase 2's framing
above; a comparable single-speaker run takes 15-30 minutes on a T4 per
`docs/EMOTION_CALIBRATION.md`). A predictor that lands at 0.432 calibrated
ECAPA cosine — roughly double the different-speaker anchor (0.225) — and
beats a no-audio baseline on every held-out sample is a materially better
place to start that descent than a random init or the training mean: fewer
steps to converge from a warm start is exactly the currency 2a is trying to
buy. It is also evidence, independent of the negative result below, that
`audio -> style_ttl` has learnable structure at all — the failure is in how
much a linear probe recovers from one utterance, not that the mapping is
unlearnable.

Keep the limit attached to it, so it is not over-read. 0.432 sits below the
same-speaker floor (0.669), so this is a starting point, not an answer. The
within-family residual — the axis the perturbation itself lives on, and the
closure's load-bearing claim — is still ~0, so this recovery contributes
nothing to the fine-grained perturbation direction; it is confined to the
between-preset slice of the target. And every clip measured here is
engine-generated audio from engine-generated styles, so whether a warm start
this good transfers to real recorded voices is untested. It connects to the
many-utterances lever recorded below on the encoder side too: if one
utterance yields roughly 5-10 usable dimensions and the limit is additive
across utterances, a warm start built by averaging (or jointly fitting) a
predictor over several utterances of the same speaker should beat a
single-utterance warm start — a cheap, concrete thing for 2a to try early,
before or alongside the refinement loop.

**Verdict: the optimistic branch is dead and the middle branch with it — the
kernel probe was worse on unseen voices for both inputs, with real control-task
leakage on ECAPA. This lands on the third branch: style space encodes something
meaningfully different from what an off-the-shelf audio encoder reads off the
waveform, and the direct encoder (2a) is the viable route. Both halves are now
in — ECAPA, then WavLM at roughly twice the R^2 with clean selectivity and
still exactly zero on the perturbation — so Phase 2b is closed.**

Two consequences worth carrying forward, one each way. It shapes 2a, and more
sharply now that both inputs are in, the direction-collapse test is run, and
the probe-reconstruction check is run alongside it. There is no information
ceiling in the audio — the change is audible, and a prosody-bearing input does
read more of it, so feed 2a WavLM-class features (layer 3-5, mean+std) rather
than a speaker-verification embedding. But the perturbation *direction* was
not recoverable by any linear probe of this class from either input, so what
remains unbounded by 2b is the encoder, not the inverse problem as a whole: 2a
is worth running, and its evaluation against the optimizer's converged style
rather than against downstream audio matters *more* after this, not less —
now for a corrected reason. It is not, as first thought, that well-chosen
audio metrics might falsely agree between two styles that are far apart in
style space; the direction-collapse test above refutes that directly —
distinct directions disagree in the audio, sharply. It is that the audio
readout is narrow, on the order of 5-10 dimensions out of 6,120 per utterance,
so an encoder can match the audio closely — exactly what the probe-
reconstruction check shows: both probes' predictions still sit inside the
different-speaker range on calibrated ECAPA cosine despite genuine R^2 (0.432
WavLM / 0.383 ECAPA against a same-speaker floor of 0.669 — see the
correction above), while leaving most of the target unconstrained.
Style-space evaluation catches that; audio proximity alone does not. **And the many-utterances lever applies to 2a's training objective as
much as it does to 2b's readout: the limit is additive across utterances, so a
single-utterance encoder objective is the wrong unit to optimize — train and
evaluate 2a against multiple renders per style, not one.** It also helps
Phases 3 and 4: *within* the preset-spanned subspace, embedding distance does
track style distance (Spearman 0.41, against 0.00 for random directions at
every magnitude), so deriving axes there is far better conditioned than the
6,144-parameter framing suggests.

Still open:

- ~~Whether the collapse account is right.~~ **Resolved (2026-09-09): no.**
  See "Correction to the synthesis" above — `phase2b_direction_collapse.py` ran
  the direct test this bullet named and refutes the collapse account; the
  Phase 2b null survives, but for the narrow-inverse reason recorded there,
  not the many-to-one one.
- **Where the sampled styles live.** Every style here was engine-generated from
  a base preset, so the sampled set may not reach where real speakers do. That
  bounds the answer rather than closing it — and closing it needs
  optimizer-extracted real-speaker styles, which is the dependency Phase 2
  exists to remove.
- **Three held-out identities** (M5, F4, F5), one language, synthetic audio
  throughout.

## Phase 3: Deriving the Axes

### Are age and vocal presentation root attributes?

Almost certainly not. They are derived and heavily **entangled** — but the
correlates are not equally weighted, and an earlier draft of this plan had them
wrong.

**Primary, well-supported:** speech rate and articulation timing. Harnsberger
et al. report a large effect of speech rate on perceived age (partial eta^2 =
0.47), with the fast/slow contrast *larger* for older speakers — the regime
where the axis matters most. Formant-frequency shifts with age are real but
vowel-specific and modest.

**Secondary and contested — do not treat as co-equal pillars:**

- **F0 mean.** Showed no age-estimation effect in the study above once speech
  rate was in the design. Age-related F0 decline is largely a female-specific
  post-menopausal phenomenon; male F0 barely differs across age groups.
- **Jitter and shimmer.** Explicitly reported as "not a robust cue"; increases
  reported only for males, only from middle age.
- **Breathiness / H1-H2.** Findings *contradict each other* across studies —
  one finds elderly women less breathy than young women, another the reverse,
  with only moderate correlation to perceived breathiness in either case.

The entanglement is quantified and unavoidable: F0 mean alone explains ~43.5%
of variance in masculinity ratings and ~24% in femininity ratings, while also
being one of the few reliable age cues in women. Lowering F0 to age a female
voice **will** read as more masculine. This is measured, not hypothetical.

So do not assume a root-attribute set. Derive it:

1. Extract styles for N labeled speakers using the phase 2 encoder.
2. Estimate the delta with a **covariate-controlled group-difference
   estimator** — regress the style tensor on the target attribute with
   nuisance covariates (sex, accent/L1, corpus, recording condition)
   included, rather than taking a raw difference of means — or use matched
   sampling across groups where the corpus allows it. Report group balance
   on those covariates so a reader can see what the estimate is confounded
   with.

   The reason this is not the emotion formula: age and presentation deltas
   are **between-speaker, not within-speaker**. Emotion's formula works
   because RAVDESS/CREMA-D/ESD record the same actor performing neutral and
   angry, so `Δ = s_angry - s_neutral` cancels speaker identity exactly —
   every confound that is a property of the person (vocal tract, accent,
   mic, session) subtracts out. Nobody is 25 and 70 in one session, and
   longitudinal corpora at useful scale do not exist, so an age or
   presentation delta is a difference of group means over different
   people — sex distribution, accent, recording era, room, and corpus
   provenance ride along inside it, and no subtraction removes them. This
   is a harder estimator than emotion's, not merely an untried one, and a
   better explanation for age control's near-absence from the literature
   than "nobody tried." Emotion keeps its existing within-speaker formula;
   it remains the easier case.
3. Find directions two ways and compare: unsupervised (PCA over the style
   space, then correlate components against measured acoustics) and supervised
   (LDA or linear probe per labeled attribute).
4. **Orthogonalize the resulting axes against each other** (Gram-Schmidt) —
   as a starting point, not a guaranteed fix. Orthogonalization does transfer
   to speech: [arXiv:2402.12423](https://arxiv.org/abs/2402.12423) shows no
   gender editing occurs when interpolating along the orthogonal component, so
   this is not merely an image-domain analogy. **But no source was found
   demonstrating that linear orthogonalization alone cleanly separates age
   from gender in a real voice system** — the disentanglement literature
   reaches for adversarial or mutual-information penalties instead. Treat
   Gram-Schmidt as the cheap first attempt, verify it with the Phase 5
   monotonicity probe rather than assuming it worked, and budget for an
   adversarial or MI-penalty fallback if residual gender drift survives under
   the age axis.

Both estimators can be restricted to the 24 active rows measured in Phase 0
(6,144 parameters rather than 12,800), holding the rest at the base voice. That
does not change the plan above; it changes what it is fitted over. Phase 2b
narrows it further, and favourably: inside the ~9 dimensions the shipped
presets span, embedding distance tracks style distance (Spearman 0.41, against
0.00 for random directions). Deriving axes there is much better conditioned
than 6,144 free parameters suggests. A direction found outside that subspace is
not inaudible — the 2026-09-09 correction below shows random directions are
plainly audible — but what it moves is emphasis and level rather than identity
or pitch register, so an axis meant to move identity should be sought inside
the preset span and one meant to move prosody need not be. Note also
that the probe found gender spread across roughly twenty rows rather than
concentrated in one component, so the Eigenvoice single-component precedent
should not be read as a prediction that a presentation axis will be similarly
compact here.

### The duration tensor cannot stay neutral

The emotion work leaves `style_dp` untouched by default (`include_duration` is
opt-in), and that was a reasonable conservative choice for emotion. It will not
survive contact with age.

Speaking rate is not merely among the age cues — at partial eta^2 = 0.47 it is
the largest single manipulated effect in the perceived-age literature, and it
lives in DP, not TTL. An age axis that moves
only TTL will produce a voice whose timbre says "older" while its timing says
"younger" — the two cues fight, and the result reads as uncanny rather than
old. The same applies, more weakly, to arousal-linked emotion.

So: derive DP deltas jointly with TTL for every axis, not as an afterthought.
DP is only 128 numbers against TTL's 12,800, so it is cheap to fit but easy to
overfit — expect it to need heavier regularization, and evaluate it separately
before combining. Its 8 rows are unit-normalized like TTL's, so DP deltas need
the same row projection after blending.

Note this is also where the plan departs from all surveyed prior art: every
latent-editing TTS system found edits a *static* embedding and leaves rate
alone. Moving DP has no direct precedent either supporting or refuting it, so
it carries genuine research risk — and, given the effect size above, genuine
upside.

Treat `include_duration=False` as a debugging switch, not a default.

### Bias the basis toward measurable acoustics

Where a choice exists, prefer axes anchored to quantities that can be measured
from output audio: F0, formant dispersion, breathiness, speaking rate. That
buys a closed loop:

```text
set parameter -> synthesize -> measure (parselmouth/praat) -> assert the
parameter moved what it claimed, monotonically
```

This converts the project from "listen and hope" into something with
regression tests. It is the highest-leverage single decision in this plan.

**Validate the measurement loop before trusting it.** LPC-based formant and VTL
estimation — what Praat and parselmouth do by default — is biased at high F0,
which is precisely the regime of women's and children's voices: the populations
the age and presentation axes most need. Check the loop against known-VTL
references first, and either tune LPC order per speaker or use a non-LPC VTL
estimator at the high-pitched end. A measurement harness that is wrong exactly
where the axis is hardest would produce confident nonsense.

## Phase 4: Datasets

Licensing is not a selection criterion — this work is going out open-source.
Select purely on speaker count, metadata quality, and recording conditions.

**The single biggest correction to this plan came from checking VCTK.** Its
`speaker-info.txt` gives ages spanning only **18-38**, and the distribution is
worse than that range implies: the bulk sits at 21-24 (26 speakers aged 22, 29
aged 23), with a handful above 26 and single speakers at 27, 32, 33 and 38.
VCTK is effectively an early-twenties corpus. An earlier draft made it the
first-pass corpus for deriving axes; **for the age axis that is impossible**,
and no amount of clean studio audio fixes it.

| Dataset | Speakers | Age metadata | Role |
|---|---|---|---|
| **VCTK** | ~107 | real ages, but **18-38, bulk 21-24** | Presentation axis and identity-space structure only. Studio-clean and well-labeled, but carries no age signal to learn from. |
| **CREMA-D** | 91 | **real ages 20-74** | Promoted to first-pass for age — the widest genuine age range in the core set, and it keeps emotion and demographics in one frame. Acted, which is a caveat for naturalness. |
| **SeniorTalk** | 202 | real ages, super-aged | Fills the elderly end no other listed corpus reaches. Conversational rather than studio. |
| **CSLU Kids** / **ChildMandarin** | 1,100+ (CSLU) | grade/age-based | Fills the child end, absent from every corpus above. |
| **Common Voice** | very large | self-reported **brackets**, optional and unverified | Scale and multilingual reach, not the primary age source. Highly variable recording conditions. |
| **Speech Accent Archive** | ~2,500+ | real age, 200+ L1s | Real ages with a consistent mic protocol; single fixed passage limits prosodic variety. |
| **LibriTTS-R** | ~2,456 | none | Presentation axis and identity-space PCA only. |
| **RAVDESS** | 24 | acted emotion | Already extracted. Keep for emotion continuity; too few and too acted for demographic axes. |
| **ESD** | 20 | none | Extending emotion multilingually, later. |

Practical notes:

- **Revised sequencing.** Start with VCTK for the presentation axis, where it
  is genuinely the best available, and with CREMA-D for age. Add SeniorTalk and
  a child corpus before claiming the age axis generalizes — an axis fitted on
  20-74 will extrapolate badly at exactly the ends users reach for first.
- **Channel leakage is the main data risk**, and it is well documented that
  speaker embeddings carry recoverable SNR and recording-condition information.
  Common Voice is the worst offender. Mitigate by SNR filtering and by
  averaging many utterances per speaker so channel cancels while identity
  persists. Verify by probing whether a top PCA component correlates with
  recording SNR rather than with any speaker attribute — and be prepared for
  the answer to be yes, in which case adversarial SNR-invariance is the
  standard next step.
- **Age will be perceived age.** Real ages exist in CREMA-D, VCTK and the
  Accent Archive, but coverage is uneven and Common Voice offers only
  unverified self-reported brackets. State this in the API and docs so the
  parameter is not over-promised.

## Phase 5: Evaluation

The emotion work had a listening matrix and nothing else. Attribute axes need
an objective story before defaults are chosen.

For each axis:

- **Monotonicity probe.** Train a classifier or regressor on *real* speech
  (age bracket, gender), run it across the axis on *synthesized* speech, and
  confirm the prediction moves monotonically with the parameter. A
  non-monotonic axis is not a control.
- **Acoustic assertion.** The parselmouth loop from phase 3 — did F0, formant
  dispersion, and rate move as the axis claims?
- **Identity policy, decided per axis.** For emotion, speaker similarity should
  stay high. For age and presentation, changing perceived identity is partly
  the point. Decide and document per axis rather than applying one threshold.
- **Pairwise and triple leakage.** VoiceShop
  ([arXiv:2404.06674](https://arxiv.org/abs/2404.06674)) evaluates composed
  edits by asking, for every attribute pair and triple, whether editing X
  shifts perceived Y. Adopt that protocol — in the one paper that claims
  success at multi-attribute editing, cross-attribute *leakage* is the reported
  failure mode, not signal quality. This is the real test of whether Phase 3's
  orthogonalization held.
- **Intelligibility at the extremes.** WER via Whisper at the ends of each
  range and beyond.

  An earlier draft assumed intelligibility breaks first and therefore sets the
  safe range. That is probably backwards, and the reason is pointed enough to
  state carefully.

  [arXiv:2402.12423](https://arxiv.org/abs/2402.12423) (ACL 2024) compares two
  ways of steering a frozen diffusion TTS model. Its proposed **h-space latent
  editing** is monotonic: "more samples classified as female as lambda
  increases." Its **speaker-embedding editing** baseline is not — "when lambda
  >= 3 even originally female voices are not classified as such."

  **The failing method is the one this plan uses.** Editing `style_ttl` is
  conditioning-embedding editing, not h-space editing. The published
  non-monotonic reversal is the documented failure mode of our family of
  approach, and the method shown to avoid it operates on an internal denoiser
  activation that a frozen ONNX graph does not expose. So this is not a
  cautionary note borrowed from a neighbouring technique — it is the closest
  published result to what we are building, and it is negative.

  Two consequences. First, monotonicity failure and WER failure are separately
  located thresholds and the usable range is the *minimum* of the two; locate
  both independently and never use WER as a proxy. Second, treat
  non-monotonicity as an **expected** outcome to measure for, not a tail risk.

  Phase 2b adds a third: WER is not merely a poor proxy, it is close to
  insensitive. Random style perturbations out to a 72-degree per-row rotation —
  twice the M1->F1 angle — left WER at 0.00 and F0 within 5 Hz. Expect the
  limit on a range to be audibility or monotonicity, not intelligibility.

  One reason for cautious optimism: the paper's baseline edits a single
  speaker vector, whereas `style_ttl` is a 50 x 256 set reused as attention
  keys throughout the vector-field estimator (confirmed at
  `py/helper.py:228-247` — it is re-consumed on every sampling step). That is a
  materially richer conditioning surface than the baseline it failed on. But
  that is a hypothesis, not a result, and Phase 0 is where it first gets
  tested.
- **Cross-language transfer.** Supertonic covers 31 languages, but axes derived
  from English VCTK may not hold elsewhere — speaking-rate norms and F0 ranges
  differ by language, and DP deltas especially may not port. Check at least one
  non-English language (Korean or Japanese, where upstream evaluation is
  strongest) before claiming an axis is general. If axes turn out to be
  language-specific, that is a scoping result worth having early rather than a
  failure.
- **Clipping and loudness.** Carry over the outstanding checks from the emotion
  roadmap — peak, RMS, and finite-audio assertions on every generated sample.

Choose default ranges from these measurements. Do not assume the extremes of
the observed data are usable.

## Relationship to the Emotion Roadmap

This is not two competing projects.
[docs/EMOTION_ROADMAP.md](docs/EMOTION_ROADMAP.md) and this plan are one
project in which **emotion is the first axis, not a separate track**. The
roadmap's own Phase 6 — a continuous arousal/valence/dominance space with
named emotions as presets within it — is this plan applied to emotion. That
is where the two documents converge.

Emotion goes first for a specific reason, not by default. **It is the only
axis whose estimator is well-posed** — the within-speaker point above — and
it carries the strongest prior art of the three. That makes it the right
place to validate machinery every axis shares, and the cheapest place to
discover a failure: the assets, listening sets, and extraction pipeline
already exist. If the additive-delta approach dies — Phase 0 fails, or the
non-monotonicity that [arXiv:2402.12423](https://arxiv.org/abs/2402.12423)
documents for embedding-space editing shows up — this is where it should
die, before two harder, unproven axes are built on top of it.

But the roadmap's existing *order* is wrong on three items, each obsolete or
harmful to do first:

| Roadmap item | Problem |
|---|---|
| Phase 1: add `gain` to `with_emotion()` | Discarded by the `with_deltas([...])` refactor in this plan's Phase 1c. Building the single-axis API first means writing it twice. |
| Phase 2: extract 5-20 speakers by optimization | This is exactly the GPU bottleneck the amortized encoder (this plan's Phase 2) removes. Doing it by hand first spends days to avoid building the thing that makes it minutes. |
| Phase 3: evaluate and select strengths | Runs on top of the `intensity=0.0` normalization bug from this plan's Phase 1a. Every "neutral" reference may be a different voice than the base, and the measurements would need redoing after the fix. |

Roadmap Phase 4 (inline tags, segment synthesis, crossfades) is product
plumbing orthogonal to every research question here, and belongs last
regardless of which plan wins.

Emotion carried end-to-end through evaluation is also the deliverable that
justifies the work externally: **"calibrated multi-speaker emotion control
with measured quality" is a shippable, self-contained result**, and the
README already sells emotion as this fork's differentiator.

## Sequencing

1. **Shared foundation** — Phase 0 (linearity gate) plus Phase 1 (composition
   fixes: `intensity=0.0` no-op, accumulate-then-normalize-once, the
   `with_deltas()` refactor). Blocks everything below it, costs about an
   hour of listening plus a small refactor, and is owed to the emotion work
   regardless — the normalization bug is corrupting emotion results right
   now.
2. **Amortized encoder** (Phase 2) — unblocks multi-speaker extraction for
   emotion and demographics at once. Turns the roadmap's Phase 2 from a GPU
   grind into a batch job.
3. **Emotion end-to-end, through evaluation** — multi-speaker deltas, the
   monotonicity probe, WER, clipping, the leakage protocol. Build the
   parselmouth measurement loop here, validated against known references
   first: every later axis is judged by it, and emotion is where it is
   cheapest to calibrate. Validates the shared machinery on the one
   well-posed axis before either demographic axis touches it.
4. **Presentation axis** — the first between-speaker axis, but the easiest
   one: gender is ~99-100% linearly decodable from ECAPA, so the Phase 2b
   probe shortcut actually applies here. With the caveat that the probe itself
   came back negative for ECAPA and for WavLM, so presentation must be derived
   in style space rather than read across from an embedding direction; what
   survives is
   that presentation lies inside the preset-spanned subspace, where 2b found
   the conditioning is good. Phase 0's probe adds that in style
   space the M1->F1 difference is spread over roughly twenty rows, not one
   component — decodable, but not a single knob to find.
5. **Age axis** — last, deliberately. No prior art, no within-speaker
   pairing, a hard dependency on moving `style_dp`, and corpora not yet
   acquired (SeniorTalk, a child corpus).

Layer 1 is a prerequisite for everything that follows it. Layer 2's probe result
decides how much native derivation layers 4 and 5 each need. Build the
parselmouth measurement loop before scaling age data collection to
SeniorTalk and a child corpus (layer 5), or there will be no way to tell
whether more data helped.

## Open Questions, Answered

**What would a good base voice be?** Do not pick one. Speaker-normalized deltas
exist precisely so directions are speaker-independent. If a base is needed for
demos, use the **centroid of the preset voices' TTL** rather than M1 — it
minimizes extrapolation distance to any attribute target. Confirm the centroid
synthesizes cleanly; that is the same convexity question as phase 0.

**Are perceived age and vocal presentation true attributes?** No — see phase 3.
Derived, entangled, and best obtained empirically with explicit
orthogonalization rather than assumed.

**Which datasets?** See phase 4, and note the correction: VCTK cannot carry the
age axis (ages 18-38, bulk 21-24). CREMA-D leads for age, VCTK for
presentation, SeniorTalk and a child corpus for the extremes.

**Could the original embedder be recovered via vec2vec, and would it be
useful?** The capability is very useful — it is the phase 2 bottleneck-breaker.
vec2vec is the wrong tool for obtaining it, because paired data is freely
generatable and supervised regression is strictly easier than unpaired
translation. See phase 2.

## Key References

Load-bearing sources, with what each is relied on for. Claims marked (v) were
verified against the primary source; the rest come from a literature sweep and
should be re-checked before anything depends on them.

- **[arXiv:2402.12423](https://arxiv.org/abs/2402.12423)** — closest published
  analog. Additive edits on a frozen diffusion TTS model; PCA and
  mean-difference direction discovery; and (v) the finding that
  speaker-embedding editing goes non-monotonic at lambda >= 3 while h-space
  editing stays monotonic. The negative half applies to us.
- **[arXiv:2503.23108](https://arxiv.org/abs/2503.23108)** — SupertonicTTS
  architecture. (v) 50 learnable vectors in the reference encoder, reused as
  keys in the VF estimator. Predecessor model: 44M params, English-only,
  vector width 128 vs our 256.
- **[arXiv:2404.06674](https://arxiv.org/abs/2404.06674)** — VoiceShop.
  Multi-attribute composition at three axes; pairwise/triple leakage protocol
  adopted in Phase 5.
- **[arXiv:2310.17502](https://arxiv.org/abs/2310.17502)** — principal
  directions over artificial speaker embeddings; the `z' = z + Ux` precedent.
- **[arXiv:2507.03377](https://arxiv.org/abs/2507.03377)** — eigenvoice model
  editing; gender loading on a single SVD component, the case for localized
  structure. Did not replicate here: gender is localized to row groups but
  spread across roughly twenty of them.
- **[arXiv:1909.03368](https://arxiv.org/abs/1909.03368)** — Hewitt & Liang,
  control tasks and probe selectivity. Phase 2b's methodology.
- **[arXiv:2101.05278](https://arxiv.org/abs/2101.05278)** — GAN inversion
  survey; the optimization/encoder/hybrid taxonomy behind Phase 2a.
- **Harnsberger et al.** (PMC4505082) — speech rate as the dominant perceived-
  age cue (partial eta^2 = 0.47); the evidence base for the DP argument.
- **VCTK `speaker-info.txt`** — (v) ages 18-38, bulk 21-24.
- **`kdrkdrkdr/supertonic.embed`** — unit-sphere row projection for both
  tensors; monolithic optimization of style_ttl.

## Success Criteria

This project has succeeded when:

1. At least two orthogonalized axes (age, presentation) move their
   corresponding classifier prediction monotonically over a documented range.
2. Composing three axes in any order produces identical audio (the
   commutativity fix in phase 1 holds under test).
3. WER at the documented range endpoints is within a stated margin of the base
   voice.
4. A new voice can be specified as a parameter vector, reproduced exactly from
   that vector, and synthesized without any per-voice asset file.

It has failed, and should be re-scoped rather than pushed, if phase 0
interpolation is artifacty or if no axis in phase 3 achieves monotonicity after
orthogonalization.
