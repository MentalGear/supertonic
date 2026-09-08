# Parametric Voice Space — Research Plan

Status: proposal. Supersedes the `new-plan` sketch.

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

### Companion experiment: is the 50x256 grid structured?

Nothing yet explains why TTL has 50 rows, or what they individually carry. The
per-row normalization implies they are meant to be independent unit vectors,
which suggests they may specialize.

Cheap probe: take voices A and B, and synthesize 50 hybrids where hybrid `i`
uses A's TTL with row `i` replaced by B's. Measure and listen.

- **Diffuse** (every row shifts the voice slightly) -> axes must span all rows;
  deltas stay dense. Proceed as planned.
- **Localized** (a handful of rows dominate identity, others prosody or
  timbre) -> derive axes per row group instead. Much better conditioned, far
  fewer parameters to fit, and it makes "change age without changing identity"
  a structural property rather than something orthogonalization has to
  enforce after the fact.

This is a few hours of work and it could substantially simplify phase 3.

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

**2a. Direct encoder.** Train `audio -> style_ttl` on optimizer-generated
pairs. Turns per-speaker extraction into a single forward pass.

**2b. Probe from an off-the-shelf speaker encoder.** Fit a map
`ECAPA/WavLM embedding -> style_ttl`. Run this explicitly as a **linear probe
first**, and report R^2 (per-row and overall), not just downstream audio
quality. The R^2 is the result:

- **High linear R^2** -> the two spaces are related by an affine map. Attribute
  directions established in ECAPA space (where there is substantial existing
  literature on age and speaker attributes) transfer into style space by
  construction. That is a large shortcut: it means the attribute-axis work can
  borrow from published speaker-embedding results rather than being derived
  from scratch.
- **Low linear R^2, high MLP R^2** -> the spaces are related but nonlinearly.
  Usable as an encoder, but attribute directions will not transfer directly and
  must be derived natively in style space (phase 3).
- **Low R^2 either way** -> style space encodes something meaningfully
  different from speaker-verification space. Informative on its own, and a
  signal that the direct encoder (2a) is the only viable route.

Report the probe result before building on either branch. It changes how much
of phase 3 is needed.

## Phase 3: Deriving the Axes

### Are age and vocal presentation root attributes?

Almost certainly not. They are derived and heavily **entangled**. Both load on
shared underlying factors: F0 mean and range, formant dispersion (approximately
vocal tract length), breathiness (H1-H2), jitter and shimmer, speech rate, and
articulatory precision. Push F0 down to raise perceived age and perceived
gender moves with it.

So do not assume a root-attribute set. Derive it:

1. Extract styles for N labeled speakers using the phase 2 encoder.
2. Compute speaker-normalized deltas — subtract each speaker's own neutral
   before averaging, as in the emotion roadmap's phase 2 formula.
3. Find directions two ways and compare: unsupervised (PCA over the style
   space, then correlate components against measured acoustics) and supervised
   (LDA or linear probe per labeled attribute).
4. **Orthogonalize the resulting axes against each other** (Gram-Schmidt).
   This is the concrete fix for entanglement — it is what makes "make older"
   stop dragging perceived gender along, and it is standard practice in image
   latent-space editing.

### The duration tensor cannot stay neutral

The emotion work leaves `style_dp` untouched by default (`include_duration` is
opt-in), and that was a reasonable conservative choice for emotion. It will not
survive contact with age.

Speaking rate, pause structure, and articulation timing are among the strongest
perceptual cues for age, and they live in DP, not TTL. An age axis that moves
only TTL will produce a voice whose timbre says "older" while its timing says
"younger" — the two cues fight, and the result reads as uncanny rather than
old. The same applies, more weakly, to arousal-linked emotion.

So: derive DP deltas jointly with TTL for every axis, not as an afterthought.
DP is only 128 numbers against TTL's 12,800, so it is cheap to fit but easy to
overfit — expect it to need heavier regularization, and evaluate it separately
before combining.

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

## Phase 4: Datasets

Licensing is not a selection criterion — this work is going out open-source.
Select purely on speaker count, metadata quality, and recording conditions.

| Dataset | Speakers | Metadata | Role |
|---|---|---|---|
| **VCTK** | 110 | age, gender, accent | First pass. Studio-clean, best quality-per-speaker, exactly the right shape for deriving initial axes. |
| **Common Voice** | very large | self-reported age bracket, gender, many languages | The only realistic source for an age axis at scale, and the only route to axes that hold across languages. |
| **LibriTTS-R** | ~1000+ | gender (no age) | Gender axis and identity-space PCA. Restored audio, so cleaner than LibriTTS. |
| **CREMA-D** | 91 | age, gender, ethnicity | Keeps emotion and demographics in one frame — lets emotion deltas be measured against demographic controls rather than confounded with them. |
| **RAVDESS** | 24 | acted emotion | Already extracted. Keep for emotion continuity; too few speakers and too acted for demographic axes. |
| **ESD** | 20 | emotion, EN/ZH | Extending emotion multilingually, later. |

Practical notes:

- **Channel leakage is the main data risk.** Common Voice recording conditions
  vary enormously and channel characteristics will be absorbed into the style
  embedding, appearing as a spurious axis. Mitigate by filtering on SNR and by
  averaging many utterances per speaker so channel cancels while speaker
  identity persists. Verify by checking whether a top PCA component correlates
  with recording SNR rather than with any speaker attribute.
- **Age will be perceived age from coarse brackets, not chronological age.** No
  corpus provides real ages at scale. State this in the API and docs so the
  parameter is not over-promised.
- Start with VCTK alone. If clean axes do not emerge from 110 well-labeled
  studio speakers, adding noisier data will not rescue them.

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
- **Intelligibility at the extremes.** WER via Whisper at the ends of each
  range and beyond. Extrapolation past the observed data will break
  intelligibility first, and this is what determines the safe parameter range.
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

## Sequencing

```text
0. Linearity gate (interpolate two presets)          [KILL GATE]
1. Fix intensity=0 normalization; accumulate-then-normalize;
   refactor to with_deltas()
2. Amortized inversion: direct encoder + linear probe from ECAPA
   -> report R^2 before proceeding                   [DECISION POINT]
3. VCTK extraction -> PCA + supervised probes -> orthogonalize axes
4. Build the parselmouth measurement loop
5. Scale to Common Voice / LibriTTS-R; re-derive emotion deltas
   multi-speaker
6. Expose the axes in the Python and web APIs
```

Phases 0 and 1 are prerequisites for everything. Phase 2's probe result decides
how much of phase 3 is needed. Do not start phase 5 scale-up before phase 4
exists, or there will be no way to tell whether more data helped.

## Open Questions, Answered

**What would a good base voice be?** Do not pick one. Speaker-normalized deltas
exist precisely so directions are speaker-independent. If a base is needed for
demos, use the **centroid of the preset voices' TTL** rather than M1 — it
minimizes extrapolation distance to any attribute target. Confirm the centroid
synthesizes cleanly; that is the same convexity question as phase 0.

**Are perceived age and vocal presentation true attributes?** No — see phase 3.
Derived, entangled, and best obtained empirically with explicit
orthogonalization rather than assumed.

**Which datasets?** See phase 4. VCTK first, Common Voice for age scale.

**Could the original embedder be recovered via vec2vec, and would it be
useful?** The capability is very useful — it is the phase 2 bottleneck-breaker.
vec2vec is the wrong tool for obtaining it, because paired data is freely
generatable and supervised regression is strictly easier than unpaired
translation. See phase 2.

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
