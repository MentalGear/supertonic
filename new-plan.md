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
almost entirely on the sign of one component.

A few hours of work, it could substantially simplify Phase 3, and either way it
would be the first documented probe of this tensor's structure.

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

Evaluate the encoder against **the optimizer's own converged style**, not only
against downstream audio quality. Audio metrics masked systematic encoder bias
in the early image-inversion work, and the same failure is available here.

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
row-structure question in Phase 0.

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
   probe shortcut actually applies here.
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
  structure.
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
