# Style Control Surface

A structural finding about how `text_encoder` and `vector_estimator`
actually consume `style_ttl`, measured directly against this fork's own
ONNX files (`assets/onnx/text_encoder.onnx`, `assets/onnx/vector_estimator.onnx`)
by the main session. Every number in this document was produced on this
machine, from the graph files themselves — no rendering, no audio, no
listener involved anywhere in it.

Read this alongside [ATTENTION_READOUT.md](ATTENTION_READOUT.md), whose
calibration failure this finding was offered as an explanation of (a claim now
downgraded, see below), and CLAUDE.md's existing notes on `style_ttl` /
`style_dp`, which it extends rather than corrects.

> **Corrections (2026-09-29).** An audit re-checked this document against the
> graphs and against the project's own data. Each correction is marked where it
> appears; in summary:
>
> - **"The 70x spread is the spectrum of `W_value`" is FALSIFIED** — by the
>   document's own proposed test #2, run and failed (see "What it explains").
> - **"`style_ttl` never reaches a `W_key` or `W_query`" is false as
>   reachability.** What is true is that it *first enters* only through
>   `W_value`. Full forward reachability reaches six `W_query` projections.
> - **"In `vector_estimator`, `W_key` comes from `noisy_latent`" was wrong.**
> - **The concentration argument was reversed** (the stack is not "more
>   concentrated because the matrices share structure"), and the
>   participation-ratio convention was undeclared.
> - **"Task #18's negative was structural, not statistical" is downgraded to
>   one untested candidate explanation.**
> - **The closing section mislabelled the prior art's release** (v2 vs. v3).

## The finding

`style_ttl` is not 12,800 free numbers read symmetrically wherever they
enter the graph. It is the **value bank of a 50-slot style-token attention
layer whose keys are learned constants**, not derived from `style_ttl`
itself. `style_ttl` first enters the graph only through `W_value`. *(Corrected
2026-09-29: this paragraph originally continued "Where a style perturbation
lands is governed by fixed, style-blind routing; only what gets retrieved
through that routing depends on the style". That holds for 2 of the 6 style
nodes — `text_encoder` `attention1` and `vector_estimator` block 5 — and not
for the other four, whose queries depend on style through the residual
stream; see "Full reachability" below.)*

## Evidence

**Six MatMuls, all `W_value` — as the *first* linear op.** Walking forward
from the `style_ttl` graph input to the first linear op it reaches, in both
graphs, `style_ttl` reaches exactly six `MatMul` nodes and nothing else (this
trace was reproduced by the audit):

- `text_encoder`: `/speech_prompted_text_encoder/attention1/W_value/linear/MatMul`,
  `/speech_prompted_text_encoder/attention2/W_value/linear/MatMul`
- `vector_estimator`: `/vector_estimator/vector_field/main_blocks.{5,11,17,23}/attention/W_value/linear/MatMul`

All six are `W_value` projections. The first linear ops downstream of
`style_ttl` are `W_value` and nothing else: **`style_ttl` first enters only
through `W_value`.** *(Corrected 2026-09-29: this paragraph originally said
"`style_ttl` never reaches a `W_key` or `W_query` projection directly". As a
statement about reachability that is false; see "Full reachability" below.
The word "directly" carried the whole claim.)*

**The keys are learned constants, not derived from style.** The key tensor
feeding `text_encoder`'s `attention1` traces back, through a `Tile` and an
`Expand`, to an initializer literally named
`tts.ttl.style_encoder.style_token_layer.style_key` — a fixed weight baked
into the graph at export time, not a function of any input. Rendering with
two completely different random `style_ttl` tensors gives a **bit-identical**
key tensor of shape `(1, 50, 256)` — max absolute difference `0.0`.
The *keys* into the 50 style rows are style-independent by construction: the
same tensor whatever `style_ttl` is. Whether the *addressing* — key against
query — is style-independent depends on the query; see the next paragraphs.

**Full reachability (corrected 2026-09-29).** The original text here made a
first-linear-op trace do the work of a reachability claim. Traced forward
through the whole graph (`Shape` ops excluded), `style_ttl` reaches:

- `W_query` of `text_encoder` `attention2`, via `attention1`'s output.
- `W_query` of `vector_estimator` main blocks 9, 11, 15, 17, 21 and 23, via the
  residual stream after block 5's `W_value`.

So "addressing over slots is style-independent" holds for **2 of the 6** style
nodes — `text_encoder` `attention1` and `vector_estimator` block 5. The other
four (`text_encoder` `attention2`; `vector_estimator` blocks 11, 17, 23) have
style-dependent queries through the residual. The keys, in contrast, are
constants everywhere.

**Where each `W_key` comes from.** The original text said "In
`vector_estimator`, `W_key` comes from `noisy_latent`". That is wrong for the
style blocks. Their `W_key` inputs have no data dependence on any graph input:
they are a `Tile` of a folded initializer, `/vector_estimator/Expand_output_0`
(bit-identical to `tts.ttl.style_encoder.style_token_layer.style_key`,
`np.array_equal` True), concatenated with the unconditional key token. The
`noisy_latent` dependence seen earlier was a `Shape`-op batch-size dependency
only, not a data one. `text_encoder` `attention2`'s `W_key` is likewise the
constant `Tile`. In the *text*-attention blocks (`main_blocks.{3,9,15,21}`),
`W_key` comes from `text_emb`, which is style-conditioned via `text_encoder`.

## Spectra: the gain structure

Each `W_value` matrix's singular-value spectrum, compared against a
shape-matched i.i.d. Gaussian null of the same size (per CLAUDE.md's
calibration rule: a diagnostic that returns the same number on noise is
measuring the shape, not the content). Participation ratio and dimensions
needed for 95% of spectral energy, each out of a 256-dimensional space.

**Convention (declared 2026-09-29; it was not before).** The participation
ratio here is `(sum s)^2 / sum s^2` over singular values `s` — the
*amplitude* convention. `py/phase2a_ceiling_null.py` uses the *energy*
convention (squared singular values). The two give different numbers for the
same matrix; under the energy convention the stack below is **113.7 against a
null of 219.2**. Do not compare figures across the two docs without checking
which is meant. The per-matrix figures below were not recomputed under the
energy convention.

| Matrix | Participation ratio | dims for 95% energy |
|---|---|---|
| text_encoder attention1 | 57.7 | 46 |
| text_encoder attention2 | 90.7 | 73 |
| vector_estimator block 5 | 130.9 | 104 |
| vector_estimator block 11 | 129.2 | 102 |
| vector_estimator block 17 | 142.7 | 117 |
| vector_estimator block 23 | 118.1 | 93 |
| **null** (i.i.d. Gaussian 256x256, 8 draws) | 184.2 +/- 0.3 | 156.4 +/- 0.5 |

All six real matrices are clearly more concentrated than the null — none of
them is a random rotation.

Matrix rank is 256 in every case at numpy's default tolerance (an earlier
version of this document said 254-256), so there is **no hard null space**
anywhere in these six matrices. But "low-gain, not invisible" does **not**
hold uniformly per layer. Condition numbers run 2.8e3 to 1.2e5, and for
`text_encoder` `attention1` 42 of 256 singular values are below 1e-3 of the
maximum — a direction there is attenuated by three orders of magnitude, which
for practical purposes is blocked. "Low-gain, not invisible" is true of the
*stack* (below), whose min/max singular value is 0.124, and not of each layer
taken alone. Either way this is a statement about gain, not about
reachability.

**Combined surface.** Stacking all six `W_value` matrices to a single
1536x256 matrix and taking its spectrum:

- participation ratio **199.7** (null: 245.2 +/- 0.0; amplitude convention —
  under the energy convention 113.7 against 219.2)
- dimensions for 50% / 90% / 95% / 99% of energy: **40 / 154 / 190 / 237**
  of 256 (null dims95: 227.2 +/- 0.4)
- top-to-bottom singular-value ratio: **8.1x** (null: 2.34x)

*(Corrected 2026-09-29. The original paragraph said the stack is "more
concentrated than any individual matrix's null ... consistent with the six
matrices sharing some structure". That has the logic backwards.)* The stack's
participation ratio, 199.7, is **above** every individual matrix's (57.7 to
142.7) and above the 256x256 null (184.2); it is below only the shape-matched
1536x256 null (245.2). Stacking made the surface *less* concentrated than any
single layer. That is what six matrices concentrating on **different**
directions look like — complementary, not shared. The stack still departs from
its own null (199.7 against 245.2; 8.1x top-to-bottom against 2.34x), so it is
structured, just not because the layers overlap.

## What it explains

*(Marked 2026-09-29: this section originally claimed to explain two earlier
results. One explanation is downgraded to a candidate; the other is
falsified.)* Two earlier results in this project, both measured before this
structural picture existed, were retro-explained by it. Neither number below
is being recomputed here — both are exactly as reported in their own docs.

- **Task #18's negative: DOWNGRADED (2026-09-29) from "structural, not
  statistical" to one untested candidate explanation.** The original claim
  follows, then why it does not stand as written. That work (see
  [ATTENTION_READOUT.md](ATTENTION_READOUT.md)'s calibration section) used
  the row-attention profile — i.e., the *routing weights* over `style_ttl`'s
  50 rows — as a per-style readout, and found it could not separate two
  shipped voices from vocoder-seed noise on the same voice (AUC 0.826,
  80-90% distribution overlap). The finding above says why: that routing is
  computed against the fixed learned `style_key` constant, so it is
  near-independent of style content by construction, in every attention node
  except `text_encoder`'s `attention2`. The instrument was reading the
  addressing scheme, which barely moves with style; the style itself lives
  in the values it addresses, which the row-attention profile never looks
  at.

  *Why it is downgraded.* "Near-independent of style content by construction,
  in every attention node except `text_encoder`'s `attention2`" is
  contradicted by the full-reachability result above: addressing is
  style-independent at 2 of the 6 style nodes, and style-dependent at four
  (queries see style through the residual stream). So the failure was not
  purely structural, and "structural, not statistical" cannot be asserted. It
  stands as **one candidate explanation**, alongside a second that is also
  untested: `analyze()` averages the style attention over both streams, which
  behave differently (see
  [ATTENTION_READOUT.md](ATTENTION_READOUT.md), "The style family"), and no
  stream-0-only calibration has been run. Neither candidate has been tested.
- **Task #20's open question: FALSIFIED (2026-09-29).** Task #20 is the
  project's label for the observation that equal-Frobenius-norm perturbations
  to `style_ttl` — the 24 `ttl_K64` styles of task #19's Part B (see
  ATTENTION_READOUT.md, "Seed-averaging") — differ in how much they move
  attention. The original text here said the spread was "roughly 70x" and
  that the combined `W_value` surface's 8.1x top-to-bottom singular-value
  ratio, "a 66x ratio in *energy* (gain squared) — which brackets the
  observed ~70x", was "the likely mechanism".

  **That is falsified, by this document's own proposed test #2 below, run and
  failed.** Pushing the 24 ladder deltas (`py/results/phase2b_subspace/
  subspace.npz`, indexed from `py/results/phase3_seed_averaging/
  part_b_ladder.json`) through the stacked six `W_value` matrices gives gain
  **4.940 to 5.126, a 1.04x range**. The measured attention signal energy for
  the same 24 styles ranges 5.73e-7 to 9.02e-6, **15.7x**.
  Spearman(gain, signal energy) = **-0.087, p = 0.69**. The audit, correlating
  gain against paired TV rather than signal energy, got -0.15 (p = 0.49): same
  conclusion. `W_value` gain barely varies across these samples — random
  perturbations in high dimension have near-equal gain — and it does not
  correlate with the effect.

  Two further errors in the original claim. **The "70x" was never an observed
  movement spread.** It is the spread in *extrapolated required N* (69.8 to
  4747, 68x); since N scales as 1/TV^2, the implied movement spread is about
  8.25x, the measured per-style paired-TV mean varies 1.76x (0.00645-0.01135),
  and the unbiased signal energy 15.7x. And **"8.1x amplitude = 66x energy
  brackets ~70x" mixed an amplitude ratio with an energy-scale quantity;** 66
  does not bracket 70 in any case. What explains the spread across styles is
  open.

## What it proposes (proposal, not result — none of this has been run, except #2, which failed)

1. **Parametrize in gain coordinates, not in the tensor's raw shape.**
   Phase 2b's rank ceiling (see `new-plan.md`) was `n_train/d` with
   `d = 6144`, giving 3.9% reachability out of sample. Against an *effective*
   dimension of ~190 (95% energy) or ~40 (50% energy) rather than the raw
   6144, `n_train = 240` stops being data-starved. Whether the data problem
   is partly a coordinate-system problem is untested, but it is now a
   specific, checkable claim rather than a guess.
2. **[RUN 2026-09-29 — FAILED.]** *Predict before rendering.* For the 24
   task-#20 styles, predict
   attention movement from each style's alignment with the high-gain
   subspace identified above. The relevant tensors (the 24 `style_ttl`
   perturbations, the six `W_value` matrices) are already on disk; this
   costs zero renders and is a falsifiable test of the whole picture in this
   document — if alignment with the high-gain subspace does not predict the
   70x spread, the bracket above is coincidental. **Result:** it did not; gain
   varies 1.04x across the 24 styles and Spearman(gain, energy) = -0.087
   (p = 0.69). The bracket was coincidental. (The comparison was run against
   the measured signal energy and against paired TV, both at ~ zero
   correlation; the "70x" was not a movement spread to begin with — see
   above.)
3. **Slot-wise control.** The keys are fixed and style-independent at all six
   style nodes. Where the *query* is also style-independent — 2 of the 6:
   `text_encoder` `attention1` and `vector_estimator` block 5 — *which* of the
   50 style slots a given text position reads is computable directly from the
   learned `style_key` constant and the text alone, no rendering required.
   *(Corrected 2026-09-29: this proposal originally said the addressing was
   style-independent everywhere outside one `text_encoder attention2`
   exception. At the other four nodes the queries depend on style through the
   residual stream, so the read is style-dependent there.)* That makes "which
   slot carries which perceptual quality" an answerable question in principle
   at those two nodes, though answering it is future work, not something this
   document establishes.

## Honest limits

This is a **weights-level** argument. Nothing in it has been connected to
audibility — no clip was rendered, no listener was asked anything, while
producing any number above. It is deterministic and seed-free, which is a
real advantage over every distance this project has tried and calibrated so
far (see ATTENTION_READOUT.md's calibration section on how much of this
project's prior instrumentation turned out to be measuring vocoder-seed
noise). But "the model is more sensitive to this direction, in gain terms"
is not the same claim as "a listener hears this direction," and the gap
between those two claims is not closed anywhere in this document.

Also: the six-MatMul trace stops at the **first linear op** downstream of
`style_ttl` in each graph. It establishes where `style_ttl` first enters
each graph's computation, not everything the resulting values touch
afterward — and that gap is where this document was wrong. The value vectors
retrieved through these layers are read by whatever comes after each
attention block, and following them (the audit's full forward reachability,
`Shape` ops excluded) reaches six `W_query` projections. The first-op trace
was correct; the reachability claim built on it was not.

## How the prior art fits

*(Corrected 2026-09-29. This section originally said `kdrkdrkdr/supertonic.embed`
does gradient descent "through these same frozen graphs". It does not: that
repo is v2 code, and this fork ships Supertonic 3. See
[PRIOR_ART.md](PRIOR_ART.md)'s correction.)*
[PRIOR_ART.md](PRIOR_ART.md)'s `saurabhv749/supertonic3-voice-clone` is the
**v3** reference: gradient descent through the frozen v3 graphs, converted
with `onnx2torch`, to move a `style_ttl` tensor toward a target.
`kdrkdrkdr/supertonic.embed` shares the technique, on v2 graphs. That is the
same premise as this document — a frozen graph bounds what you can *change*,
not what you can *read* — arrived at from the other end: those repos need
gradients to flow through the frozen weights; this document needed
intermediate *activations* to be exposed. Both turn out to be available with
no retraining. (The original framing, "arrived at from the opposite end", is
kept as a description of the two methods, not as a claim about which graphs
were used.)

The two compose directly: gradient descent in 12,800 raw dimensions with an
8.1x conditioning spread (66x in curvature/energy terms — a property of the
stack's spectrum; note the attempt above to tie it to the ladder's spread
failed) is exactly the
regime where preconditioning the optimizer in the gain basis identified
above — rather than in the tensor's native coordinates — should help
convergence, though this is again proposal, not something either repo or
this document has tested. And this fork already has the infrastructure to
try it: `py/onnx2torch_patches.py` converts all four ONNX models with two
generic version patches, with gradients verified reaching all 12,800
entries of `style_ttl` (see `new-plan.md`) — independently built, before
this document's spectral analysis existed, and now with a concrete reason
to revisit how those gradients are conditioned.
