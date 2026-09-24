# Style Control Surface

A structural finding about how `text_encoder` and `vector_estimator`
actually consume `style_ttl`, measured directly against this fork's own
ONNX files (`assets/onnx/text_encoder.onnx`, `assets/onnx/vector_estimator.onnx`)
by the main session. Every number in this document was produced on this
machine, from the graph files themselves — no rendering, no audio, no
listener involved anywhere in it.

Read this alongside [ATTENTION_READOUT.md](ATTENTION_READOUT.md), which this
finding retroactively explains, and CLAUDE.md's existing notes on
`style_ttl` / `style_dp`, which it extends rather than corrects.

## The finding

`style_ttl` is not 12,800 free numbers read symmetrically wherever they
enter the graph. It is the **value bank of a 50-slot style-token attention
layer whose keys are learned constants**, not derived from `style_ttl`
itself. Where a style perturbation lands is governed by fixed, style-blind
routing; only *what* gets retrieved through that routing depends on the
style.

## Evidence

**Six MatMuls, all `W_value`.** Walking forward from the `style_ttl` graph
input to the first linear op it reaches, in both graphs, `style_ttl` reaches
exactly six `MatMul` nodes and nothing else:

- `text_encoder`: `/speech_prompted_text_encoder/attention1/W_value/linear/MatMul`,
  `/speech_prompted_text_encoder/attention2/W_value/linear/MatMul`
- `vector_estimator`: `/vector_estimator/vector_field/main_blocks.{5,11,17,23}/attention/W_value/linear/MatMul`

All six are `W_value` projections. `style_ttl` never reaches a `W_key` or
`W_query` projection directly, in either graph.

**The keys are learned constants, not derived from style.** The key tensor
feeding `text_encoder`'s `attention1` traces back, through a `Tile` and an
`Expand`, to an initializer literally named
`tts.ttl.style_encoder.style_token_layer.style_key` — a fixed weight baked
into the graph at export time, not a function of any input. Rendering with
two completely different random `style_ttl` tensors gives a **bit-identical**
key tensor of shape `(1, 50, 256)` — max absolute difference `0.0`.
Addressing into the 50 style rows is style-independent by construction: it
is the same computation whatever `style_ttl` is.

**One exception, worth stating precisely.** `text_encoder`'s `attention2`
is the one place where style modulates its *own* routing: its `W_query`
traces back to `style_ttl`, not to a fixed initializer. In
`vector_estimator`, `W_key` comes from `noisy_latent` and `W_query` comes
from `noisy_latent`/`text_emb` — and `text_emb` itself depends on
`style_ttl` (via `text_encoder`), so `vector_estimator`'s routing is
indirectly style-dependent, through the text encoder's output, even though
no `vector_estimator` query or key traces to `style_ttl` directly.

## Spectra: the gain structure

Each `W_value` matrix's singular-value spectrum, compared against a
shape-matched i.i.d. Gaussian null of the same size (per CLAUDE.md's
calibration rule: a diagnostic that returns the same number on noise is
measuring the shape, not the content). Participation ratio and dimensions
needed for 95% of spectral energy, each out of a 256-dimensional space:

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

Matrix rank is 254-256 in every case, i.e. essentially full rank, so there
is **no hard null space** anywhere in these six matrices. Directions are
low-gain, not invisible — a perturbation along a low-participation direction
is attenuated on its way through, not blocked. This matters for how the
finding below should be read: it is a statement about gain, not about
reachability.

**Combined surface.** Stacking all six `W_value` matrices to a single
1536x256 matrix and taking its spectrum:

- participation ratio **199.7** (null: 245.2 +/- 0.0)
- dimensions for 50% / 90% / 95% / 99% of energy: **40 / 154 / 190 / 237**
  of 256 (null dims95: 227.2 +/- 0.4)
- top-to-bottom singular-value ratio: **8.1x** (null: 2.34x)

The combined surface is more concentrated than any individual matrix's
null, and markedly more concentrated than the individual matrices'
own nulls too — consistent with the six matrices sharing some structure
rather than each independently concentrating the same amount.

## What it explains

Two earlier results in this project, both measured before this structural
picture existed, are retro-explained by it. Neither number below is being
recomputed here — both are exactly as reported in their own docs; only the
explanation is new.

- **Task #18's negative was structural, not statistical.** That work (see
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
- **Task #20's open question is the spectrum of the value projections.**
  That task found that equal-Frobenius-norm perturbations to `style_ttl`
  differed by roughly 70x in how much attention movement they produced,
  with no explanation on record. The combined `W_value` surface's 8.1x
  top-to-bottom singular-value ratio is a 66x ratio in *energy*
  (gain squared) — which brackets the observed ~70x. This is offered as the
  likely mechanism, not a proof: task #20's 70x was measured on attention
  movement downstream of these matrices, not on the matrices' output
  directly, so the bracket is suggestive rather than a verified causal
  chain.

## What it proposes (proposal, not result — none of this has been run)

1. **Parametrize in gain coordinates, not in the tensor's raw shape.**
   Phase 2b's rank ceiling (see `new-plan.md`) was `n_train/d` with
   `d = 6144`, giving 3.9% reachability out of sample. Against an *effective*
   dimension of ~190 (95% energy) or ~40 (50% energy) rather than the raw
   6144, `n_train = 240` stops being data-starved. Whether the data problem
   is partly a coordinate-system problem is untested, but it is now a
   specific, checkable claim rather than a guess.
2. **Predict before rendering.** For the 24 task-#20 styles, predict
   attention movement from each style's alignment with the high-gain
   subspace identified above. The relevant tensors (the 24 `style_ttl`
   perturbations, the six `W_value` matrices) are already on disk; this
   costs zero renders and is a falsifiable test of the whole picture in this
   document — if alignment with the high-gain subspace does not predict the
   70x spread, the bracket above is coincidental.
3. **Slot-wise control.** Because the keys are fixed and style-independent
   (outside the one `text_encoder attention2` exception), *which* of the 50
   style slots a given text position reads is computable directly from the
   learned `style_key` constant and the text alone — no rendering required.
   That makes "which slot carries which perceptual quality" an answerable
   question in principle, though answering it is future work, not something
   this document establishes.

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
afterward — the value vectors retrieved through this layer are then read by
whatever comes after each attention block, which this trace does not follow.

## How the prior art fits

[PRIOR_ART.md](PRIOR_ART.md)'s `kdrkdrkdr/supertonic.embed` does gradient
descent through these same frozen graphs, converted with `onnx2torch`, to
move a `style_ttl` tensor toward a target. That is the same premise as this
document — a frozen graph bounds what you can *change*, not what you can
*read* — arrived at from the opposite end: that repo needed gradients to
flow through the frozen weights; this document needed intermediate
*activations* to be exposed. Both turn out to be available with no
retraining.

The two compose directly: gradient descent in 12,800 raw dimensions with an
8.1x conditioning spread (66x in curvature/energy terms) is exactly the
regime where preconditioning the optimizer in the gain basis identified
above — rather than in the tensor's native coordinates — should help
convergence, though this is again proposal, not something either repo or
this document has tested. And this fork already has the infrastructure to
try it: `py/onnx2torch_patches.py` converts all four ONNX models with two
generic version patches, with gradients verified reaching all 12,800
entries of `style_ttl` (see `new-plan.md`) — independently built, before
this document's spectral analysis existed, and now with a concrete reason
to revisit how those gradients are conditioned.
