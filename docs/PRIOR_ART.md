# Prior Art

Three community repos build on top of Supertonic's released ONNX graphs.
None was executed here — a research agent cloned all three and read their
code, configs, and READMEs, but ran no inference and did no GPU work. Every
runtime, VRAM, or convergence number below is therefore marked **reported by
the repo, not reproduced here**; only structural claims (what inputs go in,
what outputs come out, what loss is optimized, what license applies) were
checked against the repos' own source.

Read this alongside [STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md) —
the closing section there ties the strongest of these three back into this
fork's own weights-level finding.

## kdrkdrkdr/supertonic.embed

MIT-licensed code. Weights are not included; it downloads
Supertone/supertonic-2 ONNX plus WavLM-Large at runtime, so OpenRAIL-M
applies to the weights it pulls in, not to the repo's own code. Last activity
2026-09-05; actively maintained.

**Not a trained encoder.** It is gradient-based inverse optimization: convert
the four released ONNX graphs to differentiable PyTorch with `onnx2torch`
(plus a small `Clip`-op patch), freeze every weight, and backpropagate
through the whole frozen pipeline to move a `style_ttl` tensor until frozen
WavLM-Large layer-4 time-pooled statistics (mean, std) match a reference WAV
(MSE loss, threshold 0.24). `style_dp` is recovered separately, by a
secant/fixed-point search fitting an 8x16 vector to a target syllable rate
measured by peak-picking the reference's RMS envelope. Every row of both
tensors is re-projected onto the unit sphere after each optimization step.

Reported by the repo, not reproduced here: ~0.465 s/step solo, 0.090
s/recording/step batched at 8 speakers, 9.8 GB VRAM, ~15 minutes for a
5-speaker default batch on an RTX 3090.

Targets supertonic-2 — the same release this fork ships.

Two things worth calling out explicitly:

- Its `style_dp` treatment independently corroborates this fork's own
  finding (recorded in CLAUDE.md) that `style_dp`'s only externally visible
  effect is a single scalar duration per utterance — confirmed there by
  direct inference (`shape (1,)`) and the `# dur_onnx: [bsz]` comment at
  `py/helper.py:330`. This repo arrived at the same conclusion from a
  completely different angle: it fits `style_dp` to a *rate*, not to
  per-token durations, because there is nothing else in the graph's output
  to fit against.
- Its per-row unit-sphere re-projection after every optimizer step matches
  this fork's `Style.with_deltas` invariant — normalize per row, over the
  last axis, once — arrived at independently, for an unrelated reason
  (keeping the optimized tensor on the manifold the model was trained on,
  not composability of blends).

**Unverified**: the README's benchmark numbers (147 speakers, ECAPA 0.129 ->
0.419, WER 5.71%) have no accompanying evaluation script in the repo — there
is no way to check how they were produced. Also unverified: whether the code
runs cleanly against this fork's specific asset files. The target release
and the four-graph tensor interface match on paper, but nothing here was
executed against this fork's `assets/onnx/`.

## saurabhv749/supertonic3-voice-clone

MIT-licensed code. Open RAIL-M weights (Supertonic-3); Apache-2.0 for the
SpeechBrain ECAPA-VoxCeleb model it depends on. Last activity 2026-07-08, 7
commits.

An acknowledged fork of the same technique as `supertonic.embed` — its own
header credits "Original Author: Gyeongmin Kim" — swapping the WavLM
layer-stats objective for a SpeechBrain ECAPA-VoxCeleb speaker-verification
embedding distance (1 - cosine similarity), and optimizing `style_ttl` only:
`style_dp` is frozen, taken from the nearest built-in preset rather than
fit.

**Targets Supertonic-3**, a different and newer Supertone release (31
languages, ~99M total ONNX params, reported by the repo) — **not** the v2
four-graph set this fork ships. Its README admits emotional/prosodic cloning
is weak, because the loss rewards speaker identity, not prosody.

Value to this fork: a second data point that the gradient-inversion
technique generalizes across loss functions (WavLM layer stats vs. an
ECAPA speaker embedding) and, less usefully, across model releases. Not
directly usable against this fork's assets — different graphs, different
tensor shapes are likely, and it was not checked against them.

## ORI-Muchim/supertonictts-training

MIT-licensed code, described by its author as reverse-engineering notes for
research/educational use; does not redistribute Supertone's weights. Last
activity 2026-05-13, 15 commits, one open unanswered issue (2026-09-09,
reporting an artifact/metallic-sound bug in the AE decoder).

The only one of the three with genuine training code, not just an inference
optimization loop: real losses (an L1 flow-matching CFM objective matching
the arXiv 2503.23108 paper's formula, an AE-GAN with MPD/MRD discriminators,
a DP loss), a real KSS dataloader, three training loops (`train_ae.py`,
`train_ttl.py`, `train_dp.py`), and `export_onnx.py`. It reconstructs the
model **from scratch**, per the paper plus reverse-engineered ONNX
internals — it is not fine-tuning Supertone's shipped style-conditioning
weights. (`train_ae_ft.py` does fine-tune with the shipped `vocoder.onnx`
decoder frozen, but that is the audio codec, not `text_encoder` or
`vector_estimator` — it doesn't touch style conditioning.)

It ships a first-class style encoder
(`training/models/style_encoder.py`): a GST-style learnable-query
cross-attention pooler over AE-latent features, with output shapes exactly
`[B, 50, 256]` and `[B, 8, 16]` — matching this fork's `style_ttl` /
`style_dp` shapes exactly — and `extract_voice_style.py` runs it as a
genuine single-forward-pass audio -> style-tensor encoder, the thing an
inversion approach like the two repos above has no equivalent of.

**The catch**: no pretrained weights are included or downloadable. `*.pt`
files and `training/runs/` are gitignored, and confirmed absent from the git
tree by the research agent that read it. The only completed training run is
single-speaker Korean (KSS, 12.86 hours of audio), and the repo's own README
states outright that this cannot teach the model to use reference-speaker
variation — genuine zero-shot style encoding would need the scale the
original paper trained on (945 hours, ~2,576 speakers, reported by the
paper/repo). Reported by the repo, not reproduced here: the completed
700k-step single-speaker TTL run alone took ~74 hours.

**A verified graph mismatch worth recording.** This repo's own ONNX dump of
"the released `vector_estimator`" reports 964 nodes / 33.0M params / 132 MB.
This fork's actual `assets/onnx/vector_estimator.onnx` is **1004 nodes /
64.0M params (all float32) / 256,534,781 bytes** — measured directly by the
main session (file size confirmed again while writing this doc: `ls -la` on
`assets/onnx/vector_estimator.onnx` returns exactly `256534781` bytes).
Both claim to be supertonic-2. The cause is unconfirmed here — a different
HF snapshot, a different export precision, or something else entirely.
Anyone porting code between this fork and that repo (or reusing its node
indices, e.g. for locating `main_blocks`) should check this mismatch first;
node/parameter counts do not transfer as-is.

## Comparison table

| Repo | What it is | Inputs -> outputs (shapes) | Weights included | License | Last activity | Usable against this fork's v2 assets |
|---|---|---|---|---|---|---|
| kdrkdrkdr/supertonic.embed | Gradient inversion through frozen ONNX (onnx2torch) against a WavLM layer-stats objective | reference WAV -> `style_ttl [1,50,256]`, `style_dp [1,8,16]` | No (downloads at runtime) | MIT code / OpenRAIL-M weights | 2026-09-05, active | Targets the same supertonic-2 release; not executed against this fork's specific assets |
| saurabhv749/supertonic3-voice-clone | Same inversion technique, ECAPA speaker-embedding objective, `style_ttl` only | reference WAV -> `style_ttl` (Supertonic-3 shape, unverified against v2) | Weights: Open RAIL-M (Supertonic-3) | MIT code / Open RAIL-M + Apache-2.0 | 2026-07-08, 7 commits | No — targets Supertonic-3, a different release |
| ORI-Muchim/supertonictts-training | From-scratch reimplementation + real training code + a trained-encoder architecture | audio -> `style_ttl [B,50,256]`, `style_dp [B,8,16]` (encoder); text+style -> audio (full retrain) | No (gitignored, confirmed absent) | MIT | 2026-05-13, 15 commits | Architecture only — no usable pretrained weights, and its own `vector_estimator` dump has a verified node/param mismatch against this fork's file |

## What this changes for us

All three repos confirm the same premise this fork's own CLAUDE.md and
[STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md) already record from the
other direction: a "frozen" ONNX graph bounds what you can *change*, never
what you can *read* or *differentiate through*. `supertonic.embed` and
`supertonic3-voice-clone` both demonstrate, independently, that gradients
survive an `onnx2torch` conversion of these graphs with only minor op-level
patching — which is exactly what this fork's own
`py/onnx2torch_patches.py` also shows, with gradients confirmed reaching
12,800 of 12,800 `style_ttl` elements. Their `style_dp` treatment
(fit-a-scalar-rate, never per-token) is independent corroboration of this
fork's own shape-`(1,)` finding. Their per-row unit-norm projection is
independent corroboration of `Style.with_deltas`'s normalization invariant.

None of the three hands this fork a usable pretrained style encoder for its
own v2 assets: `supertonic.embed` optimizes per-utterance rather than
training a reusable encoder at all; `supertonic3-voice-clone` targets the
wrong model release; `supertonictts-training` has the right encoder
*architecture* and the right output shapes but no weights and an admittedly
insufficient (single-speaker, Korean-only) training run even if its own
training pipeline were run to completion. See
[STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md)'s closing section for
how the inversion approach specifically (not the missing encoder) composes
with this fork's own weights-level finding about `style_ttl`'s gain
structure.
