# Prior Art

Three community repos build on top of Supertonic's released ONNX graphs.
None was executed here — a research agent cloned all three and read their
code, configs, and READMEs, but ran no inference and did no GPU work. Every
runtime, VRAM, or convergence number below is therefore marked **reported by
the repo, not reproduced here**; only structural claims (what inputs go in,
what outputs come out, what loss is optimized, what license applies) were
checked against the repos' own source.

> **Correction (2026-09-29): this fork ships Supertonic 3, not Supertonic 2.**
> The first version of this document said `supertonic.embed` "targets
> supertonic-2 — the same release this fork ships" and dismissed
> `supertonic3-voice-clone` as targeting "the wrong model release". Both are
> backwards. The shipped assets are v3: `assets/README.md` is titled
> "Supertonic 3" and lists 31 languages; the four graphs total **99.2M**
> parameters (duration_predictor 0.9M, text_encoder 9.0M, vector_estimator
> 64.0M, vocoder 25.3M), matching v3's published ~99M; and the fork's
> README has cloned `Supertone/supertonic-3` into `assets/` since commit
> `0a98c9f` on 2026-05-06, four months before this research began on
> 2026-09-08. So **every measurement in this repository was taken on v3.**
> The v2 claim came from the research agent and was relayed without being
> checked against the assets. The sections below are corrected in place;
> the table and closing section are rewritten.
>
> Two version facts that matter for anything ported from these repos.
> Upstream describes v3's public ONNX assets as **"v2-compatible"** — same
> tensor names and shapes — but the graphs underneath differ (see the
> `vector_estimator` comparison below), so code that converts or patches
> graph internals does not transfer on the strength of a matching
> interface. And **style tensors are version-specific**: upstream's Voice
> Builder ships separate JSON files for v2 and v3. A style tensor derived
> against v2 graphs is shape-compatible with this fork's engine and is not a
> valid input to it.

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

Targets **supertonic-2** — a *different* release from the v3 engine this
fork ships (see the correction above). The technique is the valuable part;
its `onnx2torch` patches were written against v2's graph internals and have
not been checked against v3's, whose `vector_estimator` is roughly twice the
size. Any style tensors it produces are v2 tensors and not valid inputs
here.

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

**Targets Supertonic-3 — the same release this fork ships.** (The first
version of this document said the opposite; see the correction above.) Its
README admits emotional/prosodic cloning is weak, because the loss rewards
speaker identity, not prosody.

Value to this fork: of the three, **the closest match to the engine this
fork actually runs.** Its `onnx2torch` conversion was done against v3
graphs, so it is the likeliest of the three to convert this fork's assets
without re-patching, and the style tensors it produces are v3 tensors.
Still not executed here. It also shows the gradient-inversion technique
generalizes across objectives (an ECAPA speaker embedding here, WavLM
layer statistics in `supertonic.embed`).

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

**A graph mismatch worth recording, now explained.** This repo's own ONNX dump
of "the released `vector_estimator`" reports 964 nodes / 33.0M params /
132 MB. This fork's actual `assets/onnx/vector_estimator.onnx` is **1004
nodes / 64.0M float32 params / 256,534,781 bytes**. *(Corrected 2026-09-29:
an earlier version of this paragraph said "1004 nodes / 64.0M params (all
float32)" and "A verified graph mismatch ... measured directly". Only **this
fork's** side was measured directly — file size checked with `ls -la`
(exactly `256534781` bytes), node count and parameters read from the graph.
The repo's 964-node / 33.0M-param figures are relayed from its own dump and
remain **unchecked**. And "all float32" was imprecise: `vector_estimator` also
carries 85 INT64 initializers (1,127 elements); the float parameters are
64,013,449, hence "64.0M float32 params".)*

The first version of this document said both files claim to be supertonic-2
and called the mismatch unexplained; the previous revision then called the
v2-vs-v3 reading "an inference from dates and sizes, not a confirmed
identification". **It is now confirmed by file size.** The Hugging Face tree
listing for `Supertone/supertonic-2` gives `vector_estimator.onnx` =
**132,471,364 bytes** — the repo's "132 MB". The listing for
`Supertone/supertonic-3` gives **256,534,781** (`vector_estimator`),
36,416,150 (`text_encoder`), 3,700,147 (`duration_predictor`) and 101,424,195
(`vocoder`), byte-identical to this fork's files. Also, 33.0M params x 4 bytes
= 132 MB, so the repo's 132 MB file is float32, not an fp16 export of the same
graph. The repo's dump is of **v2's** `vector_estimator`, and this fork's is
**v3's**. What is *not* confirmed is the 964-node count itself, still relayed.
The practical point stands: node indices and parameter counts do not transfer
between the two, so anyone porting code from that repo should re-derive them
against this fork's file.

## Comparison table

| Repo | What it is | Inputs -> outputs (shapes) | Weights included | License | Last activity | Fit with this fork's v3 engine |
|---|---|---|---|---|---|---|
| saurabhv749/supertonic3-voice-clone | Gradient inversion through frozen ONNX (onnx2torch), ECAPA speaker-embedding objective, `style_ttl` only | reference WAV -> `style_ttl [1,50,256]` | No (downloads at runtime) | MIT code / Open RAIL-M weights + Apache-2.0 (ECAPA) | 2026-07-08, 7 commits | **Same release (v3).** Likeliest of the three to convert this fork's graphs as-is; not executed here |
| kdrkdrkdr/supertonic.embed | Same inversion technique, WavLM layer-stats objective, plus a rate-matching fit for `style_dp` | reference WAV -> `style_ttl [1,50,256]`, `style_dp [1,8,16]` | No (downloads at runtime) | MIT code / OpenRAIL-M weights | 2026-09-05, active | **Different release (v2).** Technique transfers; graph patches and output tensors do not without re-checking |
| ORI-Muchim/supertonictts-training | From-scratch reimplementation, real training code, a trained-encoder architecture | audio -> `style_ttl [B,50,256]`, `style_dp [B,8,16]` (encoder); text+style -> audio (full retrain) | No (gitignored, confirmed absent) | MIT | 2026-05-13, 15 commits | Architecture only — no usable weights, and its `vector_estimator` dump is v2's (132 MB matches the v2 file size exactly; node count relayed, unchecked) |

## What this changes for us

All three repos confirm the premise this fork's own CLAUDE.md and
[STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md) record from the other
direction: a "frozen" ONNX graph bounds what you can *change*, never what
you can *read* or *differentiate through*. Two of them get a style tensor
from a reference recording by gradient descent through the frozen graphs
rather than by training an encoder, and this fork's own
`py/onnx2torch_patches.py` — built and verified against this fork's v3
`text_encoder`, with gradients reaching all 12,800 `style_ttl` elements —
shows the same thing independently. Their `style_dp` treatment
(fit-a-scalar-rate, never per-token) independently corroborates this fork's
shape-`(1,)` finding, and their per-row unit-norm projection corroborates
`Style.with_deltas`'s normalization invariant.

**The recommendation that follows, corrected for the release:** gradient
inversion is the right approach to try before the VCTK extraction plan,
because it sidesteps the ridge rank ceiling instead of fighting it and needs
no dataset. The implementation to start from is this fork's own
`onnx2torch_patches.py` (v3-native, `text_encoder` verified) with
`supertonic3-voice-clone` as the reference for the v3 conversion of the
other three graphs. `supertonic.embed` remains the better-developed
*design* — it fits `style_dp` too, and its WavLM objective is closer to
what this project has already calibrated — but it is v2 code, and porting
it means re-checking every graph-level patch against v3.

None of the three hands this fork a usable pretrained style encoder:
`supertonic.embed` and `supertonic3-voice-clone` optimize per utterance
rather than training a reusable encoder, and `supertonictts-training` has
the right encoder *architecture* and output shapes but no weights and an
admittedly insufficient single-speaker Korean training run. See
[STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md)'s closing section for
how the inversion approach composes with this fork's weights-level finding
about `style_ttl`'s gain structure.
