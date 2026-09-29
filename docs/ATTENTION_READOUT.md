# Attention Readout

> **Correction (2026-09-29): the word-level timestamp claim is WITHDRAWN.**
> An audit checked the "word spans" against the audio they claim to describe,
> and against a text-swap control. They do not survive either.
>
> - **Audio check.** Rendering "The quick brown fox jumps over the lazy dog."
>   (M1, seed 0, 8 steps, `speed=1.0`, duration 3.255 s) and measuring a 10 ms
>   RMS envelope, speech runs 0.39-2.64 s (threshold 35 dB below peak).
>   `word_spans()` places "dog" at 2.79-2.86 s at a mean level of -70.8 dB —
>   entirely *after* speech ends. The other eight words land inside speech,
>   but that is what any ramp spreading tokens evenly across an utterance that
>   is mostly speech would also do. It fails at the tail, where the final
>   tokens ("dog", ".", `</en>`) are assigned to silence.
> - **The chosen head is a positional ramp.** Its argmax
>   (`main_blocks.9/attn`, head 1, stream 1) deviates from a straight line
>   `f * (T-1) / (L-1)` by a mean of 0.77 tokens.
> - **Text-swap control.** With duration pinned so `L` matches, a nonsense text
>   of the same character length ("Xxx xxxxx xxxxx xxx xxxxx xxxx xxx xxxx
>   xxx.") gives, at the final denoising step, a stream-0 argmax that agrees
>   with the fox sentence's on 98% of frames (mean difference 0.09 tokens) and a
>   stream-1 argmax that agrees on 55% (mean difference 0.49 tokens). At the
>   final step, **neither stream of this head tracks text content.**
> - **What the Spearman rho measured.** rho = 0.9995 measured *monotonicity*.
>   Any positional ramp is monotone. It was never a test of content alignment,
>   and it was reported as one.
>
> The original numbers below are left visible, and they do reproduce
> (including the independent CLI re-run in "Provenance"). They reproduce
> because they are the ramp. Reproducibility was checked; validity was not.
>
> **What `py/attention.py` still does correctly:** it exposes the eight
> attention maps, read bit-identically to `helper.TextToSpeech`'s own render
> (the drift-guard test still passes). **What it does not do:** give
> timestamps. Whether a content-tracking alignment exists elsewhere — another
> head, another block, an earlier denoising step, the conditional stream
> alone — is **open and untested.** The audit reports stream 0's best
> non-constant head at rho 0.665 (audit-reported; not reproduced here).
>
> Three further points from the same audit, each marked where it appears
> below:
>
> - **Every readout in this doc is from the final denoising step only**
>   (`py/attention.py` collects at `step == total_step - 1`). This was not
>   stated anywhere before.
> - **The `stream` axis is most likely the classifier-free-guidance pair**
>   (stream 0 conditional, stream 1 unconditional). That is the most likely
>   reading from graph structure, not an established one: `vector_estimator`
>   builds `Concat_5 = [text_emb, masked uncond text token]`,
>   `Concat_6 = [style-key constant, style_key_special_token]` and
>   `Concat_7 = [style_ttl, style_value_special_token]` on the batch axis, and
>   `tts.json` has an `uncond_masker` block. The audit found stream 1
>   bit-identical across two different styles at denoising step 0; a check
>   here found it *not* identical at the final step (max difference
>   0.014-0.033 in the text-attention family, larger in the style family),
>   which is consistent with the unconditional branch not reading style
>   directly while the shared latent trajectory diverges. Consistent, not
>   confirmed.
> - **The style-attention headline numbers (TV 0.283, effective rows 34.6)
>   blend both streams.** Per-stream figures are in "The style family" below.

`vector_estimator.onnx` declares one output, `denoised_latent`. It is not the
only thing the graph computes. This doc records a way to read two more things
out of it — a text/frame "alignment" (**withdrawn 2026-09-29, see above**) and
a time-varying view of `style_ttl` — with no retraining and no change to the
model's weights, what was measured doing so, and what that measurement does
and does not reach.

Read this alongside the CLAUDE.md project notes on `style_ttl` and `style_dp`
it corrects (see "Two premises this corrects," below) and alongside
[new-plan.md](../new-plan.md)'s Phase 2b/2a record.

**Read the calibration section before using any distance from this
instrument.** The readout is real, and the architectural facts below hold.
But the obvious way to use it — treat the style-row attention profile as a
per-style fingerprint and measure how far a perturbation moves it — was
calibrated against known-same and known-different pairs and **does not
separate them**. A whole different shipped voice moves the profile only 1.81x
as far as re-drawing the vocoder seed on the identical style. Everything in
the sections before "Calibration" was measured on a single render and should
be read as a description of the architecture, not as a per-style measurement.

## What was found, and how it was verified

`assets/onnx/vector_estimator.onnx` is a 1004-node graph. Its `graph.output`
list has one entry (`denoised_latent`), but the graph itself contains 8
`Softmax` nodes upstream of that output, computed and then discarded by the
exporter. A declared output list is the exporter's choice, not a property of
the weights underneath it — nothing stops those intermediate tensors from
being declared as outputs too, after the fact, on the existing file:

```python
import onnx

model = onnx.load(src_path)
softmax_outputs = []
for node in model.graph.node:
    if node.op_type == "Softmax":
        name = node.output[0]
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, None  # shape left dynamic
            )
        )
        softmax_outputs.append(name)
onnx.save(model, dst_path)
```

That is the whole trick: load, walk `graph.node` for `op_type == "Softmax"`,
append a `ValueInfoProto` for each match to `graph.output`, save. No graph
surgery beyond appending output declarations, no gradient, no `onnx2torch`
conversion of the kind the parametric-axis plan uses elsewhere in this repo.
ONNX Runtime happily returns whatever is listed in `graph.output`, including
tensors the original export never intended to expose.

Verified on this machine, one configuration: preset M1, seed 0, 8 denoising
steps, `speed=1.0`, text "The quick brown fox jumps over the lazy dog."
Predicted duration 3.25 s, giving `L=47` latent frames and `T=53` text units
(character-level, wrapped in literal `<en>`/`</en>` tag characters — see
"Word spans," below).

## Two families of attention, disambiguated by shape

Appending all 8 Softmax outputs and inspecting their shapes on that render
splits them cleanly into two families, and choosing text/latent lengths that
don't coincide (`L=47`, `T=53`, and `style_ttl`'s fixed 50 rows are three
different numbers) is what makes the split unambiguous rather than a guess:

| Nodes | Shape (this render) | Reads over |
|---|---|---|
| `main_blocks.{3,9,15,21}/attn/Softmax` | `(8, 2, 47, 53)` | **text positions** (53 = T) |
| `main_blocks.{5,11,17,23}/attention/Softmax` | `(2, 2, 47, 50)` | **style_ttl rows** (50, fixed) |

Two things to note about the naming: the two families sit in the same
`main_blocks` indices, offset by two (3/9/15/21 vs. 5/11/17/23), and the
op-type-level name (`attn` vs. `attention`) is what actually distinguishes
them — the shape is the confirmation, not the primary signal, since the shape
alone would still be ambiguous if the run happened to pick `T` equal to 50 or
53 equal to `L`. This render's numbers don't collide, and the module should
keep choosing text/latent lengths that don't, rather than hard-coding the
disambiguation to this one sentence's dimensions.

## The text family: alignment, and how the head is chosen

> **WITHDRAWN (2026-09-29).** The "alignment" below is a positional ramp, not
> a content alignment; see the correction at the top. The table is kept as
> originally reported. Read "near-perfect monotonic alignment" as "near-perfect
> monotone ramp". All figures here are from the final denoising step only.

The text family contains a near-perfect monotonic alignment between latent
frame index and the text unit it attends to most. Ranked by Spearman rho
between frame index and per-frame argmax text position, over (node, head,
stream) combinations on the verification render:

| Node | Head | Stream | Spearman rho |
|---|---|---|---|
| `main_blocks.9/attn` | 1 | 1 | **0.9995** |
| `main_blocks.9/attn` | 4 | 1 | +0.998 |
| `main_blocks.21/attn` | 3 | 1 | +0.993 |
| `main_blocks.3/attn` | 3 | 1 | +0.931 |
| `main_blocks.15/attn` | 7 | 1 | +0.908 |

Several (node, head, stream) combinations return `NaN` — their argmax is
constant across all frames, so rank correlation is undefined, not merely
weak. Every strong head in the table above sits at stream index 1; the
degenerate, constant-argmax heads sit at stream index 0. That split is
consistent enough on this render to be a useful filter, but it is one render
of one sentence in one voice, so **the alignment head is chosen at analysis
time by ranking Spearman rho, never hard-coded to a fixed (node, head,
stream) triple.** Whether stream 1 is reliably the informative half across
other voices, languages, or sentences is untested — the API should keep
selecting per render rather than assuming this render's winner generalizes.

*Added 2026-09-29:* the constant-argmax heads at stream 0 and the ramp heads
at stream 1 are what the classifier-free-guidance reading (conditional vs.
unconditional branch) would make unsurprising, but that reading is unconfirmed
(see the top of this doc), and selecting by Spearman rho picks the straightest
ramp, not the head that follows the text. The selection method is the reason
the withdrawn result came out as it did.

## Word spans: the worked example

> **WITHDRAWN (2026-09-29).** These are not timestamps. Against a 10 ms RMS
> envelope of the same render, speech ends at 2.64 s; the "dog" row below
> (2.79-2.86 s) sits after it, at -70.8 dB. The table is kept because it
> reproduces — it reproduces because it is the positional ramp.

Reading word boundaries off the winning head's per-frame argmax, at the
frame resolution implied by the vocoder's chunking (`base_chunk_size=512`
frames at `chunk_compress_factor=6`, giving 3072 samples per latent frame at
44.1 kHz, i.e. **69.66 ms/frame**):

| Word | Start (s) | End (s) |
|---|---|---|
| The | 0.35 | 0.49 |
| quick | 0.56 | 0.91 |
| brown | 0.91 | 1.18 |
| fox | 1.25 | 1.46 |
| jumps | 1.53 | 1.81 |
| over | 1.88 | 2.09 |
| the | 2.09 | 2.30 |
| lazy | 2.30 | 2.58 |
| dog | 2.79 | 2.86 |

"dog" reads as a single 69.66 ms frame — a real limit of this readout, not a
rendering quirk: boundaries quantize to the frame grid, so sub-frame timing
is not recoverable this way. (That single frame is also where the ramp runs
out: the final tokens are crowded onto the last frames, after the audio has
gone quiet.)

Text is tokenized character-level and wrapped in literal `<en>`/`</en>` tag
characters. `word_spans()` treats those tag characters as transparent, so
"The" starts at 0.35 rather than at 0.00: the frames before it are the
opening tag, which is not a word. Tag detection is by regex over the joined
token string, not a hardcoded "en", so it holds for other language codes.
Frames are half-open intervals `[f * frame_seconds, (f + 1) * frame_seconds)`
— without the `+1` on the end frame, a one-frame word like "dog" collapses to
a zero-width span and adjacent words stop sharing their boundary.

## The style family: `style_ttl` is read time-varyingly, not globally

This is the more consequential half for the parametric-axis work in
[new-plan.md](../new-plan.md). Each latent frame in the style-attention nodes
computes its own softmax distribution over the 50 rows of `style_ttl` — the
rows function as attention keys/values that vary per output frame, not as a
single conditioning vector read once per utterance. Measured on the same
render, averaged over the four style-attention nodes:

*Correction (2026-09-29), stated before the numbers: these figures are a blend
of two streams, and are from the final denoising step only.* `analyze()`
averages over axes `(0, 1)` — heads **and both streams**. The published
figures reproduce exactly, as a blend. Per stream on the same render
(audit-measured; not re-derived here):

| | blended (as published) | stream 0 | stream 1 |
|---|---|---|---|
| frame TV from utterance mean, mean | 0.283 | 0.418 | 0.255 |
| frame TV, range | 0.173-0.555 | 0.298-0.724 | 0.110-0.576 |
| effective rows, mean | 34.6 | 24.7 | 32.8 |
| effective rows, range | 10.4-43.6 | 5.1-35.1 | 6.0-43.1 |

The conclusion "`style_ttl` is read time-varyingly" **survives, and is stronger
in stream 0** (if stream 0 is the conditional branch, as the guidance reading
suggests, that is the more relevant one — unconfirmed). What does not survive
is treating 0.283 / 34.6 as a clean measurement of how `style_ttl` is read:
they describe an average of two streams that behave differently. The per-stream
entropy of the frame-averaged distribution was not measured. This dilution is
also an **untested candidate contributor** to the calibration failure below
(AUC 0.826), which was run on the blended statistic; no stream-0-only
calibration has been run.

The original blended figures, as first reported:

- **Total-variation distance** of each frame's row-attention distribution
  from the utterance-mean distribution: mean **0.283**, min 0.173, max
  0.555. (0 would mean every frame reads the same fixed mixture of rows —
  i.e., a genuinely global read.)
- **Effective row count** per frame, `exp(entropy)`: mean **34.6**, min
  10.4, max 43.6, out of 50 rows.
- **Entropy of the frame-averaged distribution**: **3.740 nats**, against
  3.912 nats for a uniform distribution over 50 rows — close to uniform on
  average, but the per-frame total-variation figure above shows that average
  is not what any single frame looks like.

**Limit, added after the calibration below.** These three numbers are a
property of the *architecture* — the rows really are attended to differently
at different latent frames, and that claim stands. They are not, on their
own, evidence that this distance can tell one *style* from another. Measured
directly (see "Calibration: does the style-attention distance separate
styles?" below), the same total-variation statistic computed between two
renders of the *same* style at two different vocoder seeds is nearly as
large as the same statistic computed between two *different* voices at one
shared seed — AUC 0.826 at the utterance level, with 80-90% of the two
distributions overlapping, and the frame-resolved version of the statistic
does worse than chance at the job (AUC 0.355) in its most direct form. Read
the numbers above as confirmation that the mechanism exists, not as a usable
per-style fingerprint.

What this means for the roadmap: CLAUDE.md's project notes already inferred,
from `style_ttl`'s dual consumption by `text_encoder` and `vector_estimator`,
that it has "per-position reach" rather than acting as a single timbre
vector — and bench 9 in
[LISTENING_BENCHES.md](LISTENING_BENCHES.md#9-phase-2a--preset-span-perturbation-same-voice-check)
heard that reach as changed word emphasis under a `style_ttl` perturbation.
This is the mechanism that inference was pointing at, now measured directly
rather than inferred from an audible side effect: the rows are attended to
differently at different latent frames, so a perturbation to a given row can
land more on some frames (and therefore some words) than others. It does not
by itself explain *which* rows drive *which* words — that mapping is not
measured here — but it confirms the reach is real and gives an instrument to
go look for it directly, on existing renders, with no new audio.

## Calibration: does the style-attention distance separate styles?

The section above establishes that the row-attention map is frame-varying.
It says nothing about whether the *distance* between two such maps can tell
two styles apart, which is the question that matters for using this
instrument as a readout rather than as an architectural curiosity. It was
calibrated once, after the fact, against exactly that question, using
`py/phase3_attention_ladder.py` (committed `e7f3c12`; reproduce with
`python3 py/phase3_attention_ladder.py --force-ladder --presetspan
--eps-ladder`). 627 renders total, primary text "The quick brown fox jumps
over the lazy dog.", `speed=1.05`, `style_dp` pinned to M1's so every render
in the comparison shares `L=45` (duration 3.0997 s).

**It failed. That is the headline, not a caveat below one.** *(Added
2026-09-29: two candidate explanations for why are recorded, neither tested:
the statistic averages two streams that behave differently, see the per-stream
table above and no stream-0-only calibration has been run; and, from
[STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md), that addressing over the
50 slots is style-independent — but only in 2 of the 6 style-reading nodes, so
that account is weaker than first stated. Both are candidates, not findings.)*
The attention
maps depend on `noisy_latent` as well as on `style_ttl`, and
`sample_noisy_latent()` draws an unseeded `np.random.randn` on every call
(see the CLAUDE.md bullet on unseeded vocoder sampling) — so a substantial
part of what these statistics measure is which seed a render happened to
draw, not which style it was given.

### Reference distributions

NULL — one style (`base_ttl`, M1's `style_ttl`) rendered at seeds 0-23, all
276 pairs — against KNOWN-DIFFERENT — the 10 shipped presets' `style_ttl`,
with `style_dp` substituted for M1's so `L` stays fixed, same seed, all 45
pairs:

| Statistic | NULL (same style, different seed) | KNOWN-DIFFERENT (different style, same seed) | AUC |
|---|---|---|---|
| utterance row-profile TV | 0.0328 +/- 0.0114 [0.0141, 0.0725] | 0.0594 +/- 0.0310 [0.0192, 0.1372] | 0.826 |
| frame-resolved TV | 0.1479 +/- 0.0267 | 0.1359 +/- 0.0547 | 0.355 (wrong direction) |

At the utterance level, 90% of same-style pairs and 80% of different-style
pairs sit inside the two distributions' shared overlap interval, and the
ratio of means is 1.81. A Mann-Whitney test on the utterance statistic gives
p=1.2e-12 — **the means differ at high significance, and the distributions
still overlap on 80-90% of their mass.** Significance of a mean difference is
not separation; this is a clean instance of the trap CLAUDE.md's calibration
bullet names.

The frame-resolved statistic does worse than chance at this task in its
unmatched form (AUC 0.355): two *different* presets rendered at *one* shared
seed are, frame by frame, on average *closer* to each other than one preset
rendered at *two* different seeds is to itself. In this design it is
measuring which seed was drawn, not which style was given.

### Matched variant

The design above is not seed-matched — KNOWN-DIFFERENT holds the seed fixed
while NULL varies it, which is what produces the frame-resolved AUC of
0.355. Re-run so both distributions carry one seed change: AUC improves and
still does not separate.

| Statistic | AUC | Fraction of KNOWN-DIFFERENT inside NULL's spread |
|---|---|---|
| utterance TV | 0.950 | 80% |
| frame TV | 0.875 | 98% |

No threshold separates the two distributions in either the matched or the
unmatched variant. The floor is not specific to M1: measured on the 10 shipped
presets (each rendered at seed 0 vs. seed 1), the same-style floor is
utterance TV 0.0378 +/- 0.0132, frame TV 0.1732 +/- 0.0103 — consistent with
the M1-only numbers above. (*Corrected 2026-09-29:* an earlier version of this
sentence said "the other 10 shipped presets". The 10 shipped presets include
M1, the base, so this is a 10-preset check with M1 in it, not 10 independent
voices beyond M1.)

### Three further floors

Statistics that are not distances on the row profile have the same problem:

- **Word boundaries**, same style across seeds: max shift 109 ms mean
  (median exactly one frame, up to 3 frames), mean shift 41.6 ms.
  *(Withdrawn with the word spans, 2026-09-29: these are shifts in a
  positional ramp's boundaries, not in where words are spoken. It is also an
  unpaired floor — see the correction below on comparing paired shifts to it.)*
- **Active-row mass** (the 24 rows a preset-span perturbation touches), same
  style across seeds: sd 0.0088, range 0.342-0.376. Between different
  presets at one seed: sd 0.017, range 0.309-0.366 — a different voice
  barely exceeds seed noise on this statistic.
- **`effective_rows`**, same style across seeds: 34.98 +/- 0.79.

### The one positive, and its caveat

One contrast does replicate across a seed change: at matched Frobenius norm,
perturbations confined to the 9-dimensional preset-span subspace move the
row-attention profile roughly 2.1x as far as random-direction perturbations
of the same size.

| | preset-span | random-direction | AUC | p |
|---|---|---|---|---|
| utterance TV (seed 0) | 0.0229 | 0.0107 | 0.874 | 6.5e-7 |
| frame TV (seed 0) | 0.0731 | 0.0294 | 0.919 | 2.6e-8 |
| utterance TV (seed 1) | 0.0223 | 0.0114 | 0.819 | 2.3e-5 |
| frame TV (seed 1) | 0.0690 | 0.0281 | 0.830 | 1.2e-5 |

`row_matched_random` — a control matched to preset-span's per-row energy
profile but not its subspace direction — lands with `random_control`, at
0.0115 / 0.0342, so the gap is the *subspace direction*, not the row-energy
distribution. *(Corrected 2026-09-29: an earlier version said "group means
reproduce to within 3% across the seed change". Of the four seed-0 vs. seed-1
pairs above, only the preset-span utterance TV is under 3% (2.6%); the random
control's utterance TV differs by 6.3%, and the frame TVs by 5.6% and 4.4%.)*
This recapitulates the earlier probe's 0.9376 vs. 0.7593 preset-span R^2 gap
([new-plan.md](../new-plan.md)) in a readout that needs no new audio and no
embedding model.

**The caveat, and its correction (2026-09-29).** This is a group-mean
contrast. The original text went on to say it "sits below the per-sample floor
measured above — preset-span's own per-sample utterance TV (mean 0.0229) is
inside the same-style floor's own spread", and concluded the instrument has
"group resolution, not sample resolution". **That comparison is the same error
as the "0.26x the floor" reading corrected below.** 0.0229 is a shared-seed
*paired* TV (perturbed against base at one vocoder seed; its own null is
zero). The floor it was set against, 0.0328 +/- 0.0114, is *unpaired* (one
style at two different seeds). A paired statistic cannot be placed "inside"
an unpaired floor's spread. So the claim that the limit is group resolution,
not sample resolution, **is not supported by the comparison it rested on, and
the true limit is unestablished** — neither "below the floor" nor "above it" is
shown. What is on record: a single-seed paired difference for the K64 ladder
styles is mostly seed-specific (SNR 0.151, see the correction below), which
is separate evidence, on a different perturbation family, that one paired
render is noisy. The unpaired calibration failure above stands as a statement
about the unpaired design.

### Verdict

The row-attention distance, at either resolution, does not separate two
styles from two seeds of one style on a per-sample basis, in any *unpaired*
variant tested (all on the stream-blended statistic; no stream-0-only variant
has been run). It preserves one group-level contrast (preset-span vs.
random-direction perturbations) across a seed change. Do not use it as a
per-style discriminator, an audibility proxy, or a substitute for a
listening test or a calibrated embedding distance. The forced ladder run
past this failed gate — magnitude dependence, timing, and why an
active-row-targeting finding did not survive its own controls — is recorded
in [new-plan.md](../new-plan.md)'s 2026-09-24 entry rather than repeated
here.

### Correction (2026-09-25): the ladder's "0.26x the floor" reading was invalid

**This is a correction to a conclusion drawn elsewhere in this project, not
to any number in this section.** Task #18 (`new-plan.md`'s 2026-09-24 entry,
committed `e7f3c12`, documented `240f17c`) took the forced ladder's
total-variation distance from base — 0.00846, mean over 240 samples,
`ttl_K4`/`K16`/`K64` — and read it as "0.26x the seed floor," i.e. as
evidence the perturbation did not move attention measurably. **That
comparison is invalid, independent of whether the perturbation moved
anything.**

The 0.00846 figure was computed with base and perturbed both rendered at a
**shared vocoder seed** — a paired, common-random-numbers design. With the
rendering pipeline deterministic under a seed (confirmed directly: a shared
seed with `style_dp` pinned gives an identical `noisy_latent` whatever the
style is), that design's own null is **exactly zero**, not 0.0328. The
0.0328 figure quoted as "the floor" above is the *unpaired* NULL from this
section's own calibration — `base_ttl` against itself at two *different*
seeds. Dividing a paired statistic by an unpaired floor compares two
different null distributions; the ratio it produces is not a signal-to-noise
number for either design.

`py/phase3_seed_averaging.py` (committed `e9ed1db`, 1410 renders) measured
mean paired TV **0.00840**. *Correction (2026-09-29): this section originally
said "The number replicates. The framing around it did not." The first half is
not like-for-like.* 0.00846 was the pooled mean over `ttl_K4` + `K16` + `K64`
(0.00647 / 0.00886 / 0.01004, n=80 each, seed 0). 0.00840 is `ttl_K64` only
(24 styles x 32 seeds). The like-for-like `K64` figure from task #18 is
0.01004, against 0.00840 here. And `seed_robustness.json` shows paired TV moving
about 2x with seed alone (K4: 0.00616 at seed 0, 0.01183 at seed 1). So the
two numbers agree in order of magnitude, which is all the data supports. The
paired-vs-unpaired framing was the error and that part of the correction
stands.

Read paired, against its own (zero) null, the perturbation signal is real
and reproducible:

- mean cosine between the disjoint-seed-half mean-difference vectors:
  **0.726 +/- 0.208** (median 0.802, p05 0.398, n=24 ladder styles)
- per-style paired-TV ranking survives an independent seed-half split:
  Spearman **0.587**, p=0.0026
- all 24 ladder styles show positive unbiased signal energy; none
  non-positive
- pairing is worth **14.75x** in noise energy relative to an unpaired
  comparison — one paired render buys what roughly 15 unpaired renders do

**What does not change:** SNR at a single seed is **0.151** (range
0.040-0.391 across the 24 styles), so 87% of the paired difference's energy
is still seed-specific at N=1. That is the same instability that made the
active-row finding above flip sign at a second seed — that warning is
unchanged by this correction. What changes is the reason: not that the
perturbation fails to move attention, but that a single seed cannot see past
its own noise.

**Also unchanged:** the unpaired calibration above (AUC 0.826
utterance-level, AUC 0.355 frame-resolved, NULL vs. KNOWN-DIFFERENT) is
still correct, exactly as stated, as a description of *that* design.
Comparing two styles in a *paired* design separates trivially at N=1,
because a paired null is zero by construction — that is a property of
pairing, not evidence against the unpaired calibration failure above, and
not a reason to prefer the paired distance as a general-purpose style
discriminator (see the seed-averaging section below for why pairing does not
by itself make the instrument cheap to use).

See "Seed-averaging" below for the full measurement this correction is drawn
from, and for the cost of trying to use either the paired or the unpaired
design as an actual per-style instrument.

## Seed-averaging (2026-09-25): does averaging over seeds buy back resolution?

The calibration above measured the row-attention distance's noise floor at a
single seed. `new-plan.md`'s 2026-09-24 entry asked the obvious next
question — whether averaging the distance over many seeds per style shrinks
that floor enough to make it a usable per-style discriminator — and flagged
it as the cheapest untested option. `py/phase3_seed_averaging.py` (committed
`e9ed1db`) answers it: 1410 renders, primary text, `speed=1.05`, `style_dp`
pinned to M1's so `L=45` is constant across every render, 0.821 s/render
measured. Sanity checks confirmed on the run: `base_ttl` equals M1 exactly,
same-seed renders are bit-identical, and a shared seed with `style_dp`
pinned gives an identical `noisy_latent` whatever the style is.

### Part A: 10 shipped presets, unpaired N-seed averages

64 seeds per preset. "Same" is two disjoint N-seed averages of one preset;
"different" is N-seed averages of two different presets on disjoint seed
sets. Criterion stated in advance: AUC >= 0.99 **and** disjoint central-95%
intervals.

| N | same mean +/- sd | different mean +/- sd | AUC | separates |
|---|---|---|---|---|
| 1 | 0.03542 +/- 0.01529 | 0.06139 +/- 0.01757 | 0.875 | no |
| 2 | 0.02468 +/- 0.01003 | 0.05593 +/- 0.01660 | 0.955 | no |
| 4 | 0.01707 +/- 0.00703 | 0.05160 +/- 0.01399 | 0.991 | no (intervals overlap) |
| 8 | 0.01248 +/- 0.00520 | 0.05035 +/- 0.01428 | 0.998 | yes |
| 16 | 0.00878 +/- 0.00361 | 0.04930 +/- 0.01366 | 1.000 | yes |
| 32 | 0.00602 +/- 0.00252 | 0.04856 +/- 0.01324 | 1.000 | yes |

At the N=8 "yes," the full min/max ranges still overlap — 36% of
different-pairs sit inside the same-style range — and only N=32 clears the
stricter full-range test (0% overlap either direction). Both flags are in
`part_a_presets.json`.

**No bias floor.** The same-style distance is pure averagable noise, not
evidence of a floor averaging cannot cross: a power-law fit gives exponent
**-0.5096** at R^2=0.99959 (N^-0.5 is the signature of pure averaging
noise), and a free two-parameter fit `a/sqrt(N) + c` puts the asymptote at
**c = 1.4e-5**, bootstrap CI95 `[1.3e-6, 2.8e-4]`. *(Corrected 2026-09-29: an
earlier version said "indistinguishable from zero and 30-600x below the
ladder's own 0.00846".)* The interval strictly excludes zero, so the accurate
statement is "consistent with zero at the ladder's scale", not "is zero". The
30-600x arithmetic is right against 0.00846 (upper bound and point estimate),
but 0.00846 is a pooled paired mean and not the relevant comparator; the
ladder's noise-free separation is 0.00301 (Part B below), which the asymptote
sits ~11x below at the CI upper bound (~215x at the point estimate). So
"averaging cannot get there at any N" is false; it can. What kills the approach is the price, not
a floor (see "Cost," below). The different-preset side converges to an
asymptotic separation of 0.0483, matching the direct full-64-seed
computation at 0.0482 +/- 0.0134.

### Part B: the ladder itself, unpaired N-seed averages

24 `ttl_K64` styles (eps=0.2, K=64, from `phase2b_subspace`) x 32 seeds each,
against base x 64 seeds — the same unpaired design as Part A, not the paired
design the correction above uses.

| N | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| AUC | 0.507 | 0.528 | 0.452 | 0.502 | 0.568 | 0.664 |

Never leaves the coin-flip region until after N=16; does not separate at 32.
Fitted asymptotes: null ~0.000145 (~0), ladder 0.00301 — a perturbation buys
about 1/16 of what a different shipped voice buys (0.00301 vs. 0.0483 from
Part A). The AUC=0.452 dip at N=4 is a condition-matching artifact, not a
signal: ladder styles are 9.3% less seed-noisy than base (per-render noise
energy ratio 0.907), which deflates the test side by a predicted ~2.4%.
Worth recording generally: with this instrument a style's across-seed
*variance* differs by style, so an AUC built from distances is not
automatically noise-matched even when averaging depth is matched on both
sides.

Extrapolated N to separate, at the order the pooled criterion would need:
an emulator fit gives N ~ 512, bias-corrected to ~1330 (its asymptote
overstates the directly-fitted one by 1.61x, and N scales as 1/TV^2 so that
overstatement compounds); an independent analytic route (disjoint-half
cross-product) gives a median of 290 per style, p95 1093, and 4747 for the
weakest of the 24 — the pooled criterion is governed by that tail. Frobenius
distance from base is constant across the 24 styles (0.96537 +/- 0.00003),
so the spread in required N is **not** a magnitude effect.

*Correction (2026-09-29): this paragraph originally ended "equal-sized
perturbations differ by roughly 70x in how much attention movement they
produce". The 70x was never an observed movement spread.* It is the spread in
extrapolated required N (69.8 to 4747, 68x). Because N scales as 1/TV^2, the
implied spread in movement is about 8.25x (the square root). Measured
directly on the same 24 styles: per-style paired-TV mean varies 1.76x
(0.00645-0.01135) and unbiased signal energy varies 15.7x (5.73e-7 to
9.02e-6). Three different quantities, three different spreads; none of them is
70x. The proposed explanation for the spread — the spectrum of `W_value` — was
tested and falsified; see [STYLE_CONTROL_SURFACE.md](STYLE_CONTROL_SURFACE.md).

### Cost, and the verdict

At 0.821 s/render (measured), scoring the existing 240-sample ladder:

| N | 1 | 32 | 512 | 1332 (bias-corrected) |
|---|---|---|---|---|
| cost | 3.3 min | 1.8 h | 28 h | 73 h |

The full 1280-sample corpus at N=1332 costs 389 h, about 16 days of CPU.
Alternatives already in this repo: a WavLM probe pass (1.57 s render + 0.95
s embed = 2.52 s/sample) covers the full 1280-sample corpus, one render
each, in 0.90 h — roughly 430x cheaper at the same corpus size. A human
listening bench covers ~20 clips (31 s of rendering) plus 10-20 minutes of a
listener's time; it does not scale past ~20 samples, but it answers a
higher-authority question than any distance measured here. The paired
design from the correction above is the cheapest route to a usable signal:
14.75x variance reduction measured directly, and reaching a 2:1
signal/noise amplitude margin needs a median of 32 paired seeds (64
renders) per sample — 3.5 h for 240 samples, 18.5 h for 1280, **8-21x**
cheaper than unpaired averaging (8.1x against unpaired N=512: 28.1 h / 149.5 h
vs. 3.47 h / 18.5 h; 21x against N=1332: 73.2 h / 389 h). *(Corrected
2026-09-29: an earlier version said "20-40x"; nothing sources 40x.)*

**Verdict: do not build unpaired seed-averaging.** Even the paired variant
costs about 20x a WavLM pass to deliver a statistic nobody has calibrated
against audibility. The one defensible use is narrow and cheap: a paired
2-render A/B to confirm a perturbation moved attention at all — 6.6 min for
240 samples — carrying the caveat that 87% of a single seed's paired
difference is still seed-specific.

## Using the module and CLI

`py/` is not a package, so import the way the tests do — put `py/` on the
path first.

```python
import sys; sys.path.insert(0, "py")
from attention import export_attention_graph, analyze, DEFAULT_ATTENTION_PATH

export_attention_graph("assets/onnx/vector_estimator.onnx", DEFAULT_ATTENTION_PATH)

wav, alignment, style_attn = analyze(
    tts, text="The quick brown fox jumps over the lazy dog.",
    lang="en", style=style, total_step=8, speed=1.0, seed=0,
)

alignment.word_spans()          # the table above -- NOT validated timestamps (withdrawn 2026-09-29)
alignment.spearman               # the winning head's rho (0.9995 on this render): monotonicity, not content alignment
style_attn.frame_divergence()    # per-frame total-variation distance from the mean
style_attn.effective_rows()      # per-frame exp(entropy)
```

`export_attention_graph(src_path, dst_path)` is idempotent — calling it again
with the instrumented file already in place does not re-instrument it a
second time — and returns the list of Softmax output names it appended.
`analyze()` re-implements the denoising loop itself (rather than driving
`helper.TextToSpeech` as a black box) so it can request the extra outputs at
each step; `Alignment` exposes `.matrix` (L×T), `.tokens`, `.frame_seconds`,
the winning `.node`/`.head`/`.stream`/`.spearman`, `.duration`, and the
`.token_spans()`/`.word_spans()` derived tables; `StyleAttention` exposes the
raw `.matrix` (L×50), `.frame_seconds`, and the `.row_weights()` /
`.frame_divergence()` / `.effective_rows()` summaries used above. A CLI wraps
the same call (`--text --voice --lang --seed --steps --speed --onnx-dir`); as
of 2026-09-29 it prints a warning above the word table that the spans are not
validated timestamps. All readouts are from the final denoising step, and
`StyleAttention.matrix` averages heads and both streams.

## Cost: a 256 MB cache, gitignored

`export_attention_graph` writes its instrumented copy to
`assets/onnx/vector_estimator.attn.onnx` by default
(`DEFAULT_ATTENTION_PATH`), roughly 256 MB — the original graph plus its
weights, duplicated, since appending output declarations does not let ONNX
share storage with the un-instrumented file. This is not meant to be
checked in: `assets/` and `*.onnx` are both already gitignored in this repo
(confirmed against `.gitignore`), the same way the shipped model files under
`assets/onnx/` are. Run the export once per machine; it is idempotent, so
repeated calls in a session or a test suite do not re-pay the cost.

## The drift guard

`py/attention.py` re-implements the denoising loop rather than calling
`helper.TextToSpeech` and only reading extra outputs off the side, because
ONNX Runtime's `graph.output` list is set at session-creation time and the
production code path does not request the Softmax tensors. That
re-implementation is exactly the kind of mirror this repo has been bitten by
before — `Style.with_deltas` in `py/helper.py`, `withDeltas` in
`web/helper.js`, and `with_deltas_np` in the Colab `style_tools.py` mirror
have diverged once already (the browser path normalized `style_ttl` over the
whole 50×256 block instead of per row, silently, until it was caught). The
same risk applies here: a second copy of the sampling loop can drift from
`helper.py`'s without anyone noticing until the audio sounds wrong.
`py/test_attention.py`'s load-bearing test is the guard against that — it
asserts that the waveform `analyze()` returns is **bit-identical** to
`helper.TextToSpeech`'s own render at the same seed, on the same inputs. If
that test passes, the alignment and style-attention readouts are being taken
from a denoising loop that produces the exact audio the production path
would, not an approximation of it that happens to sound similar. *(Added
2026-09-29: this is the part of the module that is verified. The guard checks
that the audio is right and the maps are read from that render; it says nothing
about whether the head `_select_best_alignment` picks tracks text. The test
suite's own rho check, `spearman > 0.95`, is satisfied by a positional ramp.)*

Run it, along with the rest of the blending tests, with:

```bash
python3 -m unittest discover -s py -p "test_*.py"
```

## What this does NOT reach

Worth stating plainly, because the finding above is easy to over-claim in
either direction.

- **It is not an identity, audibility, or "did the perturbation land"
  meter.** Calibrated against 276 same-style pairs (one style, 24 vocoder
  seeds) and 45 known-different pairs (10 shipped presets, one shared seed):
  the utterance-level row-attention distance separates them with AUC 0.826,
  but with 90% of same-style pairs and 80% of different-style pairs sitting
  inside the shared overlap region — a seed change moves this statistic
  almost as much as swapping the voice does (same-style floor to
  between-preset ratio: 1.81x). The frame-resolved version of the same
  statistic is worse than a coin flip at this job in its unmatched form, AUC
  0.355 — it points the wrong way, because two different voices at one seed
  sit frame-by-frame *closer* together than one voice does across two seeds.
  See "Calibration: does the style-attention distance separate styles?"
  above before reaching for either statistic as a cheap substitute for a
  listening test or an embedding distance.
- **It does not give word timestamps, of the model's own audio or of anything
  else (withdrawn 2026-09-29).** The original bullet here said this "aligns
  the model's own generated audio to the text it was given". It does not: see
  the top of this doc. It also does not align an external recording. Forced
  alignment of a human
  recording (matching real audio to a transcript) is a different problem
  that still needs an analysis model such as the Montreal Forced Aligner or
  a CTC-based aligner. Nothing here substitutes for that.
- **It does not enable transcription.** The alignment answers *where* a
  known text unit lands in the generated audio, given that text as an input
  to the graph. Recovering *which* text produced an unknown audio clip is
  posterior inference over a discrete space, and needs an evaluable
  likelihood (blocked here — the vocoder has no known inverse) and a
  proposal distribution over token sequences, which is a speech recognizer,
  i.e., the thing this readout would have to be built into, not a thing it
  already is. Use an existing ONNX ASR model (Whisper, Moonshine) for that
  task; this instrument is not a substitute and was not evaluated as one.
- **Generality is untested.** Every number in this doc comes from one voice
  (M1), one language (en), one sentence, and one seed. The stream-1 /
  stream-0 split, the specific winning nodes, and the style-attention
  statistics have not been checked against a second voice, a second
  language, or a longer or more complex sentence. Treat the mechanism
  (Softmax outputs exist and can be exposed; the style family varies per
  frame) as established — *"the text family aligns" was struck 2026-09-29* —
  and every specific number above as one data point rather than a constant of
  the model. All of it is from the final denoising step; earlier steps were
  not read.

## Provenance of the numbers in this doc

The figures outside the calibration section were produced by instrumented
inference against `assets/onnx/vector_estimator.onnx` on CPU — preset M1,
seed 0, 8 steps, `speed=1.0` — and independently reproduced by running
`py/attention.py`'s CLI after the module was written. The two runs agree on
the chosen node, head and stream, on all nine word spans, and on both
style-attention summaries. *(2026-09-29: that agreement shows the readout is
reproducible, not that it is valid — the ramp reproduces exactly. Both
style-attention summaries are stream-blended.)*

The calibration section's figures come from a separate and larger run,
`py/phase3_attention_ladder.py` (627 renders, `speed=1.05` and `style_dp`
pinned to M1's, to match the corpus it was read against), and are re-readable
from `py/results/phase3_attention_ladder/*.json` without re-rendering. The
speed difference is why the single-render numbers above and the calibration's
own base render are not directly comparable; each is internally consistent.

Two figures were corrected during that reconciliation, and the originals are
recorded here so the earlier commits can be read honestly:

- The winning head's Spearman rho is **0.9995**, not `+1.000`. The first
  measurement was printed at three decimal places, where 0.9995 displays as
  `+1.000`; it was quoted from that display. The alignment is very nearly
  monotonic, not exactly so.
- The first word's span is **0.35–0.49 s**, not `0.00–0.49`. The original
  ad-hoc script counted the `<en>` tag's frames into the first word.
  `word_spans()` treats tag characters as transparent, which is the correct
  behaviour and changes the number.

`py/test_attention.py` is what re-checks these mechanically. The doc is a
record, not the verification.

The "Correction" and "Seed-averaging" sections above come from
`py/phase3_seed_averaging.py` (1410 renders, `speed=1.05`, `style_dp` pinned
to M1's, primary text), re-readable from
`py/results/phase3_seed_averaging/*.json` (`summary.json`,
`part_a_presets.json`, `part_b_ladder.json`, `part_c_cost.json`) without
re-rendering.
