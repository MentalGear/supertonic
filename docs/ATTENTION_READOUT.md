# Attention Readout

`vector_estimator.onnx` declares one output, `denoised_latent`. It is not the
only thing the graph computes. This doc records a way to read two more things
out of it — text/frame alignment and a time-varying view of `style_ttl` — with
no retraining and no change to the model's weights, what was measured doing
so, and what that measurement does and does not reach.

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

## Word spans: the worked example

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
is not recoverable this way.

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

**It failed. That is the headline, not a caveat below one.** The attention
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
unmatched variant. The floor is not specific to M1: measured on the other 10
shipped presets (each rendered at seed 0 vs. seed 1), the same-style floor is
utterance TV 0.0378 +/- 0.0132, frame TV 0.1732 +/- 0.0103 — consistent with
the M1-only numbers above.

### Three further floors

Statistics that are not distances on the row profile have the same problem:

- **Word boundaries**, same style across seeds: max shift 109 ms mean
  (median exactly one frame, up to 3 frames), mean shift 41.6 ms.
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
distribution. Group means reproduce to within 3% across the seed change.
This recapitulates the earlier probe's 0.9376 vs. 0.7593 preset-span R^2 gap
([new-plan.md](../new-plan.md)) in a readout that needs no new audio and no
embedding model.

**The caveat that matters:** this is a group-mean contrast, not a per-sample
one, and it sits below the per-sample floor measured above — preset-span's
own per-sample utterance TV (mean 0.0229) is inside the same-style floor's
own spread. The instrument has group resolution — average many samples and
the preset-span direction stands out from a random one — but not sample
resolution: given a single rendered pair, this distance cannot tell you
whether it came from two styles or from one style at two seeds.

### Verdict

The row-attention distance, at either resolution, does not separate two
styles from two seeds of one style on a per-sample basis, in any variant
tested. It preserves one group-level contrast (preset-span vs.
random-direction perturbations) across a seed change. Do not use it as a
per-style discriminator, an audibility proxy, or a substitute for a
listening test or a calibrated embedding distance. The forced ladder run
past this failed gate — magnitude dependence, timing, and why an
active-row-targeting finding did not survive its own controls — is recorded
in [new-plan.md](../new-plan.md)'s 2026-09-24 entry rather than repeated
here.

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

alignment.word_spans()          # the table above
alignment.spearman               # the winning head's rho (0.9995 on this render)
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
the same call (`--text --voice --lang --seed --steps --speed --onnx-dir`).

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
would, not an approximation of it that happens to sound similar.

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
- **This aligns the model's own generated audio to the text it was given —
  it does not align an external recording.** Forced alignment of a human
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
  (Softmax outputs exist and can be exposed; the text family aligns; the
  style family varies per frame) as established, and every specific number
  above as one data point rather than a constant of the model.

## Provenance of the numbers in this doc

The figures outside the calibration section were produced by instrumented
inference against `assets/onnx/vector_estimator.onnx` on CPU — preset M1,
seed 0, 8 steps, `speed=1.0` — and independently reproduced by running
`py/attention.py`'s CLI after the module was written. The two runs agree on
the chosen node, head and stream, on all nine word spans, and on both
style-attention summaries.

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
