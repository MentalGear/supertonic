# Glitch Mitigation

Supertonic produces audible artifacts ("hiccups") on its own, with no style
perturbation involved. This doc records what was measured about that this
session, what was tried and contradicted, what has a partial fix in place,
and what is proposed but untested. Read
[LISTENING_BENCHES.md](LISTENING_BENCHES.md) alongside this — bench 6 is the
listener evidence this doc starts from, bench 7 is the baseline rate
measurement it points to (now answered), and bench 8 is the speed
root-cause investigation behind the fix below (partially amended by bench 7 —
see below).

## The finding

Bench 6 (Phase 2a — Capacity) asked a listener to judge nine seed-matched,
level-matched triples (true style / probe prediction / unperturbed control).
The listener annotated three glitches, and **two of the three were in
CONTROL clips** — a shipped preset (M1), unperturbed `style_ttl` and
`style_dp`, the ordinary pipeline with nothing under test. They called the
same preset rendering the same sentence "hiccupy" at one vocoder seed and
"normal and the best" at another:

- woodchuck sentence, M1, seed `20261069`: *"'chuck' after woodchuck sounds
  condensed/hiccuped in control"*
- woodchuck sentence, M1, seed `20261295`: *"control sounds normal and the
  best"*
- seashells sentence, M1, seed `20261449`: *"control has the most prominent
  case where the final word 'morning' is pronounced too quickly so it
  sounds like a hiccup"*

`sample_noisy_latent()` in `py/helper.py` draws its initial latent with
`np.random.randn` and no seed. Upstream, every render is therefore a
different draw, and a user who gets a good render or a bad one has no way to
reproduce it or re-roll it — the glitch is a property of the draw, not of
the style tensor or the text.

## What the seed actually moves

**Measured**, by `py/phase2a_timing_drift.py`
(`py/results/phase2a/timing_drift.json`). Holding style fixed and varying
only the vocoder seed (396 pairs pooled from the 12-seed-per-condition sets
in `py/results/listening_sets/phase2a_seed_variance/`, 4,356 word-boundary
comparisons), word start times move mean 0.041 s, p95 0.12 s, max 0.30 s,
while total utterance duration stays pinned to the sample across every seed
within a condition (`style_dp` fixes duration; the seed cannot move it).
Style-induced drift at a *fixed* seed — the same measurement applied to the
9 bench-6 triples, true/pred vs. control — is smaller on every statistic:
mean 0.015 s, p95 0.06 s, max 0.24 s. Only 1 of 9 triples exceeds the
sampling-floor p95 at all.

Whisper itself is deterministic on this audio: six files transcribed twice
each produced bit-identical timestamps on all 74 word pairs, so it
contributes no noise of its own. **The vocoder is the whole noise source**
in this comparison — an unseeded render moves word timing more than a
change in style does.

## Detectors: three failed, one works partly

Three single-render and one style-based detector were tried against the
bench-6 listener verdicts. All are **measured**, not proposed.

- **Log-mel spectral flux on a single render FAILS.** Thresholded at the
  99th percentile of per-frame flux pooled over 480 stock renders across
  all 10 presets and 8 texts (`py/phase2a_baseline_artifact_rate.py`,
  `py/results/phase2a/baseline_artifact_rate.json`), it flags 98.5% of all
  stock renders regardless of preset or text — it is firing on ordinary
  consonant transients, not glitches. Worse, its ordering is **reversed**
  against the listener on both bench-6 control pairs: recomputed peak flux
  is 300.3 for the clip called "hiccupy" (woodchuck, seed 20261069) vs.
  306.0 for the clip called "normal and the best" (seed 20261295) — the
  glitchy clip scores *lower*; and 182.8 for the "hiccup" seashells clip
  (seed 20261449) vs. 281.8 for its clean counterpart (seed 20262075) — the
  same direction, more pronounced. Anti-correlated with the listener on both
  available pairs. Unusable (`py/phase2a_seed_variance.py`,
  `detector_validation_against_listener`: 2/4 matches, `detector_reliable:
  false`).
- **Per-word log-mel difference (`py/phase2a_word_diff.py`) FAILS its
  two-sided validation.** It half-finds the flagged words by mean
  difference, but also scores two triples the listener could not
  distinguish from each other at or above triples the listener could
  distinguish — it does not separate the audible cases from the inaudible
  ones.
- **Timing drift as a style signature FAILS.** As above: style-induced
  drift at fixed seed (mean 0.015 s) sits under the seed-sampling floor
  (mean 0.041 s), so a detector built on style-driven timing change alone
  cannot see glitches — the seed itself is a bigger timing effect than the
  thing that was supposed to be the signal.
- **Cross-seed disagreement WORKS as a word-level diagnostic** — the one
  detector that reproduces the listener's flags.
  `py/phase2a_cross_seed_disagreement.py` holds text and style fixed and
  ranks each word by its spread (duration, start time, per-word log-mel)
  across 12 vocoder seeds. Across the 6 conditions x 12 seeds analyzed:
  - **woodchuck** ('wood', 'chuck' — the flagged affricate pair): rank 1st
    and 2nd of 13 words in 2 of the 3 woodchuck conditions tested (the
    unperturbed control, and a true-style condition); in the third
    (K4/typical) they rank 3rd and 4th, behind the modal 'could' and
    pronoun 'it'. The sentence-final **'wood?'** — separately flagged by
    the listener under a *true* style at seed 20261295 — rises from rank 5
    of 13 in the control ranking to rank **3 of 13** in the true-style
    ranking where that seed is also the single biggest cross-seed outlier
    for that condition.
  - **seashells** ('morning.' — the flagged word): ranks **2nd of 9** words
    in all 3 seashells conditions tested (control and two true-style
    conditions). 'seashore' ranks 1st in all three; it was not separately
    flagged by the listener, but its behavior is consistent with the same
    mechanism.
  - Function words (`'How'`, `'would'`, `'if'`, `'much'`) cluster toward the
    bottom of the ranking in every condition checked.

The structural reason: a hiccup has no absolute signature distinguishable
from an ordinary consonant transient inside one clip — which is exactly why
the single-render detectors above fail. An **unstable** word varies across
independent draws of the vocoder's noisy latent; a well-determined one does
not. Disagreement across draws is visible where no single draw carries a
usable signal.

## What was tried and contradicted

**Medoid-of-N selection** — render N draws from one style, keep the one
whose whole-utterance log-mel is nearest (frame-aligned mean |Δ|) to all the
others, discard the rest — does **NOT** work as tested. Using the same
cross-seed distance data from `phase2a_cross_seed_disagreement.py`
(`seed_distance_ranking`, medoid first / outlier last, out of 12 seeds), in
3 of 4 directional checks against the two bench-6 pairs, the seed the
listener called **clean** was the more extreme outlier, not the medoid:

| condition | seed | listener verdict | rank (1=medoid, 12=outlier) |
|---|---|---|---|
| woodchuck control | 20261069 | hiccupy | 8/12 |
| woodchuck control | 20261295 | clean, best | 11/12 |
| seashells control | 20261449 | hiccupy | 8/12 |
| seashells control | 20262075 | clean | **12/12** (biggest outlier) |

Cross-seed distance localizes *which words* are unstable; it does not rank
*which render* is good — on this evidence it points the wrong way about as
often as the right one. Two labelled pairs is weak evidence, so this is not
disproven outright, but it must not be built on as it stands. Revisiting it
would need either a labelled set large enough to test the direction
properly, or a distance restricted to the unstable words identified above
rather than the whole utterance (whole-utterance distance is dominated by
frames that carry no instability at all, which is a plausible reason the
whole-clip medoid picked the wrong seed).

## Mitigations

### Applied, partial evidence: restore the default `speed` to 1.0

This fork's default is now `speed=1.0` in both `py/helper.py` and
`web/helper.js` (previously `1.05`, matching upstream). This is no longer a
hypothesis on the table below — it is the shipped default — but the
listening evidence for it as a *glitch* fix is partial, and the two should
not be conflated:

- **The principled case stands on its own**, independent of any artifact.
  `dur_onnx = dur_onnx / speed` at `speed=1.05` shrinks the duration
  predictor's own trained time estimate by 4.76% on every default render.
  Upstream commit `8518b839` ("add speed parameter", 2025-11-19) introduced
  both the parameter and the `1.05` default across all nine language
  bindings in one commit, with no recorded rationale — the commit message is
  three words, and the README states only the value and a recommended
  range. Restoring `1.0` restores the model's own prediction and matches
  pre-2025-11-19 behaviour.
- **The artifact evidence is one sentence, corroborated; the second is
  known-unreliable, not merely unreproduced.** Bench 8
  (`py/phase2a_speed_rootcause.py`, see
  [LISTENING_BENCHES.md](LISTENING_BENCHES.md#8-phase-2a--speed-root-cause))
  ran a speed sweep {1.05, 1.00, 0.90} at fixed seed and text. On "She sells
  seashells by the sea shore every summer morning.", the previously flagged
  compression artifact was clearly present at `speed=1.05` and absent at
  both `1.00` and `0.90` — **now corroborated by bench 7**: bench 7's clip13
  is bit-identical to this bench's seashells speed-1.05 clip, and both were
  independently flagged "yes, clearly". The second flagged sentence ("How
  much wood would a woodchuck chuck...") showed no artifact at any speed, but
  its three ratings were logged 2 seconds apart on 4-second clips — too close
  to trust — so that row was not treated as evidence and a re-test was
  outstanding. Bench 7 supplies the explanation: the *same* bit-identical
  woodchuck clip, asked the open question "do you hear an audible artifact"
  instead of "is the artifact you flagged before still present," came back
  "yes, clearly, chuck too condensed." Bench 8's woodchuck row was suppressed
  by its own leading phrasing, not a genuine null — it remains unusable, and
  the re-test (now due with an open question) is still outstanding.
  Rating noise is on the order of ±1 category (measured from an accidental
  control: two acoustically identical clips, differing only by a discarded
  tail ~80 dB below the signal, were rated a category apart), which bounds
  how much can be read into a single-sentence result either way.
- **Speed is not the whole mechanism.** On the same seashells sentence at
  unchanged `speed=1.05`, raising `TOTAL_STEP` from 8 to 32 also removed the
  artifact (see the item below). Duration is invariant to step count, so the
  artifact is not purely a duration effect — compression appears to make the
  acoustic modeling problem harder, and too few denoising steps cannot
  resolve it at the tightened duration.
- **Measured word durations are consistent with compression.** In the
  speed sweep, flagged and utterance-final words lengthen as speed drops,
  and the final word grows super-proportionally to the clip as a whole (on
  one seed, +50% word duration against +17% clip duration from 1.05 to
  0.90) — more than a uniform time-stretch would predict.

Net: do not claim the artifact is fixed — bench 7 shows a second, unrelated
artifact family (sibilant over-drive) that speed does not touch. Claim the
default is restored to the model's own prediction, on solid independent
grounds, with one corroborated supporting listening result (seashells) and
one row (woodchuck) known to be compromised by leading question framing
rather than genuinely null, with its re-test still outstanding. To
reproduce the old behaviour, pass `speed=1.05` explicitly.

### Proposed, not established

Each entry states its cost and how it would be tested.

1. **Raise `TOTAL_STEP`** (every corpus generated this session used 8).
   Partially tested, not established: on the single flagged sentence above,
   `TOTAL_STEP=32` at unchanged `speed=1.05` removed the artifact, same as
   dropping speed did. Cost: purely an inference-time parameter, no code
   change, but proportionally slower rendering. Remaining test: by ear,
   since no automated detector here has passed validation against the
   listener — rerun a bench-6-style triple set at a higher step count,
   including the unreproduced second sentence, and ask the same listener the
   same question.
2. **Expose the vocoder seed in the public API**, in both `py/helper.py` and
   `web/helper.js`, defaulting to today's unseeded behaviour. Low cost — a
   plumbing change, no model or algorithm change. This is the precondition
   for anything below it that selects or targets a render: without a
   reproducible seed, a user can neither confirm a good render nor retry a
   bad one. Test: render the same (style, seed) twice, confirm bit-identical
   output.
3. **Predict artifact risk from text before rendering**, and spend extra
   denoising steps only where risk is high. Motivated by the cross-seed
   result above: instability is a property of the *word*, not the render,
   and it concentrated in stressed, phonetically dense content syllables in
   the two texts checked — the affricate and compound stress of
   wood-**CHUCK**, the sibilant cluster of sea-**SHORE**. It is not purely
   sibilant density, though: 'morning' is nasal, not sibilant, and still
   ranked high. Cost: a text-side risk model (rule-based on phonetic
   features, or learned from a larger labelled set) plus a way to spend
   variable compute per word, which the frozen ONNX graph may or may not
   support at word granularity — unverified. Test: does risk correlate with
   the cross-seed spread measure on a corpus beyond the two texts used here,
   and does added denoising measurably reduce spread on flagged words.
4. **Medoid-of-N** — listed here only as contradicted (see above), not as a
   live proposal. Before revisiting it: either a larger labelled set to
   properly test which direction the effect runs, or restricting the
   distance to the words the cross-seed analysis flags as unstable rather
   than scoring the whole utterance.

## The baseline artifact rate (measured)

Bench 7 (20 blind stock clips, all 10 shipped presets, a mix of 8 corpus
texts and 6 seeds per preset/text cell, 4 hidden repeats of the bench-6
control clips as internal consistency checks) answers what was previously
open here. See bench 7 in [LISTENING_BENCHES.md](LISTENING_BENCHES.md) for
the generator, inputs, and full verdict; summary:

- **Consistency: 4 of 4 hidden repeats reproduced the listener's earlier,
  independently-worded judgements**, blind — including which word was
  flagged. This is the reason the ear, not any automated measure, is treated
  as the reliable instrument in this doc.
- **Rate: of the 16 randomly drawn stock clips, 1 flagged "yes, clearly"
  (6.2%, Wilson 95% CI 1.1%-28.3%), 3 more "maybe" (25% combined, CI
  10.2%-49.5%), 12 clean.** Stock Supertonic produces a clearly audible
  artifact on roughly one render in sixteen; the interval is wide enough that
  only the order of magnitude is established.
- **Two artifact families, not one.** Time compression ("condensed",
  "time-condensed final word", "too quickly") is what the speed default,
  below, addresses. **Sibilant over-drive is separate and unexplained** — "a
  strong sharp 's' over-drive resulting in a sharp hissing" was the single
  clear flag among the randomly drawn clips, with two more "maybe"s citing
  sharp 's'-sounds. The speed fix does not touch this second family, and it
  was the only clear flag in the random draw — so the speed fix should not be
  read as addressing "the" baseline artifact rate; it addresses one of two
  known causes.
- **Every flagged clip was a female preset**: 4 of 8 female-preset clips
  flagged (F1, F3, F5 twice), 0 of 8 male-preset clips (Fisher exact
  one-sided p = 0.038). Small sample — this is a new observation, not an
  established effect — but every prior artifact measurement in this project
  used M1, so the rates and the mechanism above may both understate what a
  female preset does.
