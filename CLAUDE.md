# CLAUDE.md

Fork of Supertonic (on-device ONNX TTS) that adds voice style control,
including emotion. See [README.md](README.md) for the upstream feature set.

## Agent Workflow

**Delegate by default. Keep the main loop clean.**

The main session should do four things only: plan, delegate, check returned
work, and talk to the user. It should not be the thing reading files, running
greps, or grinding through edits — that work belongs in subagents, and doing it
inline burns the context the main loop needs for judgment.

### Model selection for subagents

- **Sonnet is the default.** Use it for the large majority of delegated work:
  searching the codebase, reading and summarizing files, running tests,
  applying mechanical edits, generating listening sets, writing docs.
- **Opus only for genuinely difficult tasks** — non-obvious debugging, design
  work with real trade-offs, numerical or DSP reasoning where a wrong answer is
  hard to spot. Opus is the ceiling, not a fallback for "this seems important."
- Never delegate to a model above Opus.

### Practice

- Prefer several parallel Sonnet subagents over one long serial investigation.
  Independent work goes out in a single message so it runs concurrently.
- Give each subagent a self-contained brief: the goal, the relevant paths, and
  what to report back. They do not share the main loop's context.
- Ask for conclusions, not transcripts. A subagent that returns file dumps has
  moved the context problem rather than solved it.
- Verify what comes back before acting on it — subagents report confidently and
  are sometimes wrong. Spot-check claims that decide a direction.
- Keep the main loop's own tool use to quick confirmations: a single file read
  to check a returned claim, a `git status`, a targeted grep.

## Project Notes

- **Style tensors are the only conditioning surface.** The ONNX graph is
  frozen. `style_ttl [1, 50, 256]` and `style_dp [1, 8, 16]` are all there is
  to control. Any "make the engine do X" idea has to reduce to producing those
  two tensors.
  Where each one enters matters, and is not what the names suggest. `style_dp`
  (128 numbers) is read by `duration_predictor` alone — it sets how long each
  text unit lasts, and so the temporal skeleton, and nothing else. `style_ttl`
  (12,800 numbers) is read twice: by `text_encoder`, producing a `text_emb`
  with a per-text-position axis, and again by `vector_estimator` at every
  denoising step. So it is not a global timbre vector, despite upstream docs
  glossing it as "timbre" — conditioning the text embedding gives it
  per-position reach, which is why perturbing it can change which words are
  emphasized while durations stay pinned. Treat "style_ttl is timbre" as a
  label that under-describes what it controls.
  `style_dp` is the opposite case — a label that over-describes one. Its graph
  emits a **single per-utterance scalar in seconds**, not a per-token array,
  despite the `Squeezeduration_dim_0` output name suggesting otherwise;
  confirmed by direct inference (`shape (1,)`), and by the `# dur_onnx: [bsz]`
  comment at `py/helper.py:330`. So for a fixed utterance the whole of
  `style_dp`'s 128 numbers is observable only through that one value — which
  `speed` already sets directly, at `py/helper.py:326`
  (`dur_onnx = dur_onnx / speed`). Treat timing as an effective scalar, not a
  second control surface. There is no per-token duration anywhere in the
  Python pipeline to intercept: text and latent are aligned inside the frozen
  `vector_estimator`, and only `text_mask`/`latent_mask` cross the boundary.
  Rate and rhythm control, if it is reachable at all, has to come through
  `style_ttl`.
- **This fork's `speed` default is `1.0`, not upstream's `1.05`, on purpose.**
  `1.05` divides the duration predictor's own trained estimate on every
  default render, a change upstream introduced with the parameter itself
  (commit `8518b839`) and never explained. `1.0` restores the model's own
  prediction; pass `speed=1.05` for upstream's behaviour. See
  [docs/GLITCH_MITIGATION.md](docs/GLITCH_MITIGATION.md) for the (partial)
  listening evidence. Do not "fix" this back to `1.05` by syncing upstream.
- **Style blending lives in `py/helper.py` (`Style`) and `web/helper.js`.**
  Changes to blending semantics must land in both — the browser path is not
  generated from the Python one, and the two silently diverged once already
  (the browser normalized `style_ttl` over the whole 50x256 block instead of
  per row). When you touch blending, change both files and add a numeric
  cross-check that the two agree. A third mirror now exists: `with_deltas_np`
  in the `style_tools.py` module that `docs/style_extraction_colab.ipynb`
  writes into its Colab workspace. It is a mirror, not a fork — the
  presentation-axis notebook cross-checks it numerically against
  `Style.with_deltas` at every weight it applies, and that check is the thing
  keeping three copies honest. Update it with the other two.
- **`with_deltas()` / `withDeltas()` is the blending API.** It takes any number
  of `(delta, weight)` pairs, accumulates them in pre-normalization space, and
  restores per-row norms exactly once at the end — so composition is order
  independent and zero weight is an exact identity. Weights are deliberately
  unclamped. `with_emotion()` is a backward-compatible single-delta wrapper.
  The invariant to preserve: **normalize per row, over the last axis, once.**
- Run the blending tests after any change there:
  `node --test web/helper.test.js` and
  `python3 -m unittest discover -s py -p "test_*.py"`.
- **Do not commit ignored assets**: ONNX models, recordings, extracted style
  JSONs, generated WAVs.
- **Ask listeners what they hear, not whether a known artifact persists.** A
  bench that asks "is the artifact you flagged before still present in this
  clip?" primes the listener to re-identify a description rather than report
  a fresh perception — and can suppress a real detection. Measured directly:
  bench 8 asked that leading question of a woodchuck clip and got "no" at
  every speed; the same clip, verified bit-identical (waveform correlation
  1.00000000), was played blind in bench 7 under the open question "do you
  hear an audible artifact in this clip?" and came back "yes, clearly, chuck
  too condensed" — seven minutes later, on the same audio. Prefer the open
  question and let the listener describe what they hear unprompted; a
  leading question is a legitimate follow-up once, never the first ask. This
  sits alongside the "make the listening task explicit" bullet below: be
  explicit about the TASK the listener is doing, never about the answer you
  expect.
- **Vocoder sampling is unseeded.** `sample_noisy_latent()` in `py/helper.py`
  draws `np.random.randn` with no seed, so rendering the same style tensor twice
  gives audibly different waveforms. Any "is this the same as before" check must
  be made at the tensor level, not on audio, unless the RNG is explicitly seeded.
- Generated audio follows the naming and manifest convention in
  [docs/EMOTION_ROADMAP.md](docs/EMOTION_ROADMAP.md). Keep comparison sets on
  identical text, base voice, and inference settings.
- **Post generated audio into the chat.** Every WAV produced for evaluation —
  the Phase 0 interpolation gate, emotion listening sets, each axis's
  monotonicity sweep — must be surfaced in the conversation, not merely written
  to `py/results/`. Files on disk are unhearable: the user cannot judge them,
  and these phases are judged by ear before they are judged by any metric. Send
  the audio itself, with the manifest line naming base voice, text, axis,
  weight, and inference settings, so each clip is identifiable without
  cross-referencing a file tree. When a set is large, send the endpoints and
  midpoint rather than all of it, and say what was omitted.
- **Publish listening sets as an Artifact, not only as loose files.** Whenever a
  set needs a verdict by ear, build a listening bench: the clips embedded as
  data URIs and playable in order, labelled with what each one is, the run
  parameters and any numeric profile alongside them, and the decision the
  listener is being asked to make stated on the page. Playing one clip stops the
  others so comparison is A/B rather than overlapping. Attach the raw WAVs too —
  the page is for judging, the files are for keeping. Reuse the established
  visual system across benches (IBM Plex Sans/Mono with Newsreader, teal accent
  on cool neutrals) so successive sets read as one series. See
  [docs/LISTENING_BENCHES.md](docs/LISTENING_BENCHES.md) for every bench built
  so far, its generator, and the verdict it produced.
- **Keep the bench generator in the repo, per phase.** A published Artifact URL
  is not a durable record on its own; the generator, its inputs, and the
  verdict obtained belong in the repo, indexed in
  [docs/LISTENING_BENCHES.md](docs/LISTENING_BENCHES.md). Generated bench HTML
  stays out of git — the files run 1.8–8.3 MB because audio is base64-embedded,
  and the WAVs are gitignored anyway.
- **Make the listening task explicit in the bench UI.** A bench exists to
  obtain a human verdict, so the page must state plainly what is being asked,
  in a visually distinct callout near the top, with a one-line restatement
  above each clip group, and say what the answer decides. Distinguish "which
  sounds better" from the question actually being asked, which is usually
  narrower — a listener answering the wrong question gives a confident,
  useless verdict. Labels must describe what the listener is hearing, not the
  mechanism that produced it: a clip labelled "reads no audio at all"
  (describing the predictor) was reasonably misread as describing the clip,
  which is obviously synthesized speech.
- **Compare spectrograms before you compare aggregates.** A scalar summary —
  median F0, mean spectral flatness, voiced fraction, WER — collapses both time
  and frequency, so a change that is localized in time (per-word emphasis) or
  that moves in opposite directions across the utterance averages to nothing. A
  flat aggregate is not evidence that nothing changed; it is evidence that
  nothing changed *in that projection*. This is not hypothetical: an
  utterance-level battery reported a perturbation ladder as acoustically flat,
  and a listener immediately heard the emphasis moving.
  So: diff the log-mel spectrograms first and look at the picture, then reach
  for aggregates to scale the finding across a sweep. Two conditions make the
  diff meaningful — hold text and `style_dp` fixed so durations are pinned, and
  seed the vocoder RNG, or the difference is mostly sampling noise. Without
  alignment (different durations or an unseeded render) a raw 2D diff is
  meaningless; align first or compare distributions instead.
- **Level-match before listening across magnitudes.** Any A/B over a
  perturbation ladder must be rms level-matched first: magnitude alone buys up
  to +4.9 dB of plain loudness in `style_ttl`, and an unmatched comparison is
  decided by which clip is louder rather than by the effect under test.
- **Calibrate a distance before citing it as evidence.** This project has now
  been misled by an uncalibrated metric three times, each caught by a
  listener or a calibrated measurement rather than by the metric itself:
  utterance-level aggregates called a moving perturbation ladder acoustically
  flat; delta-ECAPA was used as an audibility meter although it is trained to
  be prosody-invariant; and Phase 2b used log-mel distance and active-row
  style cosine as identity measures, when calibrated across 80 held-out
  samples the same-speaker and different-speaker distributions overlap
  completely. **Before using a distance as evidence, calibrate it against
  known-same and known-different pairs; if those two distributions overlap,
  the distance cannot support the claim.** A threshold derived from a single
  pair is not a calibration.
- **An estimator's expressive ceiling must be checked out of sample — an
  in-sample headroom statistic is a function of matrix shape, not content,
  and will certify any design.** Same failure mode as the bullet above — an
  uncalibrated instrument mistaken for a fact about the world — but here the
  check itself, not its absence, was the problem. Phase 2b did compute a
  ceiling before reading its within-family residual null: participation
  ratio (~303) and the fraction of residual variance inside the top-n_train
  principal components (~0.82, `var_in_top_k`, `phase2b_wavlm.py`), read
  together as 82% headroom. Both choose their subspace from the same data
  they score, and `py/phase2a_ceiling_null.py` reproduces both numbers to
  four significant figures by running the identical diagnostics on pure
  isotropic noise — they measure (n, d, n_train) alone. What actually binds a
  ridge is out-of-sample: predictions lie in the training targets' span, so a
  fresh direction reaches only about n_train/d (here 3.9%, matching the
  noise sim and Phase 2b's own reachable fraction, and consistent with the
  oracle's ~2.6% R^2 ceiling under the probe's own scoring). Operational
  test: run the diagnostic on noise of the same shape — if it returns the
  same number, it is measuring the shape.
- Active research direction: [new-plan.md](new-plan.md).
