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
- **Style blending lives in `py/helper.py` (`Style`) and `web/helper.js`.**
  Changes to blending semantics must land in both — the browser path is not
  generated from the Python one, and the two silently diverged once already
  (the browser normalized `style_ttl` over the whole 50x256 block instead of
  per row). When you touch blending, change both files and add a numeric
  cross-check that the two agree.
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
- Active research direction: [new-plan.md](new-plan.md).
