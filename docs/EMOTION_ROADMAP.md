# Emotion Control Roadmap

This roadmap tracks the path from the current calibrated `angry` and
`surprised` controls to a robust, general-purpose style system.

**The phases below remain valid; the order does not.**
[new-plan.md](../new-plan.md) supersedes this roadmap's sequencing, placing
emotion as the first axis of a shared parametric system rather than a track
run to completion on its own. Concretely: Phase 1's `gain` parameter is
replaced by that plan's `with_deltas()` refactor, Phase 2's manual
multi-speaker extraction should wait for its amortized encoder, and Phase
3's evaluation should wait for the `intensity=0.0` normalization fix — see
that document for why each holds. Phase 6 is where the two documents
converge.

## Current State

Completed:

- Added Python and browser emotion controls.
- Added continuous intensity from `0.0` to `1.0`.
- Added normalized TTL steering at runtime.
- Kept DP/duration neutral by default.
- Added optional duration blending for experiments.
- Prepared a resumable Colab extraction notebook.
- Extracted `angry.json` and `surprised.json` from matched RAVDESS recordings.
- Applied both controls to the M1 base voice and confirmed audible waveform
  differences.
- Generated a strength listening set in:

```text
py/results/listening_sets/m1_meeting/
```

The current calibration is a one-speaker proof of concept. It is useful for
validating the pipeline, but it is not yet a general speaker-independent
emotion model.

## Audio Naming Convention

Every comparison set should use identical text, base voice, and inference
settings. Encode the emotion, normalized strength, and optional gain in the
filename:

```text
{emotion}_strength_{strength}_gain_{gain}.wav
```

Examples:

```text
angry_strength_025_gain_1.0.wav
angry_strength_075_gain_2.0.wav
surprised_strength_10_gain_3.0.wav
neutral_strength_000_gain_1.0.wav
```

For the default calibrated assets, gain is `1.0`. Gain is an experimental
multiplier applied to the stored delta before blending; it is separate from
per-request intensity. Keep gain variants in the same strength-range folder so
listeners can compare them directly.

Recommended listening matrix:

```text
emotions: angry, surprised
strengths: 0.25, 0.5, 0.75, 1.0
gains: 1.0, 2.0, 3.0
```

The canonical M1 set has a `manifest.json` beside the WAV files. It includes
the gain `2.0` and `3.0` experiments, marked as experimental. Older one-off
outputs are retained under `py/results/legacy/`.

Do not distribute gain `2.0` or `3.0` as defaults without checking clipping,
naturalness, and speaker similarity. The earlier angry gain tests reached a
peak of `1.0`, which indicates clipping risk.

## Phase 1: Stabilize the Prototype

- Add a first-class `gain` parameter to Python and browser APIs instead of
  manually editing JSON files.
- Preserve normalized TTL rows after applying `gain * intensity * delta`.
- Add audio output metadata containing voice, emotion, intensity, gain, and
  source asset.
- Add automated tests for zero intensity, gain one, batch broadcasting, and
  unit-row normalization.
- Add progress callbacks or `tqdm` to the extraction optimizer.

Target API:

```python
style = base_style.with_emotion(
    emotion_style,
    intensity=0.75,
    gain=1.0,
)
```

## Phase 2: Improve Calibration Data

- Extract matched neutral, angry, and surprised styles for at least five
  speakers.
- Prefer 8-20 speakers for a general vector.
- Use multiple matched utterances per speaker.
- Compute each speaker's delta against their own neutral style.
- Average deltas only after neutral subtraction.
- Start with TTL-only deltas; evaluate DP separately.
- Store source recordings, extraction configuration, and model revisions in
  metadata.

The general delta is:

$$
\Delta_e = \frac{1}{N}\sum_i
\left(s_{i,e} - s_{i,\mathrm{neutral}}\right)
$$

## Phase 3: Evaluate and Select Strength

For every emotion, voice, strength, and gain:

- Run a speaker-similarity check.
- Run an emotion classifier or listener rating.
- Measure pitch, energy, speaking rate, duration, and clipping.
- Check intelligibility with transcription.
- Compare neutral intensity `0.0` against the unmodified base voice.

Select defaults based on measured quality rather than assuming `1.0` is best.
A separate gain can be useful for weak calibration data, but high gain may
amplify artifacts or clip audio.

## Phase 4: Inline Emotion Tags

The current ONNX graph accepts one style per inference item, so inline emotion
requires segment synthesis:

```text
This is calm. [angry intensity=0.7]This part is angry.[/angry] Now I am calm.
```

Implementation sequence:

1. Parse non-nested tags before text normalization.
2. Split into plain and emotion segments.
3. Apply each segment's emotion style and strength.
4. Synthesize each segment independently.
5. Preserve punctuation pauses.
6. Join segments with a 10-30 ms crossfade.
7. Add tests for malformed, nested, and unmatched tags.

Start with phrase or sentence boundaries. Do not promise token-level style
changes until the ONNX graph supports per-token style conditioning.

## Phase 5: Broader Style Backlog

Prioritize:

```text
happy, sad, calm, fearful, excited
```

Then consider delivery styles:

```text
whisper, shout, urgent, dramatic, storytelling, professional
```

Treat short events separately where appropriate:

```text
laugh, cry, sigh, breathless, hesitant
```

Events may work better as inline tags or short audio effects than persistent
voice-style deltas.

## Phase 6: Continuous Emotion Space

After named emotions are reliable, investigate a continuous arousal,
valence, and dominance representation. Named emotions can become presets in
that space rather than isolated vectors:

```json
{
  "name": "angry",
  "category": "emotion",
  "arousal": 0.9,
  "valence": -0.7,
  "dominance": 0.8
}
```

This should follow, not precede, multi-speaker calibration and evaluation.

## Reproducibility and Recovery

Use [emotion_calibration_colab.ipynb](emotion_calibration_colab.ipynb) on a T4
or better. The notebook stores recordings, model cache, extracted styles, and
an extraction manifest on Google Drive. If Colab disconnects, reconnect the
same Drive and rerun the setup/extraction cells; completed emotion JSONs are
skipped.

See [EMOTION_CALIBRATION.md](EMOTION_CALIBRATION.md) for the full extraction
procedure and licensing notes.
