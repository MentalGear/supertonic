# Emotion Calibration

This document describes how to create reusable `surprised` and `angry`
emotion controls for Supertonic voice styles.

## References

- [supertonic.embed](https://github.com/kdrkdrkdr/supertonic.embed) extracts
  Supertonic-compatible style JSON files from WAV recordings.
- [supertonic3-voice-clone](https://github.com/saurabhv749/supertonic3-voice-clone)
  provides a Supertonic 3 style-training workflow.
- [voice-builder-for-supertonic-3](https://github.com/Fawzan09/voice-builder-for-supertonic-3)
  demonstrates a Google Colab T4 workflow for training a single custom voice
  style. It is useful for GPU setup, but it does not publish a reusable
  emotion-calibration notebook or emotion JSON assets.
- [EmoShift](https://arxiv.org/abs/2601.22873) motivates emotion-specific
  activation steering vectors.
- [EmoSphere-TTS](https://arxiv.org/abs/2406.07803) motivates continuous,
  normalized emotion intensity.
- [RAVDESS](https://zenodo.org/records/1188976) provides matched actor and
  utterance recordings for neutral, angry, and surprised speech. RAVDESS is
  licensed CC BY-NC-SA 4.0.

## Recommended Data

Do not build a general emotion vector from one speaker. One speaker is useful
for a voice-specific experiment, but the result will also contain that
speaker's pitch, timbre, and recording conditions.

Use at least five speakers for a prototype and preferably 8-20 speakers for a
general vector. For every speaker, collect matched neutral and emotional
utterances:

```text
speaker_01/neutral_01.wav
speaker_01/angry_01.wav
speaker_01/surprised_01.wav
speaker_01/neutral_02.wav
speaker_01/angry_02.wav
speaker_01/surprised_02.wav
```

The local RAVDESS proof-of-concept recordings are in the ignored asset tree:

```text
assets/reference_recordings/ravdess_actor01/neutral.wav
assets/reference_recordings/ravdess_actor01/angry.wav
assets/reference_recordings/ravdess_actor01/surprised.wav
```

These three files use actor 01, the same statement, repetition, and intensity;
only the emotion label changes. They are suitable for pipeline validation, not
for a final speaker-independent vector.

## 1. Install Dependencies

The normal ONNX runtime dependencies are installed from:

```bash
pip install -r py/requirements.txt
```

The optional extractor requires PyTorch, torchaudio, WavLM, and ONNX-to-Torch
conversion packages. On a CPU-only machine, install CPU PyTorch first:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install onnx2torch onnxslim transformers
```

The `supertonic.embed` extractor is designed for CUDA and may be too slow on
CPU. It also downloads the WavLM checkpoint on first use. A Google Colab T4
runtime is a practical alternative to a local GPU. The community
`voice-builder-for-supertonic-3` project confirms that this class of
Supertonic style optimization fits within a T4 runtime.

The Colab adaptation should run the same extraction cell three times per
speaker, using the neutral, angry, and surprised recordings, and save all
three JSON outputs to persistent Drive storage. Do not train one combined
style from the three recordings: the emotion delta requires three separate
styles with the same speaker and matched content.

On a T4, budget roughly 15-30 minutes per short recording after the initial
model downloads. Three recordings commonly take 45-90 minutes. A multi-speaker
dataset scales approximately linearly, although batching can reduce the total
time when GPU memory allows it. The current notebook reports stage-level
progress; a future extractor revision should expose optimization-step progress
with `tqdm` or a callback so that long WavLM runs show percentage, elapsed time,
and estimated time remaining.

## 2. Download Model Assets

Download the official Supertonic 3 `onnx/` and `voice_styles/` directories
from [Hugging Face](https://huggingface.co/Supertone/supertonic-3) into the
repository `assets/` directory. The required layout is:

```text
assets/
  onnx/
    duration_predictor.onnx
    text_encoder.onnx
    vector_estimator.onnx
    vocoder.onnx
    tts.json
    unicode_indexer.json
  voice_styles/
    M1.json
    ...
```

These assets are ignored by Git and must not be committed.

## 3. Extract Full Styles from WAV Files

Configure the extractor with the downloaded assets. Its default
`src/config.yaml` expects `models/onnx` and `models/voice_styles`; change those
paths to the repository assets or create equivalent symlinks.

Create a WAV list:

```text
assets/reference_recordings/ravdess_actor01/neutral.wav|actor01_neutral
assets/reference_recordings/ravdess_actor01/angry.wav|actor01_angry
assets/reference_recordings/ravdess_actor01/surprised.wav|actor01_surprised
```

Run the extractor from the cloned `supertonic.embed` repository:

```bash
python src/run_extract_style_batch.py /path/to/wavlist.txt --out /path/to/styles
```

Each output JSON contains `style_ttl` with shape `[1, 50, 256]` and
`style_dp` with shape `[1, 8, 16]`.

## 4. Build Emotion Difference Files

For a first speaker-specific test, use the extracted neutral and emotional
styles:

```bash
python py/create_emotion_style.py \
  --neutral styles/actor01_neutral.json \
  --emotional styles/actor01_surprised.json \
  --emotion surprised \
  --output assets/emotion_styles/surprised.json

python py/create_emotion_style.py \
  --neutral styles/actor01_neutral.json \
  --emotional styles/actor01_angry.json \
  --emotion angry \
  --output assets/emotion_styles/angry.json
```

For a general vector, first compute one delta per speaker, then average the
speaker deltas. Do not average raw full styles across speakers before removing
the neutral style.

The preferred general form is:

$$
\Delta_e = \frac{1}{N}\sum_i
\left(s_{i,e} - s_{i,\mathrm{neutral}}\right)
$$

Start with `style_ttl` only. `style_dp` primarily controls duration and remains
neutral by default; the Python CLI exposes `--emotion-include-duration` for
experiments with a separately validated prosody delta.

## 5. Apply Emotion at Runtime

Python:

```bash
cd py
python example_onnx.py \
  --onnx-dir ../assets/onnx \
  --voice-style ../assets/voice_styles/M1.json \
  --emotion surprised \
  --emotion-intensity 0.75
```

The browser demo exposes the same controls after starting it with:

```bash
cd web
npm run dev
```

Intensity should be tested at `0.0`, `0.25`, `0.5`, `0.75`, and `1.0`. A future
production blend should normalize each TTL row after blending because the
extractor projects style rows onto the unit sphere:

$$
s(\alpha) = \operatorname{normalize}
\left(s_{\mathrm{voice}} + \alpha\Delta_e\right)
$$

The runtime normalizes each TTL row after blending and leaves duration neutral
by default. Emotion JSON files are still calibration artifacts and must be
evaluated before distribution.

## Inline Emotion Tags

The ONNX models receive one style tensor per inference item, so a style cannot
change halfway through a single model call. Inline controls should therefore
be implemented as segment synthesis rather than by trying to inject tags into
the model text:

```text
This is calm. [angry intensity=0.7]This part is angry.[/angry] Now I am calm again.
```

The parser should remove tags before normal text preprocessing, split the text
into tagged segments, synthesize each segment with its selected style, and
concatenate the audio with a short crossfade. Supported forms should initially
be explicit and nest-free:

```text
[angry]text[/angry]
[surprised intensity=0.5]text[/surprised]
```

For smooth transitions, interpolate the style at segment boundaries and use a
10-30 ms audio crossfade. Do not use square-bracket tags without preprocessing:
the existing text normalizer intentionally converts `[` and `]` to spaces.
Sentence-level segmentation is the safest first implementation; token-level
style changes inside one model inference are not supported by the current
ONNX graph.

## 6. Evaluate the Result

For every base voice and emotion intensity:

1. Check that synthesis succeeds and audio is finite.
2. Use a speaker-similarity model to verify that voice identity is retained.
3. Use an emotion classifier or human listeners to verify angry/surprised
   recognition.
4. Compare pitch, energy, speaking rate, and duration against the reference.
5. Check neutral output at intensity `0.0` against the unmodified voice.

RAVDESS is CC BY-NC-SA 4.0. Keep attribution and comply with its
non-commercial/share-alike terms when using those recordings or derived
calibration assets.

## Style Design Decisions

### One speaker versus a general style

A single speaker is sufficient for a voice-specific experiment or for testing
the pipeline. It is not sufficient for a speaker-independent emotion vector:
the result will also contain that speaker's pitch range, timbre, vocal effort,
and recording conditions.

Use at least five speakers for a prototype and preferably 8-20 speakers for a
general vector. Compute each speaker's emotion delta against that speaker's
neutral recording before averaging across speakers.

### TTL and duration separation

`style_ttl` is the first target for emotion steering. `style_dp` primarily
controls duration and speaking rate, so changing it can unintentionally make
speech faster or slower. The runtime leaves DP neutral by default and exposes
duration blending only as an explicit experiment.

### Strength interpolation

Emotion strength is a continuous value from `0.0` to `1.0`. Evaluate at
`0.0`, `0.25`, `0.5`, `0.75`, and `1.0` rather than assuming the relationship is
perceptually linear. TTL rows are normalized after blending because extracted
style rows lie on a unit-sphere-like manifold:

$$
s(\alpha) = \operatorname{normalize}
\left(s_{\mathrm{voice}} + \alpha\Delta_e\right)
$$

Named controls such as `angry` and `surprised` should remain presets over this
continuous strength value. A future API can expose arousal, valence, and
dominance coordinates, with named emotions mapped to those coordinates.

### Inline style tags

The current ONNX graph accepts one style tensor per inference item, so it
cannot change style halfway through one model call. Tags should therefore be
implemented as segment controls:

```text
This is calm. [angry intensity=0.7]This part is angry.[/angry] Now I am calm.
```

The parser must run before text normalization, remove tags, synthesize each
segment with its selected style, and concatenate the audio with a short
crossfade. Start with non-nested phrase or sentence tags:

```text
[angry]text[/angry]
[surprised intensity=0.5]text[/surprised]
```

The existing normalizer converts square brackets to spaces, so passing tags
directly into the current text processor will not work. A 10-30 ms crossfade
and preserved punctuation pauses should reduce audible joins. True token-level
style changes inside one ONNX inference are not supported.

### Suggested extraction backlog

Prioritize a small, broadly useful set first:

```text
surprised, angry, happy, sad, calm
```

Then consider:

```text
fearful, disgusted, excited, confident, empathetic,
whisper, shout, urgent, dramatic, storytelling,
laugh, cry, sigh, breathless, hesitant
```

Emotion, delivery style, and paralinguistic events should be tracked as
separate categories. Short events such as laughter, crying, and sighs may be
better represented as inline tags or reference clips than as persistent voice
style vectors.