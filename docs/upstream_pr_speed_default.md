# [DRAFT — NOT FILED] Restore default `speed` to `1.0`

> **This is a draft PR description written for review by this fork's
> maintainer. It has not been opened against `supertone-inc/supertonic` and
> should not be treated as filed.** If it is sent, it should go as a single
> PR touching every binding listed below, since the value must stay in sync
> across them (this project's own `CLAUDE.md` calls out exactly this failure
> mode: the Python and browser copies of a shared constant have drifted
> before).

## Summary

`speed` currently defaults to `1.05` in every language binding. Because
`dur_onnx = dur_onnx / speed`, that default silently shortens the duration
predictor's own output by 4.76% on every render where a caller doesn't pass
`speed` explicitly. This PR proposes changing the default to `1.0` — i.e.
using the model's own duration prediction unless a caller asks for something
else. It does not remove or rename the parameter, and it does not touch its
valid range.

## Why `1.0`, independent of any audible effect

1. **`1.0` is not a value choice, it's the absence of one.** The duration
   predictor is a trained component; its output is a duration in seconds
   already fitted to the model's own prosody, not a "before speed" number
   that a downstream factor is supposed to correct. `speed=1.0` means "trust
   the model's own estimate." `speed=1.05` means "the model is always 4.76%
   slower than it should be," which is a strange thing for a default to
   assert about a component it ships in the same commit.
2. **It was a silent behaviour change for every existing user.** Before
   commit `8518b839` ("add speed parameter", 2025-11-19), there was no
   `speed` parameter and no division — the predictor's duration was used
   as-is, across all bindings. That commit introduced both the parameter and
   the `1.05` default at once, across all nine bindings that existed at the
   time (Python, Node.js, Browser/WebGPU, Java, C++, C#, Go, Swift, Rust —
   Flutter was added five days later, already inheriting `1.05`). Anyone
   who upgraded without reading the diff closely got 4.76%-faster speech by
   default, with the same call signature (`speed` is keyword/optional
   everywhere) giving no indication anything had changed.
3. **No rationale is recorded anywhere for `1.05` specifically.** The commit
   message is three words. The README documents the value and a recommended
   range (`0.9`–`1.5`) but not why the default sits inside that range at
   `1.05` rather than at the model's own `1.0`. If there is a reason —
   e.g. a listening panel that preferred slightly faster delivery — it isn't
   in the repo, and this PR would be a reasonable place to record it if so.

To be fair to the original change: adding a `speed` knob at all is a good
feature, and exposing it consistently across nine independently-maintained
bindings in one commit is real, careful work. This PR is only about the
*default value* of that knob, not the knob itself.

## Supporting evidence — audible, but limited

Independent of the argument above, a listener on a downstream fork flagged a
compression-sounding artifact ("condensed", "pronounced too quickly") on
unperturbed stock output. We ran a controlled sweep to check whether the
`1.05` default was implicated:

- On **one sentence** ("She sells seashells by the sea shore every summer
  morning.", a previously-flagged seed), the artifact was clearly present at
  `speed=1.05` and audibly absent at both `speed=1.00` and `speed=0.90`, at
  matched text, style, seed, and denoising step count, level-matched for
  loudness.
- On the **same sentence, at the unchanged default `speed=1.05`**, raising
  the denoising step count from 8 to 32 *also* removed the artifact.
  Duration is invariant to step count, so `speed` is not shown to be the
  whole mechanism — the working hypothesis is that shrinking the time
  budget makes the acoustic modeling problem harder, and few denoising steps
  can't resolve it at the tightened duration. This PR's evidence is
  therefore about the *default*, not a claim that `speed` alone causes the
  artifact.
- A **second sentence** ("How much wood would a woodchuck chuck...") showed
  no artifact at any speed tested, but that result is not reliable: its
  three ratings were recorded roughly 2 seconds apart on 4-second clips,
  which is not enough separation to trust as an independent judgment. A
  re-test on this sentence is outstanding and this PR does not treat the
  null result as evidence either way.
- Rating noise on this task is on the order of **±1 category**: an
  accidental control (two acoustically identical renders, differing only by
  a discarded silent tail roughly 80 dB below the signal) was rated a full
  category apart by the same listener.
- Measured word durations are consistent with a compression effect:
  flagged and utterance-final words lengthen as `speed` drops from `1.05`
  toward `0.90`, and on one seed the final word lengthens ~50% while the
  clip as a whole grows only ~17% over the same range — more than a uniform
  time-stretch would predict.

**Read this section as supporting, not conclusive.** The principled argument
above does not depend on it. This section exists because "we also heard
something" is worth reporting honestly, including its limits — one
confirmed sentence, one unreproduced null result, and a rating-noise floor
that is not small relative to the effect.

## Reproduction recipe

Using this repository's Python path (`py/`), with any shipped preset (e.g.
`M1`) and the models downloaded per the README:

```python
from py.helper import load_text_to_speech, Style
import numpy as np

tts = load_text_to_speech("assets/onnx", use_gpu=False)
style = Style(ttl, dp)  # load a preset's style_ttl / style_dp
text = "She sells seashells by the sea shore every summer morning."

np.random.seed(20261449)  # sample_noisy_latent() is otherwise unseeded —
                            # pin it so the two renders are comparable
wav_105, dur_105 = tts(text, "en", style, total_step=8, speed=1.05)

np.random.seed(20261449)
wav_100, dur_100 = tts(text, "en", style, total_step=8, speed=1.00)

# dur_100 should read ~4.29s where dur_105 reads ~4.09s (the model's own
# duration-predictor estimate, before and after the /speed division).
# Listen to wav_105 vs wav_100 around the sentence-final word.
```

The same comparison is available pre-rendered (see Showcase below), and the
full sweep (three speeds x two sentences x multiple seeds, plus a
step-count sweep and a trim-artifact control) is reproducible via this
fork's `py/phase2a_speed_rootcause.py`.

## Showcase — clips a reviewer should listen to

These were rendered by this fork's `py/phase2a_speed_rootcause.py` and are
not committed (gitignored, per this fork's policy on generated audio); paths
are given so a reviewer with access to this fork can regenerate or locate
them under `py/results/listening_sets/phase2a_speed_rootcause/`. Same text,
same base voice, same seed, same 8 denoising steps within each row; loudness
level-matched.

| Sentence | Seed | Clip (before, `speed=1.05`) | Clip (after, `speed=1.00`) | For reference (`speed=0.90`) | Listener verdict |
|---|---|---|---|---|---|
| "She sells seashells by the sea shore every summer morning." | `20261449` | `seashells_seed20261449_speed1.05_trimmed.wav` | `seashells_seed20261449_speed1.00_trimmed.wav` | `seashells_seed20261449_speed0.90_trimmed.wav` | Artifact present at 1.05, absent at 1.00 and 0.90 |
| "How much wood would a woodchuck chuck if a woodchuck could chuck wood?" | `20261069` | `woodchuck_seed20261069_speed1.05_trimmed.wav` | `woodchuck_seed20261069_speed1.00_trimmed.wav` | `woodchuck_seed20261069_speed0.90_trimmed.wav` | No artifact at any speed tested — **ratings unreliable (logged ~2s apart), re-test outstanding**, included here for completeness rather than as supporting evidence |

Listen to the seashells row first, focused on the final word ("morning.");
the woodchuck row is included so a reviewer sees the honest null result
alongside the positive one, not just the clip that supports the change.

## Proposed diff

Changes the default value only, in every binding that defines one. No
signature, no removed parameter, no range change.

```diff
--- a/py/helper.py
+++ b/py/helper.py
@@
-        speed: float = 1.05,
+        speed: float = 1.0,
   (x3: TextToSpeech._infer, TextToSpeech.__call__, TextToSpeech.batch)
```

```diff
--- a/web/helper.js
+++ b/web/helper.js
@@
-    async _infer(textList, langList, style, totalStep, speed = 1.05, progressCallback = null) {
+    async _infer(textList, langList, style, totalStep, speed = 1.0, progressCallback = null) {
@@
-    async call(text, lang, style, totalStep, speed = 1.05, silenceDuration = 0.3, progressCallback = null) {
+    async call(text, lang, style, totalStep, speed = 1.0, silenceDuration = 0.3, progressCallback = null) {
@@
-    async batch(textList, langList, style, totalStep, speed = 1.05, progressCallback = null) {
+    async batch(textList, langList, style, totalStep, speed = 1.0, progressCallback = null) {
```

```diff
--- a/web/index.html
+++ b/web/index.html
@@
-                            <input type="number" id="speed" value="1.05"
+                            <input type="number" id="speed" value="1.0"
```

```diff
--- a/nodejs/helper.js
+++ b/nodejs/helper.js
@@
-    async _infer(textList, langList, style, totalStep, speed = 1.05) {
+    async _infer(textList, langList, style, totalStep, speed = 1.0) {
@@
-    async call(text, lang, style, totalStep, speed = 1.05, silenceDuration = 0.3) {
+    async call(text, lang, style, totalStep, speed = 1.0, silenceDuration = 0.3) {
@@
-    async batch(textList, langList, style, totalStep, speed = 1.05) {
+    async batch(textList, langList, style, totalStep, speed = 1.0) {
```

```diff
--- a/nodejs/example_onnx.js
+++ b/nodejs/example_onnx.js
@@
-        speed: 1.05,
+        speed: 1.0,
```

```diff
--- a/java/ExampleONNX.java
+++ b/java/ExampleONNX.java
@@
-        float speed = 1.05f;
+        float speed = 1.0f;
```

```diff
--- a/cpp/helper.h
+++ b/cpp/helper.h
@@
-        float speed = 1.05f,
+        float speed = 1.0f,
   (x3: the three overloads/defaults in this header)
```

```diff
--- a/cpp/example_onnx.cpp
+++ b/cpp/example_onnx.cpp
@@
-    float speed = 1.05f;
+    float speed = 1.0f;
```

```diff
--- a/csharp/Helper.cs
+++ b/csharp/Helper.cs
@@
-        private (float[] wav, float[] duration) _Infer(..., float speed = 1.05f)
+        private (float[] wav, float[] duration) _Infer(..., float speed = 1.0f)
@@
-        public (float[] wav, float[] duration) Call(..., float speed = 1.05f, float silenceDuration = 0.3f)
+        public (float[] wav, float[] duration) Call(..., float speed = 1.0f, float silenceDuration = 0.3f)
@@
-        public (float[] wav, float[] duration) Batch(..., float speed = 1.05f)
+        public (float[] wav, float[] duration) Batch(..., float speed = 1.0f)
```

```diff
--- a/csharp/ExampleONNX.cs
+++ b/csharp/ExampleONNX.cs
@@
-            public float Speed { get; set; } = 1.05f;
+            public float Speed { get; set; } = 1.0f;
```

```diff
--- a/go/example_onnx.go
+++ b/go/example_onnx.go
@@
-	flag.Float64Var(&args.speed, "speed", 1.05, "Speech speed factor (higher = faster)")
+	flag.Float64Var(&args.speed, "speed", 1.0, "Speech speed factor (higher = faster)")
```

```diff
--- a/swift/Sources/Helper.swift
+++ b/swift/Sources/Helper.swift
@@
-    private func _infer(..., speed: Float = 1.05) throws -> (wav: [Float], duration: [Float]) {
+    private func _infer(..., speed: Float = 1.0) throws -> (wav: [Float], duration: [Float]) {
@@
-    func call(..., speed: Float = 1.05, silenceDuration: Float = 0.3) throws -> (wav: [Float], duration: Float) {
+    func call(..., speed: Float = 1.0, silenceDuration: Float = 0.3) throws -> (wav: [Float], duration: Float) {
@@
-    func batch(..., speed: Float = 1.05) throws -> (wav: [Float], duration: [Float]) {
+    func batch(..., speed: Float = 1.0) throws -> (wav: [Float], duration: [Float]) {
```

```diff
--- a/swift/Sources/ExampleONNX.swift
+++ b/swift/Sources/ExampleONNX.swift
@@
-    var speed: Float = 1.05
+    var speed: Float = 1.0
@@
-                args.speed = Float(arguments[i + 1]) ?? 1.05
+                args.speed = Float(arguments[i + 1]) ?? 1.0
```

```diff
--- a/rust/src/example_onnx.rs
+++ b/rust/src/example_onnx.rs
@@
-    #[arg(long, default_value = "1.05")]
+    #[arg(long, default_value = "1.0")]
     speed: f32,
```

```diff
--- a/flutter/lib/helper.dart
+++ b/flutter/lib/helper.dart
@@
-      {double speed = 1.05, double silenceDuration = 0.3}) async {
+      {double speed = 1.0, double silenceDuration = 0.3}) async {
@@
-      {double speed = 1.05}) async {
+      {double speed = 1.0}) async {
```

```diff
--- a/flutter/lib/main.dart
+++ b/flutter/lib/main.dart
@@
-  double _speed = 1.05;
+  double _speed = 1.0;
```

Each `README.md` (root and per-language) that states "Default speed is
`1.05`" or shows `1.05` in a CLI options table would need the matching
one-line documentation update. Not included as diffs here since the exact
wording differs per binding's README.

## Alternative considered

Keep `1.05` as the default and instead document it as an intentional
loudness/pacing choice. Rejected as the basis for this PR because no such
rationale is recorded anywhere in the repository or its history to confirm
that was ever the intent, as opposed to an unexamined value carried over
from development.
