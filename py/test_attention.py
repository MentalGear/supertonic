"""Tests for py/attention.py -- the vector_estimator attention readout.

Run with:   python3 -m unittest discover -s py -p "test_*.py"
       or:  python3 py/test_attention.py

These tests need the real `assets/onnx/vector_estimator.onnx` (256 MB) and
run real inference, so they skip cleanly when that asset isn't present,
mirroring how this repo keeps model-dependent tests optional.

The instrumented graph is exported once for the whole module (`setUpClass`)
rather than per test, since a naive re-export writes a 256 MB file each time.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from helper import load_text_to_speech, load_voice_style
from attention import DEFAULT_ATTENTION_PATH, analyze, export_attention_graph

ONNX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "onnx")
VECTOR_ESTIMATOR_PATH = os.path.join(ONNX_DIR, "vector_estimator.onnx")
VOICE_STYLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "voice_styles", "M1.json"
)

TEXT = "The quick brown fox jumps over the lazy dog."
LANG = "en"
SEED = 0
TOTAL_STEP = 8
SPEED = 1.0

_HAS_ASSETS = os.path.exists(VECTOR_ESTIMATOR_PATH) and os.path.exists(VOICE_STYLE_PATH)

# Loaded lazily, once, and shared by every test class below -- both the ONNX
# sessions and the instrumented-graph export are expensive enough (256 MB
# each) that reloading them per test class would be wasteful.
_SHARED: dict = {}


def _shared_state() -> dict:
    if not _SHARED:
        _SHARED["tts"] = load_text_to_speech(ONNX_DIR)
        _SHARED["style"] = load_voice_style([VOICE_STYLE_PATH])
        export_attention_graph(VECTOR_ESTIMATOR_PATH, DEFAULT_ATTENTION_PATH)
    return _SHARED


@unittest.skipUnless(_HAS_ASSETS, "assets/onnx/vector_estimator.onnx not present")
class AttentionAssetTests(unittest.TestCase):
    """Base class that loads the model stack once for the whole hierarchy."""

    @classmethod
    def setUpClass(cls):
        state = _shared_state()
        cls.tts = state["tts"]
        cls.style = state["style"]

    @classmethod
    def analyze(cls):
        return analyze(
            cls.tts,
            TEXT,
            LANG,
            cls.style,
            total_step=TOTAL_STEP,
            speed=SPEED,
            seed=SEED,
            attention_path=DEFAULT_ATTENTION_PATH,
        )


@unittest.skipUnless(_HAS_ASSETS, "assets/onnx/vector_estimator.onnx not present")
class ExportIdempotencyTest(unittest.TestCase):
    def test_export_is_idempotent(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            dst = os.path.join(tmp_dir, "vector_estimator.attn.onnx")
            names_first = export_attention_graph(VECTOR_ESTIMATOR_PATH, dst)
            mtime_first = os.stat(dst).st_mtime_ns
            size_first = os.stat(dst).st_size

            names_second = export_attention_graph(VECTOR_ESTIMATOR_PATH, dst)
            mtime_second = os.stat(dst).st_mtime_ns
            size_second = os.stat(dst).st_size

            self.assertEqual(names_first, names_second)
            self.assertEqual(len(names_first), 8)
            self.assertEqual(mtime_first, mtime_second)
            self.assertEqual(size_first, size_second)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@unittest.skipUnless(_HAS_ASSETS, "assets/onnx/vector_estimator.onnx not present")
class AlignmentMonotonicTest(AttentionAssetTests):
    def test_alignment_is_monotonic(self):
        _wav, alignment, _style_attn = self.analyze()
        self.assertGreater(alignment.spearman, 0.95)

        argmax_seq = alignment.matrix.argmax(axis=-1)
        deltas = np.diff(argmax_seq)
        non_decreasing_fraction = np.mean(deltas >= 0)
        self.assertGreaterEqual(non_decreasing_fraction, 0.90)


@unittest.skipUnless(_HAS_ASSETS, "assets/onnx/vector_estimator.onnx not present")
class WordSpansTest(AttentionAssetTests):
    def test_word_spans_are_ordered_and_bounded(self):
        _wav, alignment, _style_attn = self.analyze()
        spans = alignment.word_spans()

        words = [w for w, _s, _e in spans]
        self.assertEqual(
            words,
            ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        )
        for word in words:
            self.assertNotIn("<", word)
            self.assertNotIn(">", word)

        starts = [s for _w, s, _e in spans]
        ends = [e for _w, _s, e in spans]
        for start, end in zip(starts, ends):
            self.assertLess(start, end)
            self.assertGreaterEqual(start, 0.0)
            self.assertLessEqual(end, alignment.duration)

        self.assertTrue(all(a <= b for a, b in zip(starts, starts[1:])))


@unittest.skipUnless(_HAS_ASSETS, "assets/onnx/vector_estimator.onnx not present")
class InstrumentedRenderMatchesHelperTest(AttentionAssetTests):
    def test_instrumented_render_matches_helper(self):
        """analyze() re-implements _infer's denoising loop by hand so it can
        collect the extra Softmax outputs. This test exists to catch that
        re-implementation drifting from helper.py's own loop: with the same
        text, voice, seed, steps and speed, the two must render
        bit-identical audio. If this ever legitimately fails (e.g. ORT
        reorders nodes differently once extra outputs are requested), do NOT
        loosen this to allclose -- report the measured max abs difference
        instead."""
        wav_helper, _dur = self.tts._infer(
            [TEXT], [LANG], self.style, TOTAL_STEP, speed=SPEED, seed=SEED
        )
        wav_analyze, _alignment, _style_attn = self.analyze()

        self.assertEqual(wav_helper.shape, wav_analyze.shape)
        np.testing.assert_array_equal(wav_helper, wav_analyze)


if __name__ == "__main__":
    unittest.main()
