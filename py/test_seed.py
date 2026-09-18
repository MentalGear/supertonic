"""Tests for the optional `seed` parameter on TextToSpeech.sample_noisy_latent.

Run with:   python -m unittest py.test_seed
       or:  python py/test_seed.py
       or:  pytest py/test_seed.py

These tests exercise `sample_noisy_latent` directly on a `TextToSpeech`
instance built with dummy ONNX sessions (None) -- `sample_noisy_latent` never
touches the sessions, only `self.sample_rate` / `self.base_chunk_size` /
`self.chunk_compress_factor` / `self.ldim`, all of which come from `cfgs`.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:  # pragma: no cover - environment dependent
    import onnxruntime  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    _stub = types.ModuleType("onnxruntime")
    _stub.InferenceSession = object
    _stub.SessionOptions = object
    sys.modules["onnxruntime"] = _stub

import numpy as np

from helper import TextToSpeech


def make_tts() -> TextToSpeech:
    cfgs = {
        "ae": {"sample_rate": 24000, "base_chunk_size": 32},
        "ttl": {"chunk_compress_factor": 4, "latent_dim": 8},
    }
    return TextToSpeech(cfgs, None, None, None, None, None)


class SeededDrawTest(unittest.TestCase):
    def test_same_seed_twice_is_identical(self):
        tts = make_tts()
        duration = np.array([1.0, 1.5], dtype=np.float32)
        a, mask_a = tts.sample_noisy_latent(duration, seed=123)
        b, mask_b = tts.sample_noisy_latent(duration, seed=123)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask_a, mask_b)

    def test_different_seeds_differ(self):
        tts = make_tts()
        duration = np.array([1.0, 1.5], dtype=np.float32)
        a, _ = tts.sample_noisy_latent(duration, seed=1)
        b, _ = tts.sample_noisy_latent(duration, seed=2)
        self.assertFalse(np.array_equal(a, b))

    def test_seeded_draw_does_not_touch_global_randn(self):
        """A seeded call must use a local generator, never the global RNG --
        reseeding the global RNG would leak into the caller's process."""
        tts = make_tts()
        duration = np.array([1.0, 1.5], dtype=np.float32)
        randn_calls = []
        orig_randn = np.random.randn

        def spy_randn(*args, **kwargs):
            randn_calls.append(args)
            return orig_randn(*args, **kwargs)

        np.random.randn = spy_randn
        try:
            tts.sample_noisy_latent(duration, seed=7)
        finally:
            np.random.randn = orig_randn
        self.assertEqual(len(randn_calls), 0)


class UnseededDrawUnchangedTest(unittest.TestCase):
    def test_no_seed_still_uses_global_randn_not_default_rng(self):
        """Default behaviour (seed=None) must still be exactly the old code
        path: draw from np.random.randn, and never construct a local
        default_rng."""
        tts = make_tts()
        duration = np.array([1.0, 1.5], dtype=np.float32)
        randn_calls = []
        default_rng_calls = []
        orig_randn = np.random.randn
        orig_default_rng = np.random.default_rng

        def spy_randn(*args, **kwargs):
            randn_calls.append(args)
            return orig_randn(*args, **kwargs)

        def spy_default_rng(*args, **kwargs):
            default_rng_calls.append(args)
            return orig_default_rng(*args, **kwargs)

        np.random.randn = spy_randn
        np.random.default_rng = spy_default_rng
        try:
            tts.sample_noisy_latent(duration)
        finally:
            np.random.randn = orig_randn
            np.random.default_rng = orig_default_rng

        self.assertEqual(len(randn_calls), 1)
        self.assertEqual(len(default_rng_calls), 0)

    def test_no_seed_reproduces_under_identical_global_rng_state(self):
        """Given the same global RNG state (as every experiment script in
        this repo arranges by hand), an unseeded call must be exactly
        reproducible -- proving seed=None truly falls through to the
        pre-existing behaviour rather than e.g. always drawing fresh
        entropy."""
        tts = make_tts()
        duration = np.array([1.0, 1.5], dtype=np.float32)

        np.random.seed(42)
        a, mask_a = tts.sample_noisy_latent(duration)
        np.random.seed(42)
        b, mask_b = tts.sample_noisy_latent(duration)

        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask_a, mask_b)

    def test_no_seed_two_calls_differ(self):
        """Without manually reseeding in between, two unseeded calls must
        still give different audio -- the whole point being fixed."""
        tts = make_tts()
        duration = np.array([1.0, 1.5], dtype=np.float32)
        a, _ = tts.sample_noisy_latent(duration)
        b, _ = tts.sample_noisy_latent(duration)
        self.assertFalse(np.array_equal(a, b))


class MultiChunkSeedDerivationTest(unittest.TestCase):
    """`__call__` chunks long text and must derive a distinct per-chunk seed
    (`seed + chunk_index`) rather than passing the same seed to every chunk's
    `_infer` call. These tests stub `_infer` so they exercise only the
    chunking/seed-derivation loop, not the full ONNX pipeline."""

    LONG_TEXT = "First sentence.\n\nSecond sentence."  # -> two chunks, one per paragraph

    @staticmethod
    def dummy_style():
        # __call__ only checks style.ttl.shape[0] == 1 before chunking;
        # _infer itself is stubbed out below so nothing else touches style.
        return types.SimpleNamespace(ttl=np.zeros((1, 50, 256), dtype=np.float32))

    def make_recording_tts(self, seeds_seen):
        tts = make_tts()

        def fake_infer(text_list, lang_list, style, total_step, speed, seed=None):
            seeds_seen.append(seed)
            return np.zeros((1, 4), dtype=np.float32), np.array([1.0])

        tts._infer = fake_infer
        return tts

    def assert_two_chunks(self, text_list):
        # Sanity check the fixture actually produces >1 chunk; otherwise the
        # test below would pass vacuously.
        from helper import chunk_text

        self.assertEqual(len(chunk_text(text_list, max_len=300)), 2)

    def test_chunks_receive_distinct_derived_seeds(self):
        self.assert_two_chunks(self.LONG_TEXT)
        seeds_seen = []
        tts = self.make_recording_tts(seeds_seen)
        tts(self.LONG_TEXT, "en", style=self.dummy_style(), total_step=1, seed=100)
        self.assertEqual(seeds_seen, [100, 101])

    def test_same_user_seed_twice_gives_identical_per_chunk_seeds(self):
        """Same user seed + same text -> same per-chunk seeds every time,
        which is what makes the end-to-end audio reproducible."""
        self.assert_two_chunks(self.LONG_TEXT)
        seeds_a, seeds_b = [], []
        self.make_recording_tts(seeds_a)(
            self.LONG_TEXT, "en", style=self.dummy_style(), total_step=1, seed=7
        )
        self.make_recording_tts(seeds_b)(
            self.LONG_TEXT, "en", style=self.dummy_style(), total_step=1, seed=7
        )
        self.assertEqual(seeds_a, seeds_b)

    def test_single_chunk_seed_unchanged(self):
        """A single-chunk render must still receive the raw seed unchanged
        (chunk_index 0 -> seed + 0 == seed): existing single-chunk callers
        see identical behaviour to before this change."""
        seeds_seen = []
        tts = self.make_recording_tts(seeds_seen)
        tts("Short text.", "en", style=self.dummy_style(), total_step=1, seed=42)
        self.assertEqual(seeds_seen, [42])

    def test_unseeded_multi_chunk_stays_none(self):
        self.assert_two_chunks(self.LONG_TEXT)
        seeds_seen = []
        tts = self.make_recording_tts(seeds_seen)
        tts(self.LONG_TEXT, "en", style=self.dummy_style(), total_step=1)
        self.assertEqual(seeds_seen, [None, None])


if __name__ == "__main__":
    unittest.main()
