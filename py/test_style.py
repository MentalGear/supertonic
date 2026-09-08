"""Tests for the style composition algebra in helper.py.

Run with:   python -m unittest py.test_style
       or:  python py/test_style.py
       or:  pytest py/test_style.py

Requires numpy. onnxruntime is stubbed out if it is not installed, since the
Style math does not need it.
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

from helper import Style

TTL_ROWS, TTL_COLS = 50, 256
DP_ROWS, DP_COLS = 8, 16


def raw_tensor(batch, rows, cols, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch, rows, cols)) * scale).astype(np.float32)


def unit_tensor(batch, rows, cols, seed):
    data = raw_tensor(batch, rows, cols, seed)
    return (data / np.linalg.norm(data, axis=-1, keepdims=True)).astype(np.float32)


def make_style(batch, seed, unit=True):
    if unit:
        return Style(
            unit_tensor(batch, TTL_ROWS, TTL_COLS, seed),
            unit_tensor(batch, DP_ROWS, DP_COLS, seed + 7),
        )
    return Style(
        raw_tensor(batch, TTL_ROWS, TTL_COLS, seed, 3.0),
        raw_tensor(batch, DP_ROWS, DP_COLS, seed + 7, 3.0),
    )


def make_delta(batch, seed):
    return Style(
        raw_tensor(batch, TTL_ROWS, TTL_COLS, seed, 0.2),
        raw_tensor(batch, DP_ROWS, DP_COLS, seed + 11, 0.2),
    )


class ZeroWeightNoOpTest(unittest.TestCase):
    def test_empty_delta_list_is_exact_no_op(self):
        base = make_style(1, 1)
        out = base.with_deltas([])
        self.assertTrue(np.array_equal(out.ttl, base.ttl))
        self.assertTrue(np.array_equal(out.dp, base.dp))
        self.assertEqual(out.ttl.shape, base.ttl.shape)

    def test_all_zero_weights_are_exact_no_op_with_duration(self):
        base = make_style(1, 2)
        out = base.with_deltas(
            [(make_delta(1, 3), 0.0), (make_delta(1, 4), 0.0)], include_duration=True
        )
        self.assertTrue(np.array_equal(out.ttl, base.ttl))
        self.assertTrue(np.array_equal(out.dp, base.dp))

    def test_zero_weight_is_exact_no_op_for_non_unit_rows(self):
        base = make_style(1, 5, unit=False)
        out = base.with_deltas([(make_delta(1, 6), 0.0)])
        self.assertTrue(np.array_equal(out.ttl, base.ttl))

    def test_with_emotion_at_intensity_zero_is_exact_no_op(self):
        # Regression: the old code returned normalize(self.ttl) at intensity 0.
        base = make_style(1, 7)
        out = base.with_emotion(make_delta(1, 8), 0.0)
        self.assertTrue(np.array_equal(out.ttl, base.ttl))
        self.assertTrue(np.array_equal(out.dp, base.dp))

    def test_with_emotion_at_intensity_zero_is_no_op_for_non_unit_rows(self):
        base = make_style(1, 9, unit=False)
        out = base.with_emotion(make_delta(1, 10), 0.0)
        self.assertTrue(np.array_equal(out.ttl, base.ttl))

    def test_no_op_result_does_not_alias_the_base(self):
        base = make_style(1, 11)
        out = base.with_deltas([])
        out.ttl[0, 0, 0] += 1.0
        out.dp[0, 0, 0] += 1.0
        self.assertNotEqual(out.ttl[0, 0, 0], base.ttl[0, 0, 0])
        self.assertNotEqual(out.dp[0, 0, 0], base.dp[0, 0, 0])


class SingleDeltaTest(unittest.TestCase):
    def test_single_delta_matches_add_then_row_normalize(self):
        base = make_style(1, 12)
        delta = make_delta(1, 13)
        intensity = 0.6

        expected = base.ttl.astype(np.float64) + intensity * delta.ttl.astype(np.float64)
        expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)

        via_emotion = base.with_emotion(delta, intensity)
        via_deltas = base.with_deltas([(delta, intensity)])
        np.testing.assert_allclose(via_emotion.ttl, expected, atol=1e-6)
        np.testing.assert_array_equal(via_deltas.ttl, via_emotion.ttl)


class OrderIndependenceTest(unittest.TestCase):
    def test_composition_is_order_independent(self):
        base = make_style(1, 14)
        a, b, c = make_delta(1, 15), make_delta(1, 16), make_delta(1, 17)
        forward = base.with_deltas(
            [(a, 0.4), (b, -0.3), (c, 0.9)], include_duration=True
        )
        reverse = base.with_deltas(
            [(c, 0.9), (b, -0.3), (a, 0.4)], include_duration=True
        )
        np.testing.assert_allclose(forward.ttl, reverse.ttl, atol=1e-6)
        np.testing.assert_allclose(forward.dp, reverse.dp, atol=1e-6)

    def test_sequential_blends_differ_from_composed_blend(self):
        # Documents why with_deltas exists: normalizing between deltas is not
        # the same operation, and is not commutative.
        base = make_style(1, 18)
        a, b = make_delta(1, 19), make_delta(1, 20)
        composed = base.with_deltas([(a, 0.5), (b, 0.5)])
        sequential = base.with_deltas([(a, 0.5)]).with_deltas([(b, 0.5)])
        self.assertGreater(np.abs(composed.ttl - sequential.ttl).max(), 1e-4)


class RowNormTest(unittest.TestCase):
    def test_unit_row_norms_are_preserved(self):
        base = make_style(1, 21)
        out = base.with_deltas([(make_delta(1, 22), 0.8), (make_delta(1, 23), -0.5)])
        norms = np.linalg.norm(out.ttl, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_non_unit_row_norms_are_restored_not_forced_to_unit(self):
        base = make_style(1, 24, unit=False)
        out = base.with_deltas([(make_delta(1, 25), 1.0)])
        before = np.linalg.norm(base.ttl, axis=-1)
        after = np.linalg.norm(out.ttl, axis=-1)
        np.testing.assert_allclose(after, before, rtol=1e-5)
        self.assertGreater(np.abs(after - 1.0).min(), 1e-3)

    def test_weights_are_not_clamped(self):
        base = make_style(1, 26)
        delta = make_delta(1, 27)
        strong = base.with_deltas([(delta, 3.5)])
        nominal = base.with_deltas([(delta, 1.0)])
        self.assertGreater(np.abs(strong.ttl - nominal.ttl).max(), 1e-4)

        negative = base.with_deltas([(delta, -2.0)])
        self.assertTrue(np.all(np.isfinite(negative.ttl)))
        norms = np.linalg.norm(negative.ttl, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_output_dtype_is_float32(self):
        base = make_style(1, 28)
        out = base.with_deltas([(make_delta(1, 29), 0.5)], include_duration=True)
        self.assertEqual(out.ttl.dtype, np.float32)
        self.assertEqual(out.dp.dtype, np.float32)


class BroadcastTest(unittest.TestCase):
    def test_single_delta_broadcasts_across_batched_voice(self):
        batched = make_style(3, 30)
        delta = make_delta(1, 31)
        blended = batched.with_deltas([(delta, 0.7)])
        self.assertEqual(blended.ttl.shape, (3, TTL_ROWS, TTL_COLS))
        for i in range(3):
            single = Style(batched.ttl[i : i + 1], batched.dp[i : i + 1])
            expected = single.with_deltas([(delta, 0.7)])
            np.testing.assert_array_equal(blended.ttl[i : i + 1], expected.ttl)

    def test_batched_delta_applies_elementwise(self):
        batched = make_style(2, 32)
        delta = make_delta(2, 33)
        blended = batched.with_deltas([(delta, 1.0)])
        for i in range(2):
            single = Style(batched.ttl[i : i + 1], batched.dp[i : i + 1])
            expected = single.with_deltas(
                [(Style(delta.ttl[i : i + 1], delta.dp[i : i + 1]), 1.0)]
            )
            np.testing.assert_array_equal(blended.ttl[i : i + 1], expected.ttl)


class DurationTest(unittest.TestCase):
    def test_dp_untouched_when_include_duration_is_false(self):
        base = make_style(1, 34)
        out = base.with_deltas([(make_delta(1, 35), 1.0)])
        self.assertTrue(np.array_equal(out.dp, base.dp))
        self.assertGreater(np.abs(out.ttl - base.ttl).max(), 1e-4)

    def test_dp_moves_and_keeps_row_norms_when_included(self):
        base = make_style(1, 36)
        out = base.with_deltas([(make_delta(1, 37), 1.0)], include_duration=True)
        self.assertGreater(np.abs(out.dp - base.dp).max(), 1e-4)
        norms = np.linalg.norm(out.dp, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.base = make_style(2, 38)
        self.delta = make_delta(1, 39)

    def test_malformed_delta_entries_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\(Style, weight\) pair"):
            self.base.with_deltas([self.delta])
        with self.assertRaisesRegex(ValueError, r"\(Style, weight\) pair"):
            self.base.with_deltas([(self.delta,)])
        with self.assertRaisesRegex(ValueError, r"\(Style, weight\) pair"):
            self.base.with_deltas([(self.delta, 1.0, 2.0)])
        with self.assertRaisesRegex(ValueError, "must be a Style instance"):
            self.base.with_deltas([(None, 1.0)])

    def test_non_finite_weights_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf"), "1.0", None):
            with self.assertRaisesRegex(ValueError, "finite number"):
                self.base.with_deltas([(self.delta, bad)])

    def test_shape_and_batch_validation(self):
        wrong_ttl = Style(
            raw_tensor(1, TTL_ROWS, 128, 40), raw_tensor(1, DP_ROWS, DP_COLS, 41)
        )
        with self.assertRaisesRegex(ValueError, "TTL shape .* does not match voice TTL shape"):
            self.base.with_deltas([(wrong_ttl, 1.0)])

        wrong_dp = Style(
            raw_tensor(1, TTL_ROWS, TTL_COLS, 42), raw_tensor(1, DP_ROWS, 8, 43)
        )
        with self.assertRaisesRegex(ValueError, "DP shape .* does not match voice DP shape"):
            self.base.with_deltas([(wrong_dp, 1.0)])

        wrong_batch = make_delta(3, 44)
        with self.assertRaisesRegex(ValueError, "TTL batch 3 does not match voice batch 2"):
            self.base.with_deltas([(wrong_batch, 1.0)])

    def test_with_emotion_keeps_its_constraints_and_messages(self):
        base = make_style(1, 45)
        delta = make_delta(1, 46)
        with self.assertRaisesRegex(ValueError, "Emotion intensity must be between 0.0 and 1.0"):
            base.with_emotion(delta, 1.5)
        with self.assertRaisesRegex(ValueError, "Emotion intensity must be between 0.0 and 1.0"):
            base.with_emotion(delta, -0.1)

        wrong_ttl = Style(
            raw_tensor(1, TTL_ROWS, 128, 47), raw_tensor(1, DP_ROWS, DP_COLS, 48)
        )
        with self.assertRaisesRegex(
            ValueError, "Emotion TTL shape .* does not match voice TTL shape"
        ):
            base.with_emotion(wrong_ttl, 1.0)

        wrong_dp = Style(
            raw_tensor(1, TTL_ROWS, TTL_COLS, 49), raw_tensor(1, DP_ROWS, 8, 50)
        )
        with self.assertRaisesRegex(
            ValueError, "Emotion DP shape .* does not match voice DP shape"
        ):
            base.with_emotion(wrong_dp, 1.0)

        batched = make_style(2, 51)
        wrong_batch = make_delta(3, 52)
        with self.assertRaisesRegex(
            ValueError, "Emotion TTL batch 3 does not match voice batch 2"
        ):
            batched.with_emotion(wrong_batch, 1.0)


if __name__ == "__main__":
    unittest.main()
