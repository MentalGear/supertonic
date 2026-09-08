/**
 * Tests for the style composition algebra in helper.js.
 *
 * Run with:   node --test web/helper.test.js
 *        or:  node web/helper.test.js
 *
 * No test framework is required -- this uses node:test and node:assert only.
 * helper.js imports onnxruntime-web for the inference path, which is not
 * installed (and not needed) for the Style math, so the module is stubbed with
 * a minimal Tensor before helper.js is imported.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const TENSOR_STUB =
    'data:text/javascript,' +
    'export class Tensor{constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims;}}';

registerHooks({
    resolve(specifier, context, nextResolve) {
        if (specifier === 'onnxruntime-web') {
            return { url: TENSOR_STUB, shortCircuit: true };
        }
        return nextResolve(specifier, context);
    }
});

const { Style } = await import('./helper.js');

// --- test fixtures -------------------------------------------------------

const TTL_ROWS = 50;
const TTL_COLS = 256;
const DP_ROWS = 8;
const DP_COLS = 16;

class Tensor {
    constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
    }
}

/** Deterministic LCG so every run sees the same tensors. */
function makeRng(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296 - 0.5;
    };
}

function rowNorms(data, rowLength) {
    const norms = [];
    for (let start = 0; start < data.length; start += rowLength) {
        let sum = 0;
        for (let i = 0; i < rowLength; i++) sum += data[start + i] ** 2;
        norms.push(Math.sqrt(sum));
    }
    return norms;
}

/** Random tensor whose rows are unit norm, like the released presets. */
function unitTensor(batch, rows, cols, seed) {
    const rng = makeRng(seed);
    const data = new Float32Array(batch * rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = rng();
    for (let start = 0; start < data.length; start += cols) {
        let sum = 0;
        for (let i = 0; i < cols; i++) sum += data[start + i] ** 2;
        const norm = Math.sqrt(sum);
        for (let i = 0; i < cols; i++) data[start + i] /= norm;
    }
    return new Tensor('float32', data, [batch, rows, cols]);
}

function rawTensor(batch, rows, cols, seed, scale = 1) {
    const rng = makeRng(seed);
    const data = new Float32Array(batch * rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = rng() * scale;
    return new Tensor('float32', data, [batch, rows, cols]);
}

function makeStyle(batch, seed, { unit = true } = {}) {
    const ttl = unit
        ? unitTensor(batch, TTL_ROWS, TTL_COLS, seed)
        : rawTensor(batch, TTL_ROWS, TTL_COLS, seed, 3);
    const dp = unit
        ? unitTensor(batch, DP_ROWS, DP_COLS, seed + 7)
        : rawTensor(batch, DP_ROWS, DP_COLS, seed + 7, 3);
    return new Style(ttl, dp);
}

function makeDelta(batch, seed) {
    return new Style(
        rawTensor(batch, TTL_ROWS, TTL_COLS, seed, 0.2),
        rawTensor(batch, DP_ROWS, DP_COLS, seed + 11, 0.2)
    );
}

function assertClose(actual, expected, tolerance, message) {
    assert.equal(actual.length, expected.length, `${message}: length`);
    let worst = 0;
    for (let i = 0; i < actual.length; i++) {
        worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
    }
    assert.ok(worst <= tolerance, `${message}: max abs diff ${worst} > ${tolerance}`);
}

function assertExact(actual, expected, message) {
    assert.equal(actual.length, expected.length, `${message}: length`);
    for (let i = 0; i < actual.length; i++) {
        assert.equal(actual[i], expected[i], `${message}: element ${i}`);
    }
}

// --- zero weights are an exact no-op -------------------------------------

test('empty delta list is an exact no-op', () => {
    const base = makeStyle(1, 1);
    const out = base.withDeltas([]);
    assertExact(out.ttl.data, base.ttl.data, 'ttl');
    assertExact(out.dp.data, base.dp.data, 'dp');
    assert.deepEqual(out.ttl.dims, base.ttl.dims);
});

test('all-zero weights are an exact no-op, even with include duration', () => {
    const base = makeStyle(1, 2);
    const out = base.withDeltas([[makeDelta(1, 3), 0], [makeDelta(1, 4), 0]], true);
    assertExact(out.ttl.data, base.ttl.data, 'ttl');
    assertExact(out.dp.data, base.dp.data, 'dp');
});

test('zero weight is an exact no-op on a base with non-unit rows', () => {
    const base = makeStyle(1, 5, { unit: false });
    const out = base.withDeltas([[makeDelta(1, 6), 0]]);
    assertExact(out.ttl.data, base.ttl.data, 'ttl');
});

test('withEmotion at intensity 0 is an exact no-op (regression: used to rescale every row)', () => {
    const base = makeStyle(1, 7);
    const out = base.withEmotion(makeDelta(1, 8), 0);
    assertExact(out.ttl.data, base.ttl.data, 'ttl');
    assertExact(out.dp.data, base.dp.data, 'dp');
});

test('withEmotion at intensity 0 is a no-op for non-unit rows too', () => {
    const base = makeStyle(1, 9, { unit: false });
    const out = base.withEmotion(makeDelta(1, 10), 0);
    assertExact(out.ttl.data, base.ttl.data, 'ttl');
});

// --- single delta matches the reference blend ----------------------------

test('single delta matches add-then-per-row-normalize for unit-norm input', () => {
    const base = makeStyle(1, 11);
    const delta = makeDelta(1, 12);
    const intensity = 0.6;

    // Reference: the intended semantics, computed independently in float64.
    const expected = new Float64Array(base.ttl.data.length);
    for (let i = 0; i < expected.length; i++) {
        expected[i] = base.ttl.data[i] + intensity * delta.ttl.data[i];
    }
    for (let start = 0; start < expected.length; start += TTL_COLS) {
        let sum = 0;
        for (let i = 0; i < TTL_COLS; i++) sum += expected[start + i] ** 2;
        const norm = Math.sqrt(sum);
        for (let i = 0; i < TTL_COLS; i++) expected[start + i] /= norm;
    }

    const viaEmotion = base.withEmotion(delta, intensity);
    const viaDeltas = base.withDeltas([[delta, intensity]]);
    assertClose(viaEmotion.ttl.data, expected, 1e-6, 'withEmotion vs reference');
    assertExact(viaDeltas.ttl.data, viaEmotion.ttl.data, 'withDeltas vs withEmotion');
});

// --- order independence (commutativity) ----------------------------------

test('composition is order independent', () => {
    const base = makeStyle(1, 13);
    const a = makeDelta(1, 14);
    const b = makeDelta(1, 15);
    const c = makeDelta(1, 16);

    const ab = base.withDeltas([[a, 0.4], [b, -0.3], [c, 0.9]], true);
    const ba = base.withDeltas([[c, 0.9], [b, -0.3], [a, 0.4]], true);
    assertClose(ab.ttl.data, ba.ttl.data, 1e-6, 'ttl order independence');
    assertClose(ab.dp.data, ba.dp.data, 1e-6, 'dp order independence');
});

test('sequential blends are NOT equivalent to one composed blend (documents why with_deltas exists)', () => {
    const base = makeStyle(1, 17);
    const a = makeDelta(1, 18);
    const b = makeDelta(1, 19);

    const composed = base.withDeltas([[a, 0.5], [b, 0.5]]);
    const sequential = base.withDeltas([[a, 0.5]]).withDeltas([[b, 0.5]]);
    let worst = 0;
    for (let i = 0; i < composed.ttl.data.length; i++) {
        worst = Math.max(worst, Math.abs(composed.ttl.data[i] - sequential.ttl.data[i]));
    }
    assert.ok(worst > 1e-4, 'sequential blending should differ from single-shot composition');
});

// --- row norms preserved -------------------------------------------------

test('TTL row norms are preserved after blending', () => {
    const base = makeStyle(1, 20);
    const out = base.withDeltas([[makeDelta(1, 21), 0.8], [makeDelta(1, 22), -0.5]]);
    const before = rowNorms(base.ttl.data, TTL_COLS);
    const after = rowNorms(out.ttl.data, TTL_COLS);
    assert.equal(after.length, TTL_ROWS);
    for (let i = 0; i < after.length; i++) {
        assert.ok(Math.abs(after[i] - 1) < 1e-5, `row ${i} norm ${after[i]} is not unit`);
        assert.ok(Math.abs(after[i] - before[i]) < 1e-5, `row ${i} norm drifted`);
    }
});

test('non-unit TTL row norms are restored to their original values, not to 1', () => {
    const base = makeStyle(1, 23, { unit: false });
    const out = base.withDeltas([[makeDelta(1, 24), 1.0]]);
    const before = rowNorms(base.ttl.data, TTL_COLS);
    const after = rowNorms(out.ttl.data, TTL_COLS);
    for (let i = 0; i < after.length; i++) {
        assert.ok(Math.abs(after[i] - before[i]) / before[i] < 1e-5, `row ${i}: ${after[i]} vs ${before[i]}`);
        assert.ok(Math.abs(after[i] - 1) > 1e-3, `row ${i} was forced to unit norm`);
    }
});

test('weights beyond [0, 1] are accepted and not clamped', () => {
    const base = makeStyle(1, 25);
    const delta = makeDelta(1, 26);
    const strong = base.withDeltas([[delta, 3.5]]);
    const nominal = base.withDeltas([[delta, 1.0]]);
    let worst = 0;
    for (let i = 0; i < strong.ttl.data.length; i++) {
        worst = Math.max(worst, Math.abs(strong.ttl.data[i] - nominal.ttl.data[i]));
    }
    assert.ok(worst > 1e-4, 'weight 3.5 was clamped to 1.0');
    const negative = base.withDeltas([[delta, -2.0]]);
    assert.ok(negative.ttl.data.every(Number.isFinite));
    for (const norm of rowNorms(negative.ttl.data, TTL_COLS)) {
        assert.ok(Math.abs(norm - 1) < 1e-5);
    }
});

// --- batch broadcasting --------------------------------------------------

test('a single delta broadcasts across a batched voice', () => {
    const batched = makeStyle(3, 27);
    const delta = makeDelta(1, 28);
    const blended = batched.withDeltas([[delta, 0.7]]);
    assert.deepEqual(blended.ttl.dims, [3, TTL_ROWS, TTL_COLS]);

    const stride = TTL_ROWS * TTL_COLS;
    for (let batch = 0; batch < 3; batch++) {
        // Same batch item, blended on its own, must match the batched result.
        const single = new Style(
            new Tensor('float32', batched.ttl.data.slice(batch * stride, (batch + 1) * stride), [1, TTL_ROWS, TTL_COLS]),
            new Tensor('float32', batched.dp.data.slice(batch * DP_ROWS * DP_COLS, (batch + 1) * DP_ROWS * DP_COLS), [1, DP_ROWS, DP_COLS])
        ).withDeltas([[delta, 0.7]]);
        assertExact(
            blended.ttl.data.slice(batch * stride, (batch + 1) * stride),
            single.ttl.data,
            `batch ${batch}`
        );
    }
});

test('a batched delta applies elementwise to a batched voice', () => {
    const batched = makeStyle(2, 29);
    const delta = makeDelta(2, 30);
    const blended = batched.withDeltas([[delta, 1.0]]);
    const stride = TTL_ROWS * TTL_COLS;
    const second = new Style(
        new Tensor('float32', batched.ttl.data.slice(stride), [1, TTL_ROWS, TTL_COLS]),
        new Tensor('float32', batched.dp.data.slice(DP_ROWS * DP_COLS), [1, DP_ROWS, DP_COLS])
    ).withDeltas([[
        new Style(
            new Tensor('float32', delta.ttl.data.slice(stride), [1, TTL_ROWS, TTL_COLS]),
            new Tensor('float32', delta.dp.data.slice(DP_ROWS * DP_COLS), [1, DP_ROWS, DP_COLS])
        ), 1.0]]);
    assertExact(blended.ttl.data.slice(stride), second.ttl.data, 'second batch item');
});

// --- duration handling ---------------------------------------------------

test('DP is untouched when includeDuration is false', () => {
    const base = makeStyle(1, 31);
    const out = base.withDeltas([[makeDelta(1, 32), 1.0]]);
    assertExact(out.dp.data, base.dp.data, 'dp');
    // ... and TTL did move.
    let worst = 0;
    for (let i = 0; i < out.ttl.data.length; i++) {
        worst = Math.max(worst, Math.abs(out.ttl.data[i] - base.ttl.data[i]));
    }
    assert.ok(worst > 1e-4, 'ttl should have changed');
});

test('DP moves and keeps its row norms when includeDuration is true', () => {
    const base = makeStyle(1, 33);
    const out = base.withDeltas([[makeDelta(1, 34), 1.0]], true);
    let worst = 0;
    for (let i = 0; i < out.dp.data.length; i++) {
        worst = Math.max(worst, Math.abs(out.dp.data[i] - base.dp.data[i]));
    }
    assert.ok(worst > 1e-4, 'dp should have changed');
    for (const norm of rowNorms(out.dp.data, DP_COLS)) {
        assert.ok(Math.abs(norm - 1) < 1e-5, `dp row norm ${norm} is not unit`);
    }
});

// --- malformed input -----------------------------------------------------

test('malformed input is rejected', () => {
    const base = makeStyle(2, 35);
    const delta = makeDelta(1, 36);

    assert.throws(() => base.withDeltas(delta), /array of \[style, weight\] pairs/);
    assert.throws(() => base.withDeltas([delta]), /\[style, weight\] pair/);
    assert.throws(() => base.withDeltas([[delta]]), /\[style, weight\] pair/);
    assert.throws(() => base.withDeltas([[delta, 1, 2]]), /\[style, weight\] pair/);
    assert.throws(() => base.withDeltas([[null, 1]]), /must be a Style instance/);
    assert.throws(() => base.withDeltas([[delta, NaN]]), /finite number/);
    assert.throws(() => base.withDeltas([[delta, Infinity]]), /finite number/);
    assert.throws(() => base.withDeltas([[delta, '1']]), /finite number/);

    const wrongTtl = new Style(
        rawTensor(1, TTL_ROWS, 128, 37),
        rawTensor(1, DP_ROWS, DP_COLS, 38)
    );
    assert.throws(() => base.withDeltas([[wrongTtl, 1]]), /TTL dimensions do not match/);

    const wrongDp = new Style(
        rawTensor(1, TTL_ROWS, TTL_COLS, 39),
        rawTensor(1, DP_ROWS, 8, 40)
    );
    assert.throws(() => base.withDeltas([[wrongDp, 1]]), /DP dimensions do not match/);

    const wrongBatch = makeDelta(3, 41);
    assert.throws(() => base.withDeltas([[wrongBatch, 1]]), /batch dimensions do not match/);
});

test('withEmotion keeps its [0, 1] intensity constraint and its error messages', () => {
    const base = makeStyle(1, 42);
    const delta = makeDelta(1, 43);
    assert.throws(() => base.withEmotion(delta, 1.5), /Emotion intensity must be between 0 and 1/);
    assert.throws(() => base.withEmotion(delta, -0.1), /Emotion intensity must be between 0 and 1/);

    const wrongTtl = new Style(rawTensor(1, TTL_ROWS, 128, 44), rawTensor(1, DP_ROWS, DP_COLS, 45));
    assert.throws(() => base.withEmotion(wrongTtl, 1), /Emotion TTL dimensions do not match the voice style/);

    const wrongDp = new Style(rawTensor(1, TTL_ROWS, TTL_COLS, 46), rawTensor(1, DP_ROWS, 8, 47));
    assert.throws(() => base.withEmotion(wrongDp, 1), /Emotion DP dimensions do not match the voice style/);

    const wrongBatch = makeDelta(4, 48);
    assert.throws(() => base.withEmotion(wrongBatch, 1), /Emotion batch dimensions do not match the voice style/);
});
