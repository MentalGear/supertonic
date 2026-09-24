"""Phase 3 / task 18: read the style-row attention across the Phase 2b
perturbation-ladder corpora -- calibration first.

The thing this script is built to avoid is the failure mode CLAUDE.md records
three times over: citing a distance as evidence without calibrating it against
known-same and known-different pairs. The attention maps returned by
`attention.analyze()` are a function of `noisy_latent` as well as of
`style_ttl`, and `noisy_latent` comes out of the vocoder RNG. So the *same*
style rendered at two seeds gives two different attention maps, and every
distance computed on attention has a noise floor that has to be measured
before any ladder number means anything.

Order of operations, enforced by the code and not just by the prose:

  Stage 0  sanity      base_ttl == M1's style_ttl exactly; repeated render at
                       the same seed is bit-identical; L constant per set.
  Stage 1  calibration NULL (base_ttl at 24 seeds -> same-style noise floor)
                       vs KNOWN-DIFFERENT (the 10 shipped presets' style_ttl,
                       with M1's style_dp substituted so L and duration stay
                       pinned). Both distributions are reported with
                       mean/sd/min/max, an overlap interval, and a rank AUC.
                       If they overlap substantially the ladder stage is
                       *skipped* -- `--force-ladder` overrides, and the output
                       records that it was forced.
  Stage 2  ladder      N=80 styles from each of ttl_K4/K16/K64 on the primary
                       text at ONE vocoder seed, answering the brief's six
                       numbered questions, with a shuffled-row control on the
                       active-row question.
  Stage 3  generality  a subset on the longer text, headline result only.
  Stage 4  secondary   (--presetspan) preset_span vs random_control vs
                       row_matched_random, the three-way that mattered to the
                       earlier probe work.

Distances are defined on the row-attention profile
`w = StyleAttention.matrix.mean(axis=0)` (a 50-vector summing to 1):

    utterance TV     0.5 * |w_a - w_b|.sum()
    frame TV         mean over frames of 0.5 * |M_a[f] - M_b[f]|.sum()

The frame-resolved variant exists because an utterance-level mean can cancel a
localized change -- the same reason CLAUDE.md says to diff spectrograms before
reaching for aggregates. It is only meaningful because `style_dp` is held at
M1's for every render here, so L is constant and the maps are comparable
elementwise; Stage 0 asserts that rather than assuming it.

Run from the repo root (so `assets/onnx` resolves):
    python3 py/phase3_attention_ladder.py
    python3 py/phase3_attention_ladder.py --presetspan --eps-ladder
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention import DEFAULT_ATTENTION_PATH, analyze, export_attention_graph  # noqa: E402
from helper import Style, load_text_to_speech, load_voice_style  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(REPO, "assets", "onnx")
VOICE_DIR = os.path.join(REPO, "assets", "voice_styles")
CORPUS = os.path.join(REPO, "py", "results", "phase2b_subspace")
OUT_DIR = os.path.join(REPO, "py", "results", "phase3_attention_ladder")

PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# Corpus-matching inference settings. Both are load-bearing: speed=1.05 is what
# the corpus was generated at (NOT this fork's 1.0 default), and style_dp fixed
# at M1's is what keeps L constant across every render.
SPEED = 1.05
TOTAL_STEP = 8
LANG = "en"

PRIMARY_TEXT_IDX = 0
LONG_TEXT_IDX = 5

N_NULL_SEEDS = 24
LADDER_SEED = 0          # the one vocoder seed every ladder render uses
N_LADDER = 80            # samples per K
SAMPLE_RNG_SEED = 20260924
N_SHUFFLE = 200          # random 24-row subsets per sample in the control
FRAME_MS = 1000.0 * 512 * 6 / 44100.0  # 69.66 ms, one latent frame


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def tv(a, b):
    """Total-variation distance between two row-attention profiles."""
    return float(0.5 * np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).sum())


def frame_tv(A, B):
    """Mean over frames of the per-frame TV between two aligned (L, 50) maps."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise AssertionError(f"frame_tv on mismatched shapes {A.shape} vs {B.shape}")
    return float((0.5 * np.abs(A - B).sum(axis=-1)).mean())


def dist_stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "p05": float(np.percentile(v, 5)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
    }


def overlap_report(same, diff):
    """How much the known-same and known-different distributions overlap.

    `auc` is the rank probability that a random known-different pair scores
    above a random known-same pair (0.5 = indistinguishable, 1.0 = perfect
    separation). `overlap_interval` is [min(diff), max(same)]; when that
    interval is empty the two distributions are cleanly separated, and the
    counts say how many pairs of each fall inside it when it is not.
    """
    same = np.asarray(same, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    u, p = mannwhitneyu(diff, same, alternative="greater")
    auc = float(u / (diff.size * same.size))
    lo, hi = float(diff.min()), float(same.max())
    separated = lo > hi
    n_same_in = int(((same >= lo) & (same <= hi)).sum()) if not separated else 0
    n_diff_in = int(((diff >= lo) & (diff <= hi)).sum()) if not separated else 0
    pooled_sd = np.sqrt(
        ((same.size - 1) * same.var(ddof=1) + (diff.size - 1) * diff.var(ddof=1))
        / (same.size + diff.size - 2)
    )
    return {
        "same": dist_stats(same),
        "different": dist_stats(diff),
        "auc_different_above_same": auc,
        "mannwhitney_p": float(p),
        "cohens_d": float((diff.mean() - same.mean()) / pooled_sd) if pooled_sd > 0 else None,
        "separated": bool(separated),
        "gap": float(lo - hi),
        "overlap_interval": [lo, hi] if not separated else None,
        "n_same_in_overlap": n_same_in,
        "n_diff_in_overlap": n_diff_in,
        "frac_same_in_overlap": n_same_in / same.size if not separated else 0.0,
        "frac_diff_in_overlap": n_diff_in / diff.size if not separated else 0.0,
        "ratio_of_means": float(diff.mean() / same.mean()) if same.mean() > 0 else None,
    }


def _f(v):
    """None -> NaN, so per-sample rho columns stay float arrays (to_py turns
    non-finite floats back into JSON null)."""
    return float("nan") if v is None else float(v)


def spearman_with_p(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": None, "p": None, "n": int(x.size)}
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(x.size)}


def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    return obj


def write_json(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(to_py(payload), f, indent=1)
    print(f"[write] {path}")
    return path


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

class Renderer:
    """One instrumented session, reused for every render in the run."""

    def __init__(self, attention_path):
        self.tts = load_text_to_speech(ONNX_DIR)
        self.attention_path = attention_path
        self.m1 = load_voice_style([os.path.join(VOICE_DIR, "M1.json")])
        self.n_renders = 0
        self.t_render = 0.0

    def render(self, ttl_50x256, text, seed):
        """Render one (style_ttl, text, seed) with style_dp pinned to M1's."""
        ttl = np.ascontiguousarray(
            np.asarray(ttl_50x256, dtype=np.float32).reshape(1, 50, 256)
        )
        style = Style(ttl, self.m1.dp)
        t0 = time.time()
        wav, alignment, style_attn = analyze(
            self.tts,
            text,
            LANG,
            style,
            total_step=TOTAL_STEP,
            speed=SPEED,
            seed=seed,
            attention_path=self.attention_path,
        )
        self.t_render += time.time() - t0
        self.n_renders += 1
        M = np.asarray(style_attn.matrix, dtype=np.float64)
        return {
            "M": M,                                # (L, 50)
            "w": M.mean(axis=0),                   # (50,)
            "L": int(M.shape[0]),
            "duration": float(alignment.duration),
            "node": alignment.node,
            "head": int(alignment.head),
            "stream": int(alignment.stream),
            "spearman": float(alignment.spearman),
            "word_spans": alignment.word_spans(),
            "effective_rows": np.asarray(style_attn.effective_rows(), dtype=np.float64),
            "frame_divergence": np.asarray(style_attn.frame_divergence(), dtype=np.float64),
            "wav_rms": float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float64) ** 2))),
        }


def assert_constant_L(renders, label):
    Ls = sorted({r["L"] for r in renders})
    if len(Ls) != 1:
        raise AssertionError(
            f"L is NOT constant across the '{label}' comparison set: saw {Ls}. "
            "Elementwise (L,50) comparison is invalid; stopping."
        )
    return Ls[0]


def word_shift_ms(base_spans, other_spans):
    """Absolute word-boundary shifts in ms, base vs other.

    Returns None if the word label sequences differ (nothing to align).
    """
    if [s[0] for s in base_spans] != [s[0] for s in other_spans]:
        return None
    shifts = []
    for (_, b0, b1), (_, o0, o1) in zip(base_spans, other_spans):
        shifts.append(abs(o0 - b0) * 1000.0)
        shifts.append(abs(o1 - b1) * 1000.0)
    return np.asarray(shifts, dtype=np.float64)


# --------------------------------------------------------------------------
# Stage 0: sanity
# --------------------------------------------------------------------------

def stage0_sanity(rend, base_ttl, text):
    out = {}

    m1_ttl = rend.m1.ttl[0]
    max_dev = float(np.abs(np.asarray(base_ttl, dtype=np.float64) - m1_ttl).max())
    out["base_ttl_vs_M1_max_abs_dev"] = max_dev
    out["base_ttl_is_M1"] = bool(max_dev == 0.0)
    if max_dev > 1e-6:
        out["WARNING"] = (
            "corpus base_ttl is NOT M1's style_ttl -- every 'distance from base' "
            "number below is relative to something else."
        )

    a = rend.render(base_ttl, text, LADDER_SEED)
    b = rend.render(base_ttl, text, LADDER_SEED)
    out["repeat_same_seed_row_profile_bit_identical"] = bool(np.array_equal(a["w"], b["w"]))
    out["repeat_same_seed_matrix_bit_identical"] = bool(np.array_equal(a["M"], b["M"]))
    out["repeat_same_seed_tv"] = tv(a["w"], b["w"])
    out["repeat_same_seed_frame_tv"] = frame_tv(a["M"], b["M"])
    out["deterministic_under_seed"] = bool(
        out["repeat_same_seed_matrix_bit_identical"]
    )
    out["base_render"] = {
        "L": a["L"],
        "duration_s": a["duration"],
        "node": a["node"],
        "head": a["head"],
        "stream": a["stream"],
        "spearman": a["spearman"],
        "frame_ms": FRAME_MS,
        "row_profile": a["w"].tolist(),
        "effective_rows_mean": float(a["effective_rows"].mean()),
        "frame_divergence_mean": float(a["frame_divergence"].mean()),
    }
    return out, a


# --------------------------------------------------------------------------
# Stage 1: calibration
# --------------------------------------------------------------------------

def stage1_calibration(rend, base_ttl, text, base_render, active_rows):
    print(f"[stage1] null: base_ttl at {N_NULL_SEEDS} seeds")
    null = []
    for s in range(N_NULL_SEEDS):
        r = rend.render(base_ttl, text, s) if s != LADDER_SEED else base_render
        null.append(r)
        print(f"  null seed {s} L={r['L']}", flush=True)
    L_null = assert_constant_L(null, "null / same-style seeds")

    print("[stage1] known-different: 10 presets, M1's style_dp, seeds 0 and 1")
    presets = {}
    for name in PRESETS:
        st = load_voice_style([os.path.join(VOICE_DIR, f"{name}.json")])
        presets[name] = {}
        for s in (0, 1):
            presets[name][s] = rend.render(st.ttl[0], text, s)
        print(
            f"  preset {name} L={presets[name][0]['L']} "
            f"dur={presets[name][0]['duration']:.3f}",
            flush=True,
        )
    flat = [presets[n][s] for n in PRESETS for s in (0, 1)]
    L_pre = assert_constant_L(flat + null, "presets + null (style_dp pinned)")

    # same-style noise floor: all pairs within the null set
    same_tv = [tv(null[i]["w"], null[j]["w"]) for i, j in combinations(range(len(null)), 2)]
    same_ftv = [frame_tv(null[i]["M"], null[j]["M"]) for i, j in combinations(range(len(null)), 2)]

    # a second same-style estimate from the presets: each preset at seed 0 vs 1
    same_tv_presets = [tv(presets[n][0]["w"], presets[n][1]["w"]) for n in PRESETS]
    same_ftv_presets = [frame_tv(presets[n][0]["M"], presets[n][1]["M"]) for n in PRESETS]

    # known-different, same seed (the brief's definition: isolates style_ttl)
    diff_tv, diff_ftv, diff_labels = [], [], []
    for a, b in combinations(PRESETS, 2):
        diff_tv.append(tv(presets[a][0]["w"], presets[b][0]["w"]))
        diff_ftv.append(frame_tv(presets[a][0]["M"], presets[b][0]["M"]))
        diff_labels.append(f"{a}|{b}")
    # known-different, cross seed (adds seed noise on top of the style difference
    # -- the honest worst case, since the same-seed version has none)
    diff_tv_xs, diff_ftv_xs = [], []
    for a, b in combinations(PRESETS, 2):
        diff_tv_xs.append(tv(presets[a][0]["w"], presets[b][1]["w"]))
        diff_ftv_xs.append(frame_tv(presets[a][0]["M"], presets[b][1]["M"]))

    # Calibration for the two ladder readouts that are NOT distances on the row
    # profile: word-boundary timing (Q5) and effective row count (Q6). Both
    # need their own noise floor for the same reason the TV distance does.
    base0 = null[0]
    null_word_max, null_word_mean = [], []
    for r in null[1:]:
        sh = word_shift_ms(base0["word_spans"], r["word_spans"])
        if sh is not None:
            null_word_max.append(float(sh.max()))
            null_word_mean.append(float(sh.mean()))
    diff_word_max, diff_word_mean = [], []
    for a, b in combinations(PRESETS, 2):
        sh = word_shift_ms(presets[a][0]["word_spans"], presets[b][0]["word_spans"])
        if sh is not None:
            diff_word_max.append(float(sh.max()))
            diff_word_mean.append(float(sh.mean()))
    act = np.asarray(active_rows, dtype=int)
    null_active_mass = [float(r["w"][act].sum()) for r in null]
    preset_active_mass = {n: float(presets[n][0]["w"][act].sum()) for n in PRESETS}
    null_eff = [float(r["effective_rows"].mean()) for r in null]
    diff_eff = [float(presets[n][0]["effective_rows"].mean()) for n in PRESETS]

    cal = {
        "word_boundary_noise_floor": {
            "frame_ms": FRAME_MS,
            "same_style_across_seeds": {
                "max_shift_ms": dist_stats(null_word_max) if null_word_max else None,
                "mean_shift_ms": dist_stats(null_word_mean) if null_word_mean else None,
                "n_pairs_with_matching_word_labels": len(null_word_max),
                "n_pairs": len(null) - 1,
            },
            "between_presets_same_seed": {
                "max_shift_ms": dist_stats(diff_word_max) if diff_word_max else None,
                "mean_shift_ms": dist_stats(diff_word_mean) if diff_word_mean else None,
                "n_pairs_with_matching_word_labels": len(diff_word_max),
                "n_pairs": 45,
            },
            "note": "the floor Q5's word-shift numbers have to clear: style_dp is "
                    "pinned in all of these, so any shift is text_emb's doing -- "
                    "but the seed moves boundaries too, and that is this floor.",
        },
        "active_row_mass_noise_floor": {
            "same_style_across_seeds": dist_stats(null_active_mass),
            "between_presets_same_seed": dist_stats(list(preset_active_mass.values())),
            "per_seed": null_active_mass,
            "per_preset": preset_active_mass,
            "note": "Q3 compares a sample's mass on the 24 perturbed rows with ONE "
                    "base render's. This is how much that base number moves when "
                    "nothing but the vocoder seed changes -- the floor any "
                    "'perturbation shifts mass' claim has to clear.",
        },
        "effective_rows_noise_floor": {
            "same_style_across_seeds": dist_stats(null_eff),
            "between_presets_same_seed": dist_stats(diff_eff),
        },
        "design": {
            "text": text,
            "speed": SPEED,
            "total_step": TOTAL_STEP,
            "style_dp": "M1's, substituted for every render including the presets",
            "null": f"base_ttl (=M1 style_ttl) at seeds 0..{N_NULL_SEEDS - 1}, all {len(same_tv)} pairs",
            "known_different": "10 shipped presets' style_ttl, all 45 distinct-preset pairs",
            "L_null": L_null,
            "L_all": L_pre,
        },
        "utterance_tv": overlap_report(same_tv, diff_tv),
        "frame_tv": overlap_report(same_ftv, diff_ftv),
        "utterance_tv_cross_seed_different": overlap_report(same_tv, diff_tv_xs),
        "frame_tv_cross_seed_different": overlap_report(same_ftv, diff_ftv_xs),
        "same_style_within_preset_pairs": {
            "utterance_tv": dist_stats(same_tv_presets),
            "frame_tv": dist_stats(same_ftv_presets),
            "note": "each preset at seed 0 vs seed 1 -- a same-style floor measured "
                    "on 10 styles other than the base, as a check that the null "
                    "measured on M1 alone is not special to M1",
        },
        "raw": {
            "null_pair_tv": same_tv,
            "null_pair_frame_tv": same_ftv,
            "different_pair_labels": diff_labels,
            "different_pair_tv": diff_tv,
            "different_pair_frame_tv": diff_ftv,
            "different_pair_tv_cross_seed": diff_tv_xs,
            "different_pair_frame_tv_cross_seed": diff_ftv_xs,
            "preset_row_profiles": {n: presets[n][0]["w"].tolist() for n in PRESETS},
            "preset_durations": {n: presets[n][0]["duration"] for n in PRESETS},
            "preset_alignment": {
                n: {
                    "node": presets[n][0]["node"],
                    "head": presets[n][0]["head"],
                    "stream": presets[n][0]["stream"],
                    "spearman": presets[n][0]["spearman"],
                }
                for n in PRESETS
            },
        },
    }

    # The verdict that gates stage 2. Both metrics must separate, and the
    # ladder's perturbations (eps 0.2, Frobenius ~0.98) are far smaller than a
    # whole-preset difference (Frobenius 1.8-3.5), so a merely-significant
    # separation between presets is not enough -- the noise floor has to be
    # small compared with the between-preset scale, which `headroom` records.
    u, f = cal["utterance_tv"], cal["frame_tv"]
    cal["verdict"] = {
        "utterance_tv_separates": bool(u["separated"]),
        "frame_tv_separates": bool(f["separated"]),
        "utterance_tv_auc": u["auc_different_above_same"],
        "frame_tv_auc": f["auc_different_above_same"],
        "headroom_utterance_tv": u["ratio_of_means"],
        "headroom_frame_tv": f["ratio_of_means"],
        "separates": bool(u["separated"] and f["separated"]),
    }
    return cal


# --------------------------------------------------------------------------
# Stage 2: the ladder
# --------------------------------------------------------------------------

def per_row_perturbation(ttl_sample, base_ttl):
    """(50,) per-row L2 norm of the perturbation. Zero on untouched rows."""
    d = np.asarray(ttl_sample, dtype=np.float64) - np.asarray(base_ttl, dtype=np.float64)
    return np.linalg.norm(d, axis=-1)


def run_one_ladder(rend, base_ttl, base_render, corpus, keys, text, n_per_k,
                   rng_seed, label):
    """Render n_per_k samples from each key in `keys` and collect every metric."""
    rng = np.random.default_rng(rng_seed)
    active = np.asarray(corpus["active_rows"], dtype=int)
    inactive = np.asarray(sorted(set(range(50)) - set(active.tolist())), dtype=int)
    base_w = base_render["w"]
    base_M = base_render["M"]
    base_spans = base_render["word_spans"]

    groups = {}
    for key in keys:
        ttl_all = corpus[key]
        n_avail = ttl_all.shape[0]
        take = min(n_per_k, n_avail)
        idx = np.sort(rng.choice(n_avail, size=take, replace=False))
        rows = []
        renders = []
        print(f"[{label}] {key}: {take} of {n_avail}", flush=True)
        for j, i in enumerate(idx):
            ttl = ttl_all[int(i)]
            r = rend.render(ttl, text, LADDER_SEED)
            renders.append(r)

            dw = r["w"] - base_w
            pr = per_row_perturbation(ttl, base_ttl)
            frob = float(np.linalg.norm(np.asarray(ttl, dtype=np.float64)
                                        - np.asarray(base_ttl, dtype=np.float64)))

            # Q3: mass on the perturbed rows vs the untouched ones.
            d_active = float(dw[active].sum())
            shuf = np.empty(N_SHUFFLE)
            for m in range(N_SHUFFLE):
                pick = rng.choice(50, size=active.size, replace=False)
                shuf[m] = dw[pick].sum()
            # per-row targeting: does the change track which rows moved, and by
            # how much? Both within the 24 perturbed rows (where pr varies) and
            # across all 50 (where the active/inactive split dominates).
            rho_active = spearman_with_p(pr[active], dw[active])
            rho_active_abs = spearman_with_p(pr[active], np.abs(dw[active]))
            perm = rng.permutation(active.size)
            rho_active_perm = spearman_with_p(pr[active][perm], dw[active])
            perm50 = rng.permutation(50)
            rho_all = spearman_with_p(pr, np.abs(dw))
            rho_all_perm = spearman_with_p(pr[perm50], np.abs(dw))

            shifts = word_shift_ms(base_spans, r["word_spans"])
            rows.append({
                "corpus_index": int(i),
                "dw": dw.tolist(),
                "per_row_perturbation": pr.tolist(),
                "frobenius": frob,
                "tv": tv(r["w"], base_w),
                "frame_tv": frame_tv(r["M"], base_M),
                "active_mass": float(r["w"][active].sum()),
                "inactive_mass": float(r["w"][inactive].sum()),
                "d_active_mass": d_active,
                "shuffled_d_mass_mean": float(shuf.mean()),
                "shuffled_d_mass_sd": float(shuf.std(ddof=1)),
                "shuffled_abs_ge_real": float((np.abs(shuf) >= abs(d_active)).mean()),
                "rho_active_signed": _f(rho_active["rho"]),
                "rho_active_signed_perm": _f(rho_active_perm["rho"]),
                "rho_active_abs": _f(rho_active_abs["rho"]),
                "rho_all50_abs": _f(rho_all["rho"]),
                "rho_all50_abs_perm": _f(rho_all_perm["rho"]),
                "effective_rows_mean": float(r["effective_rows"].mean()),
                "effective_rows_min": float(r["effective_rows"].min()),
                "effective_rows_max": float(r["effective_rows"].max()),
                "frame_divergence_mean": float(r["frame_divergence"].mean()),
                "profile_entropy_nats": float(
                    -(r["w"][r["w"] > 0] * np.log(r["w"][r["w"] > 0])).sum()
                ),
                "node": r["node"], "head": r["head"], "stream": r["stream"],
                "spearman": r["spearman"], "L": r["L"], "duration": r["duration"],
                "word_span_labels_match_base": shifts is not None,
                "word_shift_ms_max": float(shifts.max()) if shifts is not None else None,
                "word_shift_ms_mean": float(shifts.mean()) if shifts is not None else None,
                "word_shift_ms_nonzero_frac": (
                    float((shifts > 0).mean()) if shifts is not None else None
                ),
            })
            if (j + 1) % 20 == 0:
                print(f"    {j + 1}/{take}", flush=True)
        assert_constant_L(renders + [base_render], f"{label}/{key}")
        groups[key] = rows
    return groups


def stage2b_seed_robustness(rend, base_ttl, corpus, groups, text, n_per_k, alt_seed):
    """Re-render a subset of the ladder at a SECOND vocoder seed.

    Every stage-2 number is taken at one seed, so it is noise-free *within* the
    comparison but specific to that seed. This asks the out-of-sample question:
    does the per-sample distance ranking survive a seed change, or is the
    ranking a property of one noise draw? If the cross-seed Spearman is low,
    the ladder ordering is not a property of the styles.
    """
    base_alt = rend.render(base_ttl, text, alt_seed)
    out = {"alt_seed": alt_seed, "n_per_K": n_per_k, "by_K": {}}
    renders = [base_alt]
    for key, rows in groups.items():
        sub = rows[:n_per_k]
        tv0 = [r["tv"] for r in sub]
        ftv0 = [r["frame_tv"] for r in sub]
        frob = [r["frobenius"] for r in sub]
        tv1, ftv1, dact1 = [], [], []
        active = np.asarray(corpus["active_rows"], dtype=int)
        print(f"[stage2b] {key}: {len(sub)} at seed {alt_seed}", flush=True)
        for r in sub:
            rr = rend.render(corpus[key][r["corpus_index"]], text, alt_seed)
            renders.append(rr)
            tv1.append(tv(rr["w"], base_alt["w"]))
            ftv1.append(frame_tv(rr["M"], base_alt["M"]))
            dact1.append(float((rr["w"] - base_alt["w"])[active].sum()))
        out["by_K"][key] = {
            "frac_samples_d_active_negative_seed0": float(
                np.mean([1.0 if r["d_active_mass"] < 0 else 0.0 for r in sub])
            ),
            "frac_samples_d_active_negative_alt": float(
                np.mean([1.0 if v < 0 else 0.0 for v in dact1])
            ),
            "sign_flips_with_seed": bool(
                (np.mean([r["d_active_mass"] for r in sub]) < 0) != (np.mean(dact1) < 0)
            ),
            "d_active_mass_alt_per_sample": [float(v) for v in dact1],
            "tv_seed0_mean": float(np.mean(tv0)),
            "tv_alt_mean": float(np.mean(tv1)),
            "frame_tv_seed0_mean": float(np.mean(ftv0)),
            "frame_tv_alt_mean": float(np.mean(ftv1)),
            "spearman_tv_across_seeds": spearman_with_p(tv0, tv1),
            "spearman_frame_tv_across_seeds": spearman_with_p(ftv0, ftv1),
            "spearman_frobenius_vs_tv_alt": spearman_with_p(frob, tv1),
            "spearman_frobenius_vs_frame_tv_alt": spearman_with_p(ftv0, ftv1),
            "d_active_mass_alt_mean": float(np.mean(dact1)),
            "d_active_mass_seed0_mean": float(np.mean([r["d_active_mass"] for r in sub])),
            "spearman_d_active_mass_across_seeds": spearman_with_p(
                [r["d_active_mass"] for r in sub], dact1
            ),
        }
    assert_constant_L(renders, "seed-robustness")
    return out


def _paired_p(a, b):
    """Wilcoxon signed-rank on a - b, ignoring non-finite pairs."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size < 5:
        return None
    return float(wilcoxon(d)[1])


def _sign_consistency_control(rows, active_rows):
    """The shape-artifact control for the one effect that does come out of the
    ladder: attention mass moving off the perturbed rows.

    Per sample, a random 24-row subset is not distinguishable from the real
    active set (the per-sample shuffle in `run_one_ladder` measures that). The
    claim that survives is about the *sign being the same in sample after
    sample*, and the matching control is therefore a *fixed* random 24-row
    subset scored across all samples, exactly as the real active set is. If a
    fixed random subset is just as sign-consistent, the consistency is a
    property of the 50-vector's shape, not of which rows were perturbed.
    """
    if active_rows is None:
        return None
    dw = np.asarray([r["dw"] for r in rows], dtype=np.float64)   # (n, 50)
    active = np.asarray(active_rows, dtype=int)
    real_frac_neg = float((dw[:, active].sum(axis=1) < 0).mean())
    rng = np.random.default_rng(4242)
    fracs = np.empty(N_SHUFFLE)
    means = np.empty(N_SHUFFLE)
    for m in range(N_SHUFFLE):
        pick = rng.choice(50, size=active.size, replace=False)
        col = dw[:, pick].sum(axis=1)
        fracs[m] = (col < 0).mean()
        means[m] = col.mean()
    real_mean = float(dw[:, active].sum(axis=1).mean())
    return {
        "real_frac_samples_negative": real_frac_neg,
        "real_mean_d_active_mass": real_mean,
        "fixed_random_subsets": {
            "n_subsets": N_SHUFFLE,
            "frac_negative_mean": float(fracs.mean()),
            "frac_negative_sd": float(fracs.std(ddof=1)),
            "frac_negative_min": float(fracs.min()),
            "frac_negative_max": float(fracs.max()),
            "mean_d_mass_mean": float(means.mean()),
            "mean_d_mass_sd": float(means.std(ddof=1)),
        },
        "p_two_sided_frac_negative": float(
            (np.abs(fracs - 0.5) >= abs(real_frac_neg - 0.5)).mean()
        ),
        "p_one_sided_mean_d_mass": float((means <= real_mean).mean()),
        "note": "p_* are the fraction of fixed random 24-row subsets that are at "
                "least as extreme as the real active set, scored the same way. "
                "A large value means the effect is a shape artifact.",
    }


def summarize_group(rows, base_render, active_size=24, active_rows=None):
    """Answer the brief's six numbered questions for one K group."""
    g = lambda k: np.asarray([r[k] for r in rows], dtype=np.float64)  # noqa: E731
    frob = g("frobenius")
    tvs = g("tv")
    ftvs = g("frame_tv")
    d_act = g("d_active_mass")

    shifts_max = np.asarray(
        [r["word_shift_ms_max"] for r in rows if r["word_shift_ms_max"] is not None],
        dtype=np.float64,
    )
    shifts_mean = np.asarray(
        [r["word_shift_ms_mean"] for r in rows if r["word_shift_ms_mean"] is not None],
        dtype=np.float64,
    )

    def sign_test(v):
        v = np.asarray(v, dtype=np.float64)
        v = v[v != 0]
        if v.size < 5:
            return {"n": int(v.size), "p": None, "frac_positive": None}
        stat, p = wilcoxon(v)
        return {
            "n": int(v.size),
            "wilcoxon_p": float(p),
            "frac_positive": float((v > 0).mean()),
        }

    real_rho = np.asarray([r["rho_active_signed"] for r in rows], dtype=np.float64)
    perm_rho = np.asarray([r["rho_active_signed_perm"] for r in rows], dtype=np.float64)
    real_rho_all = np.asarray([r["rho_all50_abs"] for r in rows], dtype=np.float64)
    perm_rho_all = np.asarray([r["rho_all50_abs_perm"] for r in rows], dtype=np.float64)

    out = {
        "n": len(rows),
        "q1_tv_from_base": dist_stats(tvs),
        "q2_frame_tv_from_base": dist_stats(ftvs),
        "q3_active_rows": {
            "active_row_count": active_size,
            "base_active_mass": None,  # filled by caller
            "sample_active_mass": dist_stats(g("active_mass")),
            "d_active_mass": dist_stats(d_act),
            "d_active_mass_sign_test": sign_test(d_act),
            "sign_consistency_control": _sign_consistency_control(rows, active_rows),
            "mean_dw_per_row": np.mean([r["dw"] for r in rows], axis=0).tolist(),
            "shuffled_control": {
                "mean_frac_shuffled_abs_ge_real": float(g("shuffled_abs_ge_real").mean()),
                "frac_samples_real_beats_95pct_of_shuffles": float(
                    (g("shuffled_abs_ge_real") < 0.05).mean()
                ),
                "shuffled_d_mass_sd_mean": float(g("shuffled_d_mass_sd").mean()),
                "note": "shuffled = 200 random 24-row subsets per sample; "
                        "frac_shuffled_abs_ge_real is the fraction of those whose "
                        "|mass change| is at least the real active set's. 0.5 means "
                        "the real active set is unremarkable among random subsets.",
            },
            "per_row_targeting": {
                "rho_pr_vs_dw_within_active_mean": float(np.nanmean(real_rho)),
                "rho_pr_vs_dw_within_active_sd": float(np.nanstd(real_rho, ddof=1)),
                "rho_shuffled_labels_mean": float(np.nanmean(perm_rho)),
                "rho_within_active_vs_shuffled_wilcoxon_p": _paired_p(real_rho, perm_rho),
                "rho_pr_vs_absdw_all50_mean": float(np.nanmean(real_rho_all)),
                "rho_all50_shuffled_mean": float(np.nanmean(perm_rho_all)),
                "rho_all50_vs_shuffled_wilcoxon_p": _paired_p(real_rho_all, perm_rho_all),
            },
        },
        "q4_magnitude": {
            "frobenius": dist_stats(frob),
            "spearman_frobenius_vs_tv": spearman_with_p(frob, tvs),
            "spearman_frobenius_vs_frame_tv": spearman_with_p(frob, ftvs),
            "spearman_frobenius_vs_d_active_mass": spearman_with_p(frob, d_act),
        },
        "q5_alignment": {
            "winning_node_head_stream_counts": {},
            "spearman_rho": dist_stats(g("spearman")),
            "word_span_labels_match_base_frac": float(
                np.mean([1.0 if r["word_span_labels_match_base"] else 0.0 for r in rows])
            ),
            "word_shift_ms_max": dist_stats(shifts_max) if shifts_max.size else None,
            "word_shift_ms_mean": dist_stats(shifts_mean) if shifts_mean.size else None,
            "frame_ms": FRAME_MS,
            "n_samples_with_any_nonzero_shift": int(
                sum(1 for r in rows if r["word_shift_ms_max"] not in (None, 0.0))
            ),
        },
        "q6_effective_rows": {
            "effective_rows_mean": dist_stats(g("effective_rows_mean")),
            "profile_entropy_nats": dist_stats(g("profile_entropy_nats")),
            "frame_divergence_mean": dist_stats(g("frame_divergence_mean")),
            "spearman_frobenius_vs_effective_rows": spearman_with_p(
                frob, g("effective_rows_mean")
            ),
        },
    }
    counts = {}
    for r in rows:
        k = f"{r['node']}|h{r['head']}|s{r['stream']}"
        counts[k] = counts.get(k, 0) + 1
    out["q5_alignment"]["winning_node_head_stream_counts"] = counts
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    global N_NULL_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ladder", type=int, default=N_LADDER)
    ap.add_argument("--n-null", type=int, default=N_NULL_SEEDS)
    ap.add_argument("--n-generality", type=int, default=10,
                    help="samples per K on the longer text")
    ap.add_argument("--force-ladder", action="store_true",
                    help="run the ladder even if calibration does not separate; "
                         "recorded in the output")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--presetspan", action="store_true")
    ap.add_argument("--eps-ladder", action="store_true")
    ap.add_argument("--n-secondary", type=int, default=30)
    ap.add_argument("--n-seed-robust", type=int, default=20,
                    help="samples per K re-rendered at a second vocoder seed")
    args = ap.parse_args()

    N_NULL_SEEDS = args.n_null

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    manifest = json.load(open(os.path.join(CORPUS, "manifest.json")))
    texts = manifest["texts"]
    primary, longer = texts[PRIMARY_TEXT_IDX], texts[LONG_TEXT_IDX]
    z = np.load(os.path.join(CORPUS, "subspace.npz"))
    corpus = {k: z[k] for k in ("base_ttl", "active_rows", "ttl_K4", "ttl_K16", "ttl_K64")}
    base_ttl = corpus["base_ttl"]

    attention_path = os.path.join(ONNX_DIR, "vector_estimator.attn.onnx")
    export_attention_graph(os.path.join(ONNX_DIR, "vector_estimator.onnx"), attention_path)
    print(f"[setup] instrumented graph: {attention_path}")
    rend = Renderer(attention_path)

    print("[stage0] sanity")
    sanity, base_render = stage0_sanity(rend, base_ttl, primary)
    print(json.dumps({k: v for k, v in sanity.items() if k != "base_render"}, indent=1))
    if not sanity["deterministic_under_seed"]:
        write_json("sanity.json", sanity)
        raise SystemExit(
            "STOP: repeated render at the same seed and style is not bit-identical. "
            "Everything downstream depends on that; nothing else was run."
        )

    cal = stage1_calibration(rend, base_ttl, primary, base_render,
                             corpus["active_rows"])
    write_json("calibration.json", {
        "corpus": CORPUS,
        "settings": {"speed": SPEED, "total_step": TOTAL_STEP, "lang": LANG,
                     "primary_text": primary, "ladder_seed": LADDER_SEED},
        "sanity": sanity,
        "calibration": cal,
    })
    print("[stage1] verdict:", json.dumps(to_py(cal["verdict"]), indent=1))

    separates = cal["verdict"]["separates"]
    if not separates and not args.force_ladder:
        write_json("ladder.json", {
            "status": "NOT RUN",
            "reason": "calibration did not separate known-same from "
                      "known-different; the distance cannot support ladder claims.",
            "calibration_verdict": cal["verdict"],
        })
        print("[stage2] SKIPPED -- calibration does not separate.")
        return

    if args.skip_ladder:
        print("[stage2] skipped by flag")
        return

    active = np.asarray(corpus["active_rows"], dtype=int)
    base_active_mass = float(base_render["w"][active].sum())

    groups = run_one_ladder(
        rend, base_ttl, base_render, corpus,
        ["ttl_K4", "ttl_K16", "ttl_K64"], primary, args.n_ladder,
        SAMPLE_RNG_SEED, "stage2",
    )
    summaries = {}
    for key, rows in groups.items():
        s = summarize_group(rows, base_render, active.size, active)
        s["q3_active_rows"]["base_active_mass"] = base_active_mass
        s["q3_active_rows"]["base_active_mass_uniform_expectation"] = active.size / 50.0
        summaries[key] = s

    pooled = [r for rows in groups.values() for r in rows]
    pooled_summary = summarize_group(pooled, base_render, active.size, active)
    pooled_summary["q3_active_rows"]["base_active_mass"] = base_active_mass

    # cross-K comparison for Q6 and Q1
    across_k = {
        "tv_mean_by_K": {k: float(np.mean([r["tv"] for r in v])) for k, v in groups.items()},
        "frame_tv_mean_by_K": {k: float(np.mean([r["frame_tv"] for r in v])) for k, v in groups.items()},
        "frobenius_mean_by_K": {k: float(np.mean([r["frobenius"] for r in v])) for k, v in groups.items()},
        "effective_rows_mean_by_K": {
            k: float(np.mean([r["effective_rows_mean"] for r in v])) for k, v in groups.items()
        },
        "base_effective_rows_mean": float(base_render["effective_rows"].mean()),
        "base_profile_entropy_nats": float(
            -(base_render["w"][base_render["w"] > 0]
              * np.log(base_render["w"][base_render["w"] > 0])).sum()
        ),
    }

    # calibration context: where do the ladder distances sit relative to the
    # two reference distributions? This is the only honest way to read them.
    cal_ctx = {}
    for metric, calkey in (("tv", "utterance_tv"), ("frame_tv", "frame_tv")):
        same = np.asarray(cal["raw"][f"null_pair_{'tv' if metric == 'tv' else 'frame_tv'}"])
        diff = np.asarray(cal["raw"][f"different_pair_{'tv' if metric == 'tv' else 'frame_tv'}"])
        vals = np.asarray([r[metric] for r in pooled], dtype=np.float64)
        cal_ctx[metric] = {
            "ladder_mean": float(vals.mean()),
            "same_style_floor_mean": float(same.mean()),
            "same_style_floor_max": float(same.max()),
            "between_preset_mean": float(diff.mean()),
            "frac_ladder_above_same_style_max": float((vals > same.max()).mean()),
            "frac_ladder_below_between_preset_min": float((vals < diff.min()).mean()),
            "ladder_over_floor": float(vals.mean() / same.mean()),
            "ladder_over_between_preset": float(vals.mean() / diff.mean()),
            "calibration_auc": cal[calkey]["auc_different_above_same"],
        }

    write_json("ladder.json", {
        "status": "RUN" + (" (FORCED past calibration)" if not separates else ""),
        "settings": {"speed": SPEED, "total_step": TOTAL_STEP, "lang": LANG,
                     "text": primary, "vocoder_seed": LADDER_SEED,
                     "style_dp": "M1's for every render",
                     "n_per_K": args.n_ladder, "sample_rng_seed": SAMPLE_RNG_SEED,
                     "n_shuffle_control": N_SHUFFLE},
        "calibration_verdict": cal["verdict"],
        "calibration_context": cal_ctx,
        "by_K": summaries,
        "pooled": pooled_summary,
        "across_K": across_k,
        "active_rows": active.tolist(),
        "per_sample": groups,
    })

    # ---------------- stage 2b: does the ordering survive a seed change? ----
    if args.n_seed_robust > 0:
        rob = stage2b_seed_robustness(
            rend, base_ttl, corpus, groups, primary, args.n_seed_robust,
            alt_seed=LADDER_SEED + 1,
        )
        write_json("seed_robustness.json", rob)

    # ---------------- stage 3: generality on the longer text ----------------
    if args.n_generality > 0:
        print("[stage3] generality on the longer text")
        long_base_a = rend.render(base_ttl, longer, LADDER_SEED)
        long_base_b = rend.render(base_ttl, longer, LADDER_SEED)
        det = bool(np.array_equal(long_base_a["M"], long_base_b["M"]))
        long_null = [long_base_a] + [
            rend.render(base_ttl, longer, s) for s in range(1, 9)
        ]
        assert_constant_L(long_null, "generality null")
        long_same_tv = [tv(long_null[i]["w"], long_null[j]["w"])
                        for i, j in combinations(range(len(long_null)), 2)]
        long_same_ftv = [frame_tv(long_null[i]["M"], long_null[j]["M"])
                         for i, j in combinations(range(len(long_null)), 2)]
        g_groups = run_one_ladder(
            rend, base_ttl, long_base_a, corpus,
            ["ttl_K4", "ttl_K16", "ttl_K64"], longer, args.n_generality,
            SAMPLE_RNG_SEED + 1, "stage3",
        )
        g_summaries = {}
        for key, rows in g_groups.items():
            s = summarize_group(rows, long_base_a, active.size, active)
            s["q3_active_rows"]["base_active_mass"] = float(long_base_a["w"][active].sum())
            g_summaries[key] = s
        g_pooled = [r for rows in g_groups.values() for r in rows]
        gp = summarize_group(g_pooled, long_base_a, active.size, active)
        gp["q3_active_rows"]["base_active_mass"] = float(long_base_a["w"][active].sum())
        write_json("generality_long_text.json", {
            "text": longer,
            "deterministic_under_seed": det,
            "L": long_base_a["L"],
            "duration_s": long_base_a["duration"],
            "same_style_floor": {
                "seeds": list(range(9)),
                "utterance_tv": dist_stats(long_same_tv),
                "frame_tv": dist_stats(long_same_ftv),
            },
            "by_K": g_summaries,
            "pooled": gp,
            "per_sample": g_groups,
            "n_per_K": args.n_generality,
        })

    # ---------------- stage 4: secondary corpora ----------------
    secondary = {}
    base_alt_ps = None
    if args.presetspan:
        for cond in ("preset_span", "random_control", "row_matched_random"):
            path = os.path.join(REPO, "py", "results", "phase2b_presetspan", cond, "subspace.npz")
            zz = np.load(path)
            c = {k: zz[k] for k in zz.files}
            if float(np.abs(c["base_ttl"] - base_ttl).max()) != 0.0:
                secondary.setdefault("warnings", []).append(
                    f"{cond}: base_ttl differs from the primary corpus's"
                )
            print(f"[stage4] presetspan/{cond}")
            gg = run_one_ladder(
                rend, base_ttl, base_render, c, ["ttl_K9"], primary,
                args.n_secondary, SAMPLE_RNG_SEED + 2, f"stage4/{cond}",
            )
            s = summarize_group(gg["ttl_K9"], base_render, active.size, active)
            s["q3_active_rows"]["base_active_mass"] = base_active_mass
            secondary[cond] = {"summary": s, "per_sample": gg["ttl_K9"]}

            # The preset-span contrast is the one ladder result that clears its
            # controls at seed 0. Repeat it at a second vocoder seed: if the
            # contrast is a property of the subspace it must survive, and if it
            # is a property of one noise draw it will not.
            if base_alt_ps is None:
                base_alt_ps = rend.render(base_ttl, primary, LADDER_SEED + 1)
            alt_tv, alt_ftv = [], []
            for r in gg["ttl_K9"]:
                rr2 = rend.render(c["ttl_K9"][r["corpus_index"]], primary,
                                  LADDER_SEED + 1)
                alt_tv.append(tv(rr2["w"], base_alt_ps["w"]))
                alt_ftv.append(frame_tv(rr2["M"], base_alt_ps["M"]))
            secondary[cond]["alt_seed"] = {
                "seed": LADDER_SEED + 1,
                "tv": alt_tv,
                "frame_tv": alt_ftv,
                "tv_mean": float(np.mean(alt_tv)),
                "frame_tv_mean": float(np.mean(alt_ftv)),
                "spearman_tv_across_seeds": spearman_with_p(
                    [r["tv"] for r in gg["ttl_K9"]], alt_tv
                ),
            }
    if args.eps_ladder:
        for name, sub in (("eps0.05_K4", "phase2b_subspace_matched_k4"),
                          ("eps0.05_K16", "phase2b_subspace_matched_k16")):
            path = os.path.join(REPO, "py", "results", sub, "subspace.npz")
            if not os.path.exists(path):
                continue
            zz = np.load(path)
            c = {k: zz[k] for k in zz.files}
            key = [k for k in zz.files if k.startswith("ttl_K")][0]
            print(f"[stage4] {name} ({sub})")
            gg = run_one_ladder(
                rend, base_ttl, base_render, c, [key], primary,
                args.n_secondary, SAMPLE_RNG_SEED + 3, f"stage4/{name}",
            )
            s = summarize_group(gg[key], base_render, active.size, active)
            s["q3_active_rows"]["base_active_mass"] = base_active_mass
            secondary[name] = {"summary": s, "per_sample": gg[key],
                               "eps": json.load(open(os.path.join(
                                   REPO, "py", "results", sub, "manifest.json")))["eps"]}
    # Q4 done properly: within one corpus the realized Frobenius norm is fixed
    # by construction (eps is a constant, so every sample sits on the same
    # sphere), which makes a within-corpus magnitude correlation a correlation
    # against ~0 variance. The only real magnitude contrast available is
    # BETWEEN eps levels, so pool eps 0.05 against eps 0.20 here.
    if args.eps_ladder:
        cross = {}
        for name, big in (("K4", "ttl_K4"), ("K16", "ttl_K16")):
            small = secondary.get(f"eps0.05_{name}")
            if small is None or big not in groups:
                continue
            a = small["per_sample"]
            b = groups[big]
            frob = [r["frobenius"] for r in a] + [r["frobenius"] for r in b]
            tvv = [r["tv"] for r in a] + [r["tv"] for r in b]
            ftvv = [r["frame_tv"] for r in a] + [r["frame_tv"] for r in b]
            dact = [r["d_active_mass"] for r in a] + [r["d_active_mass"] for r in b]
            cross[name] = {
                "n_eps0.05": len(a), "n_eps0.20": len(b),
                "frobenius_mean_eps0.05": float(np.mean([r["frobenius"] for r in a])),
                "frobenius_mean_eps0.20": float(np.mean([r["frobenius"] for r in b])),
                "tv_mean_eps0.05": float(np.mean([r["tv"] for r in a])),
                "tv_mean_eps0.20": float(np.mean([r["tv"] for r in b])),
                "frame_tv_mean_eps0.05": float(np.mean([r["frame_tv"] for r in a])),
                "frame_tv_mean_eps0.20": float(np.mean([r["frame_tv"] for r in b])),
                "d_active_mass_mean_eps0.05": float(np.mean([r["d_active_mass"] for r in a])),
                "d_active_mass_mean_eps0.20": float(np.mean([r["d_active_mass"] for r in b])),
                "spearman_frobenius_vs_tv_pooled": spearman_with_p(frob, tvv),
                "spearman_frobenius_vs_frame_tv_pooled": spearman_with_p(frob, ftvv),
                "spearman_frobenius_vs_d_active_mass_pooled": spearman_with_p(frob, dact),
                "note": "two magnitude levels only -- a monotonic step, not a ladder",
            }
        if cross:
            secondary["cross_eps_magnitude"] = cross

    if args.presetspan:
        three = {}
        for cond in ("preset_span", "random_control", "row_matched_random"):
            if cond in secondary:
                rr = secondary[cond]["per_sample"]
                three[cond] = {
                    "tv_mean": float(np.mean([r["tv"] for r in rr])),
                    "frame_tv_mean": float(np.mean([r["frame_tv"] for r in rr])),
                    "d_active_mass_mean": float(np.mean([r["d_active_mass"] for r in rr])),
                    "frobenius_mean": float(np.mean([r["frobenius"] for r in rr])),
                }
        if "preset_span" in secondary and "random_control" in secondary:
            a = [r["tv"] for r in secondary["preset_span"]["per_sample"]]
            b = [r["tv"] for r in secondary["random_control"]["per_sample"]]
            u, pp = mannwhitneyu(a, b, alternative="two-sided")
            three["preset_span_vs_random_control_tv"] = {
                "auc": float(u / (len(a) * len(b))), "p": float(pp)
            }
            a2 = [r["frame_tv"] for r in secondary["preset_span"]["per_sample"]]
            b2 = [r["frame_tv"] for r in secondary["random_control"]["per_sample"]]
            u2, pp2 = mannwhitneyu(a2, b2, alternative="two-sided")
            three["preset_span_vs_random_control_frame_tv"] = {
                "auc": float(u2 / (len(a2) * len(b2))), "p": float(pp2)
            }
        if ("preset_span" in secondary and "random_control" in secondary
                and "alt_seed" in secondary["preset_span"]):
            for metric in ("tv", "frame_tv"):
                a3 = secondary["preset_span"]["alt_seed"][metric]
                b3 = secondary["random_control"]["alt_seed"][metric]
                u3, p3 = mannwhitneyu(a3, b3, alternative="two-sided")
                three[f"preset_span_vs_random_control_{metric}_ALT_SEED"] = {
                    "auc": float(u3 / (len(a3) * len(b3))), "p": float(p3),
                    "preset_span_mean": float(np.mean(a3)),
                    "random_control_mean": float(np.mean(b3)),
                }
        if three:
            secondary["three_way"] = three

    if secondary:
        secondary["_settings"] = {
            "text": primary, "vocoder_seed": LADDER_SEED, "speed": SPEED,
            "n_per_condition": args.n_secondary,
            "same_style_floor_utterance_tv": cal["utterance_tv"]["same"],
            "same_style_floor_frame_tv": cal["frame_tv"]["same"],
        }
        write_json("secondary.json", secondary)

    print(f"[done] {rend.n_renders} renders, {rend.t_render:.1f}s in analyze(), "
          f"{time.time() - t_start:.1f}s wall")


if __name__ == "__main__":
    main()
