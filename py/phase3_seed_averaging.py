"""Phase 3 / task 18 follow-up: how much seed averaging does the style-row
attention readout need, and is that price realistic?

Task 18 (`py/phase3_attention_ladder.py`, `docs/ATTENTION_READOUT.md`) killed
the style-row attention profile as a per-style readout at one render per
style: the same style re-rendered at a second vocoder seed moves the profile
0.0328 +/- 0.0114 in total variation, a whole different shipped preset moves
it only 0.0594 +/- 0.0310, AUC 0.826. The obvious repair is to average the
profile over several seeds. It obviously helps. The question this script
answers is *how much averaging*, and whether that is affordable.

The profile is

    w = StyleAttention.matrix.mean(axis=0)      # (50,), sums to 1
    TV(a, b) = 0.5 * |a - b|.sum()

Design, and the two things it is careful about:

  * **Condition matching.** Every distance in a comparison averages the same
    number of renders on both sides, and both sides always use *disjoint
    seed index sets*. Disjointness is load-bearing rather than pedantic:
    `sample_noisy_latent` draws from `default_rng(seed)` and `style_dp` is
    pinned to M1's for every render here, so L is constant and two renders
    that share a seed share their `noisy_latent` *exactly*, whatever style
    they were given. A comparison in which one side shares seeds and the
    other does not is measuring the seed, which is how task 18's
    frame-resolved statistic came out backwards (AUC 0.355).

  * **Every distance gets its floor at the same N.** There is no number in
    the output that is not sitting next to its same-condition distribution
    computed with the same amount of averaging.

Parts:

  A  the separation curve for the 10 shipped presets, 64 seeds each, over
     N in {1, 2, 4, 8, 16, 32}: same-style (two disjoint N-seed averages of
     one preset) vs different-preset (N-seed averages of two presets, on
     disjoint seed sets). Fit the same-style curve against 1/sqrt(N) and
     estimate its asymptote -- the number that decides whether averaging can
     ever reach the ladder.

  B  the thing actually wanted: 24 eps-0.2 samples from `ttl_K64` at 32
     seeds each against `base_ttl` at 64, same N grid. At what N is
     "perturbed vs base" distinguishable from "base vs base"?
     Plus the paired (common-random-numbers) alternative, which costs no
     extra renders because the seed sets already overlap: Delta(s) =
     w_pert(s) - w_base(s) at a *shared* seed s, its variance decomposition,
     and whether it is stable across seeds.

  C  extrapolation (a residual bootstrap emulator, validated against the
     measured grid) and the wall-clock price, against the alternatives this
     repo already has.

Run from the repo root (so `assets/onnx` resolves):
    python3 py/phase3_seed_averaging.py                 # render + analyse
    python3 py/phase3_seed_averaging.py --analyze-only  # re-analyse cache

Writes only `py/results/phase3_seed_averaging/*.json` (gitignored).
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse task 18's renderer and statistics verbatim -- same instrument, same
# distance, same overlap report, so the two runs' numbers are comparable.
from phase3_attention_ladder import (  # noqa: E402
    CORPUS,
    LANG,
    ONNX_DIR,
    PRESETS,
    SPEED,
    TOTAL_STEP,
    VOICE_DIR,
    Renderer,
    dist_stats,
    overlap_report,
    to_py,
    tv,
)
from attention import export_attention_graph  # noqa: E402
from helper import load_voice_style  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "py", "results", "phase3_seed_averaging")

PRIMARY_TEXT_IDX = 0

N_SEEDS_PRESET = 64        # seeds per shipped preset (Part A, and base in B)
N_SEEDS_LADDER = 32        # seeds per ladder sample (Part B)
N_LADDER_STYLES = 24       # ttl_K64 samples
LADDER_SAMPLE_SEED = 20260924   # fixed: which ttl_K64 rows are used

N_GRID = [1, 2, 4, 8, 16, 32]
N_SPLITS_SAME = 50         # random disjoint splits per preset, per N
N_SPLITS_DIFF = 10         # random disjoint draws per preset pair, per N
N_SPLITS_B = 200           # draws per N for the Part B null
N_SPLITS_B_STYLE = 20      # draws per ladder style, per N
SPLIT_RNG_SEED = 424242

# Separation criterion, stated once and applied everywhere:
AUC_SEP = 0.99             # rank AUC of different-above-same
# ...AND no overlap of the two central 95% intervals, i.e.
# percentile(same, 97.5) < percentile(different, 2.5).

# Extrapolation grid for the residual-bootstrap emulator.
EMU_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
EMU_REPS = 400

# Reference numbers from task 18, carried in the output so every curve is
# readable against the run it follows (docs/ATTENTION_READOUT.md).
TASK18 = {
    "same_style_one_seed_tv_mean": 0.0328,
    "same_style_one_seed_tv_sd": 0.0114,
    "different_preset_one_seed_tv_mean": 0.0594,
    "different_preset_one_seed_tv_sd": 0.0310,
    "auc": 0.826,
    "ladder_eps0.2_tv_from_base_mean_shared_seed": 0.00846,
    "note": (
        "task 18's ladder TV was measured at a SHARED vocoder seed (base and "
        "perturbed both at seed 0), i.e. a paired/common-random-numbers "
        "design; its 0.00846 is not comparable to this run's disjoint-seed "
        "Part B numbers, but is comparable to Part B's paired section."
    ),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def write_json(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(to_py(payload), f, indent=1)
    print(f"[write] {path}", flush=True)
    return path


def safe_stats(values):
    """dist_stats, but an empty input is a result (e.g. every style's signal
    estimate came out non-positive), not a crash."""
    v = [float(x) for x in values if np.isfinite(x)]
    if not v:
        return {"n": 0, "note": "no finite values"}
    return dist_stats(v)


def central95(values):
    v = np.asarray(values, dtype=np.float64)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def separation_verdict(same, diff, auc):
    """The criterion, stated explicitly: AUC >= 0.99 AND the central 95%
    intervals do not overlap."""
    s_lo, s_hi = central95(same)
    d_lo, d_hi = central95(diff)
    ok_auc = bool(auc >= AUC_SEP)
    ok_iv = bool(d_lo > s_hi)
    return {
        "auc": float(auc),
        "auc_threshold": AUC_SEP,
        "auc_ok": ok_auc,
        "same_central95": [s_lo, s_hi],
        "different_central95": [d_lo, d_hi],
        "central95_disjoint": ok_iv,
        "separates": bool(ok_auc and ok_iv),
    }


def disjoint_split(rng, pool, n):
    """2n distinct seed indices from `pool`, split into two disjoint n-sets."""
    pick = rng.choice(pool, size=2 * n, replace=False)
    return pick[:n], pick[n:]


def avg(W, idx):
    return np.asarray(W)[np.asarray(idx, dtype=int)].mean(axis=0)


def _fit(fn, xs, ys, p0, bounds):
    try:
        popt, _ = curve_fit(fn, np.asarray(xs, float), np.asarray(ys, float),
                            p0=p0, bounds=bounds, maxfev=40000)
    except Exception as exc:  # pragma: no cover - fit failure is a result
        return None, None, str(exc)
    pred = fn(np.asarray(xs, float), *popt)
    ss_res = float(((np.asarray(ys, float) - pred) ** 2).sum())
    ss_tot = float(((np.asarray(ys, float) - np.mean(ys)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return [float(p) for p in popt], r2, None


def _sqrt_model(n, a, c):
    """Same-style mean TV: pure 1/sqrt(N) noise plus a non-averagable floor."""
    return a / np.sqrt(n) + c


def _power_model(n, a, b):
    return a * n ** b


def _bias_noise_model(n, c, a):
    """Different-style mean TV: a true separation c, blurred by 1/sqrt(N) noise."""
    return np.sqrt(c ** 2 + a ** 2 / n)


class ResidualBootstrap:
    """Emulator for 'average of N independent seeds', usable past the seeds
    actually rendered.

    Centres the style's per-seed profiles on their own mean and resamples the
    *residuals* with replacement. Each side's expectation is the centre
    exactly, so the same-style distance goes to 0 as N grows (unlike a naive
    bootstrap over the raw profiles split into two fixed pools, which would
    converge to the difference of the two pool means and invent a floor).

    It is still an extrapolation: it can only reshuffle the seed noise it has
    actually seen, and the centre itself carries the residual noise of the
    finite seed set. `emulator_validation` in the output is the check --
    it re-runs the emulator on the measured N grid and compares.
    """

    def __init__(self, W):
        W = np.asarray(W, dtype=np.float64)
        self.center = W.mean(axis=0)
        self.resid = W - self.center
        self.n = W.shape[0]
        # trace of the per-seed covariance: total seed-noise energy of one render
        self.noise_energy = float((self.resid ** 2).sum() / max(self.n - 1, 1))

    def draw(self, rng, n):
        idx = rng.integers(0, self.n, size=n)
        return self.center + self.resid[idx].mean(axis=0)


# --------------------------------------------------------------------------
# render phase
# --------------------------------------------------------------------------

def render_phase(args, corpus, primary_text):
    attention_path = os.path.join(ONNX_DIR, "vector_estimator.attn.onnx")
    export_attention_graph(
        os.path.join(ONNX_DIR, "vector_estimator.onnx"), attention_path
    )
    print(f"[setup] instrumented graph: {attention_path}", flush=True)
    rend = Renderer(attention_path)

    base_ttl = corpus["base_ttl"]
    sanity = {}
    sanity["base_ttl_is_M1_exactly"] = bool(
        np.array_equal(np.asarray(base_ttl), np.asarray(rend.m1.ttl[0]))
    )

    # Determinism under a seed: everything downstream assumes it.
    a = rend.render(base_ttl, primary_text, 0)
    b = rend.render(base_ttl, primary_text, 0)
    sanity["repeat_same_seed_matrix_bit_identical"] = bool(np.array_equal(a["M"], b["M"]))
    sanity["deterministic_under_seed"] = sanity["repeat_same_seed_matrix_bit_identical"]
    sanity["base_L"] = int(a["L"])
    sanity["base_duration_s"] = float(a["duration"])

    # The claim disjointness rests on: same seed + pinned style_dp => the same
    # noisy_latent, whatever style_ttl is. Checked directly, not assumed.
    tts = rend.tts
    ids0, mask0 = tts.text_processor([primary_text], [LANG])
    dur_m1, *_ = tts.dp_ort.run(
        None, {"text_ids": ids0, "style_dp": rend.m1.dp, "text_mask": mask0}
    )
    dur_m1 = dur_m1 / SPEED
    lat_a, _ = tts.sample_noisy_latent(dur_m1, seed=3)
    lat_b, _ = tts.sample_noisy_latent(dur_m1, seed=3)
    lat_c, _ = tts.sample_noisy_latent(dur_m1, seed=4)
    sanity["same_seed_same_noisy_latent"] = bool(np.array_equal(lat_a, lat_b))
    sanity["different_seed_different_noisy_latent"] = bool(not np.array_equal(lat_a, lat_c))
    sanity["noisy_latent_shape"] = list(np.asarray(lat_a).shape)

    if not sanity["deterministic_under_seed"]:
        write_json("sanity.json", sanity)
        raise SystemExit(
            "STOP: repeated render at one seed and style is not bit-identical."
        )

    # Which ttl_K64 rows Part B uses -- fixed sampling seed, recorded.
    srng = np.random.default_rng(LADDER_SAMPLE_SEED)
    n_avail = int(corpus["ttl_K64"].shape[0])
    ladder_idx = np.sort(
        srng.choice(n_avail, size=args.n_styles, replace=False)
    ).astype(int)

    profiles = {
        "meta": {
            "speed": SPEED,
            "total_step": TOTAL_STEP,
            "lang": LANG,
            "text": primary_text,
            "style_dp": "pinned to M1's for every render",
            "n_seeds_preset": args.n_seeds_preset,
            "n_seeds_ladder": args.n_seeds_ladder,
            "ladder_source": "ttl_K64 (eps=0.2, K=64) from phase2b_subspace",
            "ladder_sample_seed": LADDER_SAMPLE_SEED,
            "ladder_indices": ladder_idx.tolist(),
            "corpus": CORPUS,
        },
        "sanity": sanity,
        "presets": {},
        "ladder": {},
        "L": {},
        "frobenius_from_base": {},
    }
    all_L = set()

    t0 = time.time()
    for name in PRESETS:
        st = load_voice_style([os.path.join(VOICE_DIR, f"{name}.json")])
        ws, Ls = [], set()
        for s in range(args.n_seeds_preset):
            r = rend.render(st.ttl[0], primary_text, s)
            ws.append(r["w"].tolist())
            Ls.add(r["L"])
        if len(Ls) != 1:
            raise AssertionError(f"L not constant within preset {name}: {sorted(Ls)}")
        all_L |= Ls
        profiles["presets"][name] = ws
        profiles["L"][f"preset:{name}"] = int(next(iter(Ls)))
        profiles["frobenius_from_base"][f"preset:{name}"] = float(
            np.linalg.norm(np.asarray(st.ttl[0], np.float64) - np.asarray(base_ttl, np.float64))
        )
        print(f"[render] preset {name}: {args.n_seeds_preset} seeds "
              f"({rend.n_renders} renders, {time.time() - t0:.0f}s)", flush=True)
        write_json("profiles.json", profiles)

    ttl_all = corpus["ttl_K64"]
    for k, i in enumerate(ladder_idx):
        ttl = ttl_all[int(i)]
        ws, Ls = [], set()
        for s in range(args.n_seeds_ladder):
            r = rend.render(ttl, primary_text, s)
            ws.append(r["w"].tolist())
            Ls.add(r["L"])
        if len(Ls) != 1:
            raise AssertionError(f"L not constant within ladder style {i}: {sorted(Ls)}")
        all_L |= Ls
        profiles["ladder"][str(int(i))] = ws
        profiles["L"][f"ladder:{int(i)}"] = int(next(iter(Ls)))
        profiles["frobenius_from_base"][f"ladder:{int(i)}"] = float(
            np.linalg.norm(np.asarray(ttl, np.float64) - np.asarray(base_ttl, np.float64))
        )
        print(f"[render] ladder {k + 1}/{len(ladder_idx)} (corpus idx {int(i)}): "
              f"{args.n_seeds_ladder} seeds ({rend.n_renders} renders, "
              f"{time.time() - t0:.0f}s)", flush=True)
        write_json("profiles.json", profiles)

    if len(all_L) != 1:
        raise AssertionError(
            f"L is NOT constant across the whole run: {sorted(all_L)}. Every "
            "comparison here assumes one L; stopping."
        )
    profiles["meta"]["L_constant"] = int(next(iter(all_L)))
    profiles["meta"]["n_renders"] = int(rend.n_renders)
    profiles["meta"]["render_seconds_total"] = float(rend.t_render)
    profiles["meta"]["seconds_per_render"] = float(rend.t_render / rend.n_renders)
    profiles["meta"]["wall_seconds_total"] = float(time.time() - t0)
    write_json("profiles.json", profiles)
    print(f"[render] done: {rend.n_renders} renders, "
          f"{rend.t_render / rend.n_renders:.3f} s/render", flush=True)
    return profiles


# --------------------------------------------------------------------------
# Part A -- the separation curve for presets
# --------------------------------------------------------------------------

def part_a(profiles):
    W = {k: np.asarray(v, dtype=np.float64) for k, v in profiles["presets"].items()}
    names = list(W.keys())
    n_seeds = W[names[0]].shape[0]
    pool = np.arange(n_seeds)
    rng = np.random.default_rng(SPLIT_RNG_SEED)

    grid = [n for n in N_GRID if 2 * n <= n_seeds]
    by_N = {}
    same_by_N, diff_by_N = {}, {}
    for n in grid:
        same_vals = []
        for name in names:
            for _ in range(N_SPLITS_SAME):
                A, B = disjoint_split(rng, pool, n)
                same_vals.append(tv(avg(W[name], A), avg(W[name], B)))
        diff_vals = []
        for x, y in combinations(names, 2):
            for _ in range(N_SPLITS_DIFF):
                A, B = disjoint_split(rng, pool, n)
                diff_vals.append(tv(avg(W[x], A), avg(W[y], B)))
        rep = overlap_report(same_vals, diff_vals)
        rep["verdict"] = separation_verdict(same_vals, diff_vals,
                                            rep["auc_different_above_same"])
        by_N[n] = rep
        same_by_N[n] = np.asarray(same_vals)
        diff_by_N[n] = np.asarray(diff_vals)
        print(f"[A] N={n:3d} same {rep['same']['mean']:.5f}+/-{rep['same']['sd']:.5f}  "
              f"diff {rep['different']['mean']:.5f}+/-{rep['different']['sd']:.5f}  "
              f"AUC {rep['auc_different_above_same']:.4f}  "
              f"separates={rep['verdict']['separates']}", flush=True)

    ns = np.array(grid, dtype=float)
    same_means = np.array([by_N[n]["same"]["mean"] for n in grid])
    diff_means = np.array([by_N[n]["different"]["mean"] for n in grid])

    pw, pw_r2, pw_err = _fit(_power_model, ns, same_means, [0.03, -0.5],
                             ([1e-6, -3.0], [10.0, 0.0]))
    sq, sq_r2, sq_err = _fit(_sqrt_model, ns, same_means, [0.03, 0.0],
                             ([0.0, 0.0], [10.0, 1.0]))
    # pure 1/sqrt(N) with zero asymptote, for comparison
    pure_a = float(np.sum(same_means / np.sqrt(ns)) / np.sum(1.0 / ns))
    pure_pred = pure_a / np.sqrt(ns)
    pure_r2 = float(1.0 - ((same_means - pure_pred) ** 2).sum()
                    / ((same_means - same_means.mean()) ** 2).sum())
    pure_max_rel = float(np.abs(pure_pred - same_means).max() / same_means.min())

    # bootstrap CI on the fitted asymptote, resampling the split values
    boot_c = []
    brng = np.random.default_rng(SPLIT_RNG_SEED + 1)
    for _ in range(300):
        ys = []
        for n in grid:
            v = same_by_N[n]
            ys.append(float(v[brng.integers(0, v.size, size=v.size)].mean()))
        popt, _r2, _e = _fit(_sqrt_model, ns, np.asarray(ys), [0.03, 0.0],
                             ([0.0, 0.0], [10.0, 1.0]))
        if popt is not None:
            boot_c.append(popt[1])
    boot_c = np.asarray(boot_c) if boot_c else np.asarray([np.nan])

    dfit, dfit_r2, dfit_err = _fit(_bias_noise_model, ns, diff_means, [0.05, 0.03],
                                   ([0.0, 0.0], [10.0, 10.0]))

    # the paired (shared-seed) contrast, for reference: this is the design
    # task 18's ladder numbers used, and its same-style null is exactly 0
    # because a render is deterministic given (style, seed).
    paired_diff = []
    for x, y in combinations(names, 2):
        for s in range(n_seeds):
            paired_diff.append(tv(W[x][s], W[y][s]))

    # full-data estimate of the noise-free separation
    centers = {k: W[k].mean(axis=0) for k in names}
    full_pair_tv = {f"{x}|{y}": tv(centers[x], centers[y])
                    for x, y in combinations(names, 2)}
    noise_energy = {k: float(((W[k] - centers[k]) ** 2).sum() / (n_seeds - 1))
                    for k in names}

    return {
        "design": {
            "presets": names,
            "seeds_per_preset": int(n_seeds),
            "N_grid": grid,
            "same_style": f"two DISJOINT N-seed averages of one preset, "
                          f"{N_SPLITS_SAME} random splits x {len(names)} presets",
            "different_preset": f"N-seed averages of two presets on DISJOINT seed "
                                f"sets, {N_SPLITS_DIFF} draws x 45 pairs",
            "separation_criterion": f"AUC >= {AUC_SEP} AND central-95% intervals disjoint",
        },
        "by_N": {str(n): by_N[n] for n in grid},
        "curve": {
            "N": grid,
            "same_mean": same_means.tolist(),
            "same_sd": [by_N[n]["same"]["sd"] for n in grid],
            "different_mean": diff_means.tolist(),
            "different_sd": [by_N[n]["different"]["sd"] for n in grid],
            "auc": [by_N[n]["auc_different_above_same"] for n in grid],
            "ratio_of_means": [by_N[n]["ratio_of_means"] for n in grid],
        },
        "fits": {
            "same_power_law": {
                "model": "a * N**b", "a": pw[0] if pw else None,
                "b": pw[1] if pw else None, "r2": pw_r2, "error": pw_err,
                "note": "b = -0.5 is pure averagable noise",
            },
            "same_sqrt_plus_floor": {
                "model": "a / sqrt(N) + c", "a": sq[0] if sq else None,
                "asymptote_c": sq[1] if sq else None, "r2": sq_r2, "error": sq_err,
                "asymptote_bootstrap_ci95": [float(np.nanpercentile(boot_c, 2.5)),
                                             float(np.nanpercentile(boot_c, 97.5))],
                "asymptote_bootstrap_median": float(np.nanmedian(boot_c)),
            },
            "same_pure_inverse_sqrt": {
                "model": "a / sqrt(N), asymptote forced to 0",
                "a": pure_a, "r2": pure_r2,
                "max_abs_error_relative_to_smallest_measured": pure_max_rel,
                "predicted": pure_pred.tolist(),
            },
            "different_bias_plus_noise": {
                "model": "sqrt(c**2 + a**2 / N)",
                "asymptotic_separation_c": dfit[0] if dfit else None,
                "noise_a": dfit[1] if dfit else None, "r2": dfit_r2, "error": dfit_err,
            },
        },
        "noise_free_separation_estimates": {
            "tv_between_full_64_seed_means": dist_stats(list(full_pair_tv.values())),
            "per_pair": full_pair_tv,
            "per_render_seed_noise_energy_sum_var": noise_energy,
            "note": "TV between full-seed means is an upper bound: each centre "
                    "still carries the seed noise of a finite seed set.",
        },
        "paired_shared_seed_reference": {
            "different_preset_tv_at_shared_seed": dist_stats(paired_diff),
            "same_style_tv_at_shared_seed": 0.0,
            "note": "exactly 0 by determinism -- a paired design separates two "
                    "styles trivially at N=1, but tells you nothing about how "
                    "much of the difference is stable across seeds.",
        },
    }


# --------------------------------------------------------------------------
# Part B -- the ladder
# --------------------------------------------------------------------------

def part_b(profiles):
    Wb = np.asarray(profiles["presets"]["M1"], dtype=np.float64)   # base_ttl == M1
    n_base = Wb.shape[0]
    ladder = {k: np.asarray(v, dtype=np.float64) for k, v in profiles["ladder"].items()}
    keys = list(ladder.keys())
    n_lad = ladder[keys[0]].shape[0]
    rng = np.random.default_rng(SPLIT_RNG_SEED + 100)

    lad_pool = np.arange(n_lad)              # 0..31  (ladder renders exist here)
    base_hi = np.arange(n_lad, n_base)       # 32..63 (disjoint from every ladder seed)
    base_all = np.arange(n_base)

    grid = [n for n in N_GRID if n <= min(n_lad, base_hi.size) and 2 * n <= n_base]
    by_N = {}
    for n in grid:
        # null: base vs base, two disjoint N-seed averages
        null_vals = []
        for _ in range(N_SPLITS_B):
            A, B = disjoint_split(rng, base_all, n)
            null_vals.append(tv(avg(Wb, A), avg(Wb, B)))
        # test: ladder sample (N of its 32 seeds) vs base (N of seeds 32..63)
        reps = N_SPLITS_B_STYLE if n < n_lad else 1
        test_vals = []
        for k in keys:
            for _ in range(reps):
                A = rng.choice(lad_pool, size=n, replace=False)
                B = rng.choice(base_hi, size=n, replace=False)
                test_vals.append(tv(avg(ladder[k], A), avg(Wb, B)))
        rep = overlap_report(null_vals, test_vals)
        rep["verdict"] = separation_verdict(null_vals, test_vals,
                                            rep["auc_different_above_same"])
        by_N[n] = rep
        print(f"[B] N={n:3d} base-vs-base {rep['same']['mean']:.5f}  "
              f"ladder-vs-base {rep['different']['mean']:.5f}  "
              f"AUC {rep['auc_different_above_same']:.4f}  "
              f"separates={rep['verdict']['separates']}", flush=True)

    ns = np.array(grid, float)
    null_means = np.array([by_N[n]["same"]["mean"] for n in grid])
    test_means = np.array([by_N[n]["different"]["mean"] for n in grid])
    nfit, nfit_r2, nfit_err = _fit(_sqrt_model, ns, null_means, [0.03, 0.0],
                                   ([0.0, 0.0], [10.0, 1.0]))
    tfit, tfit_r2, tfit_err = _fit(_bias_noise_model, ns, test_means, [0.005, 0.03],
                                   ([0.0, 0.0], [10.0, 10.0]))

    # ---- paired / common-random-numbers analysis (no extra renders) --------
    # Delta(s) = w_pert(s) - w_base(s) at the SAME seed s, s = 0..n_lad-1.
    paired = {}
    paired_tv_all, snr1, n_for_snr1 = [], [], []
    half_a, half_b = np.arange(n_lad // 2), np.arange(n_lad // 2, n_lad)
    cos_halves, tv_half_a, tv_half_b = [], [], []
    unpaired_noise, paired_noise = [], []
    for k in keys:
        D = ladder[k][:n_lad] - Wb[:n_lad]                  # (32, 50)
        dbar = D.mean(axis=0)
        resid = D - dbar
        # per-seed noise energy of the paired difference
        e_noise = float((resid ** 2).sum() / (n_lad - 1))
        # unbiased estimate of the true (seed-free) difference energy
        e_signal = float((dbar ** 2).sum() - e_noise / n_lad)
        ptv = [0.5 * float(np.abs(d).sum()) for d in D]
        paired_tv_all.extend(ptv)
        snr = e_signal / e_noise if e_noise > 0 else float("nan")
        snr1.append(snr)
        # N such that the averaged noise energy falls to 1/4 of the signal
        # energy (a 2:1 amplitude margin): N = 4 / snr
        n_for_snr1.append(4.0 / snr if snr > 0 else float("inf"))
        da, db = D[half_a].mean(axis=0), D[half_b].mean(axis=0)
        cos = float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db)))
        cos_halves.append(cos)
        tv_half_a.append(0.5 * float(np.abs(da).sum()))
        tv_half_b.append(0.5 * float(np.abs(db).sum()))
        # noise energy of a single unpaired render, for the pairing gain
        wl = ladder[k][:n_lad]
        unpaired_noise.append(
            float(((wl - wl.mean(axis=0)) ** 2).sum() / (n_lad - 1))
            + float(((Wb[:n_lad] - Wb[:n_lad].mean(axis=0)) ** 2).sum() / (n_lad - 1))
        )
        paired_noise.append(e_noise)
        paired[k] = {
            "paired_tv_mean": float(np.mean(ptv)),
            "paired_tv_sd": float(np.std(ptv, ddof=1)),
            "signal_energy_unbiased": e_signal,
            "per_seed_noise_energy": e_noise,
            "snr_at_N1": snr,
            "half_split_cosine": cos,
        }
    rho_halves = spearmanr(tv_half_a, tv_half_b)

    paired_summary = {
        "definition": "Delta(s) = w_perturbed(s) - w_base(s) at a SHARED seed s",
        "null_is_exactly_zero": True,
        "paired_tv_per_seed": safe_stats(paired_tv_all),
        "task18_shared_seed_reference": TASK18["ladder_eps0.2_tv_from_base_mean_shared_seed"],
        "snr_at_N1": safe_stats(snr1),
        "n_styles_with_nonpositive_signal_estimate": int(
            sum(1 for v in snr1 if not (v > 0))
        ),
        "N_for_2to1_amplitude_margin": safe_stats(
            [v for v in n_for_snr1 if np.isfinite(v) and v > 0]
        ),
        "half_split_cosine_of_mean_difference": safe_stats(cos_halves),
        "half_split_tv_rank_spearman": {
            "rho": float(rho_halves.statistic), "p": float(rho_halves.pvalue),
            "n": len(tv_half_a),
        },
        "pairing_variance_reduction": {
            "unpaired_noise_energy_mean": float(np.mean(unpaired_noise)),
            "paired_noise_energy_mean": float(np.mean(paired_noise)),
            "ratio_unpaired_over_paired": float(
                np.mean(unpaired_noise) / np.mean(paired_noise)
            ),
            "note": "how many unpaired renders one paired render is worth on "
                    "noise energy; 1.0 would mean pairing buys nothing.",
        },
        "per_style": paired,
    }

    # Condition-matching check the AUC depends on. The test side's spread is
    # (noise_ladder + noise_base)/N and the null side's is 2*noise_base/N, so
    # if the ladder styles are less seed-noisy than base, the test distances
    # shrink relative to the null for a reason that has nothing to do with the
    # perturbation -- and the AUC can drop BELOW 0.5 without the perturbation
    # having failed to land. Reported so the AUC can be read honestly.
    base_noise = float(((Wb - Wb.mean(axis=0)) ** 2).sum() / (n_base - 1))
    lad_noise = {k: float(((v - v.mean(axis=0)) ** 2).sum() / (n_lad - 1))
                 for k, v in ladder.items()}
    noise_check = {
        "base_per_render_noise_energy": base_noise,
        "ladder_per_render_noise_energy": safe_stats(list(lad_noise.values())),
        "ratio_ladder_over_base_mean": float(np.mean(list(lad_noise.values())) / base_noise),
        "predicted_null_over_test_spread_ratio": float(
            np.sqrt(2 * base_noise / (np.mean(list(lad_noise.values())) + base_noise))
        ),
        "note": "a ratio != 1.0 means the two sides are not noise-matched; "
                "values < 1 deflate the test side and push AUC below 0.5 "
                "independently of any perturbation effect.",
    }

    frob = [profiles["frobenius_from_base"][f"ladder:{k}"] for k in keys]
    return {
        "noise_energy_check": noise_check,
        "design": {
            "n_styles": len(keys),
            "corpus_indices": [int(k) for k in keys],
            "seeds_per_style": int(n_lad),
            "seeds_for_base": int(n_base),
            "N_grid": grid,
            "null": "base_ttl vs base_ttl, two disjoint N-seed averages from its 64",
            "test": "ladder sample over N of seeds 0-31 vs base over N of seeds "
                    "32-63 (seed sets always disjoint, so no shared noisy_latent)",
            "eps": 0.2,
            "frobenius_from_base": dist_stats(frob),
        },
        "by_N": {str(n): by_N[n] for n in grid},
        "curve": {
            "N": grid,
            "base_vs_base_mean": null_means.tolist(),
            "base_vs_base_sd": [by_N[n]["same"]["sd"] for n in grid],
            "ladder_vs_base_mean": test_means.tolist(),
            "ladder_vs_base_sd": [by_N[n]["different"]["sd"] for n in grid],
            "auc": [by_N[n]["auc_different_above_same"] for n in grid],
        },
        "fits": {
            "null_sqrt_plus_floor": {
                "model": "a / sqrt(N) + c", "a": nfit[0] if nfit else None,
                "asymptote_c": nfit[1] if nfit else None, "r2": nfit_r2,
                "error": nfit_err,
            },
            "ladder_bias_plus_noise": {
                "model": "sqrt(c**2 + a**2 / N)",
                "asymptotic_separation_c": tfit[0] if tfit else None,
                "noise_a": tfit[1] if tfit else None, "r2": tfit_r2, "error": tfit_err,
            },
        },
        "detection_N_measured": next(
            (n for n in grid if by_N[n]["verdict"]["separates"]), None
        ),
        "paired_alternative": paired_summary,
    }


# --------------------------------------------------------------------------
# Part C -- extrapolation and price
# --------------------------------------------------------------------------

def emulate(bs_a, bs_b, n, reps, rng):
    return np.asarray([tv(bs_a.draw(rng, n), bs_b.draw(rng, n)) for _ in range(reps)])


def extrapolate(same_sources, diff_source_pairs, label, measured_by_N):
    """Residual-bootstrap emulator, validated on the measured grid, then run
    past it. `same_sources` is a list of (bootstrap, bootstrap) for the
    same-style side (two independent draws from one style's residual pool);
    `diff_source_pairs` the corresponding different-style pairs."""
    rng = np.random.default_rng(SPLIT_RNG_SEED + 999)
    rows = {}
    for n in EMU_GRID:
        same_vals, diff_vals = [], []
        per_source = max(1, EMU_REPS // max(len(same_sources), 1))
        for bs in same_sources:
            same_vals.append(emulate(bs, bs, n, per_source, rng))
        per_pair = max(1, EMU_REPS // max(len(diff_source_pairs), 1))
        for ba, bb in diff_source_pairs:
            diff_vals.append(emulate(ba, bb, n, per_pair, rng))
        same_vals = np.concatenate(same_vals)
        diff_vals = np.concatenate(diff_vals)
        rep = overlap_report(same_vals, diff_vals)
        v = separation_verdict(same_vals, diff_vals, rep["auc_different_above_same"])
        rows[n] = {
            "same_mean": rep["same"]["mean"], "same_sd": rep["same"]["sd"],
            "different_mean": rep["different"]["mean"],
            "different_sd": rep["different"]["sd"],
            "auc": rep["auc_different_above_same"],
            "verdict": v,
            "measured": bool(n in measured_by_N),
        }
    # validation against the measured grid
    val = []
    for n_str, m in measured_by_N.items():
        n = int(n_str)
        if n in rows:
            val.append({
                "N": n,
                "measured_same_mean": m["same"]["mean"],
                "emulated_same_mean": rows[n]["same_mean"],
                "same_ratio_emu_over_measured": rows[n]["same_mean"] / m["same"]["mean"],
                "measured_different_mean": m["different"]["mean"],
                "emulated_different_mean": rows[n]["different_mean"],
                "different_ratio_emu_over_measured":
                    rows[n]["different_mean"] / m["different"]["mean"],
                "measured_auc": m["auc_different_above_same"],
                "emulated_auc": rows[n]["auc"],
            })
    n_needed = next((n for n in EMU_GRID if rows[n]["verdict"]["separates"]), None)
    return {
        "label": label,
        "note": "EXTRAPOLATION past N=32: a residual bootstrap can only "
                "reshuffle seed noise it has already seen, and the per-style "
                "centre still carries the finite-seed noise of 32-64 renders, "
                "which biases the different-style side UP and therefore the "
                "required N DOWN. Read `validation` before using `N_required`.",
        "validation": val,
        "by_N": {str(n): rows[n] for n in EMU_GRID},
        "N_required": n_needed,
        "N_required_is_extrapolated": bool(n_needed is not None and n_needed > max(N_GRID)),
    }


def analytic_required_N(profiles, b_res):
    """A bias-corrected cross-check on the emulator's N_required for the ladder.

    The emulator's different-style side converges to TV between two *finite
    seed set* centres, which still carry seed noise -- so its asymptote is
    biased up and its N_required down. This estimates each style's noise-free
    separation without that bias, using the fact that two DISJOINT halves of
    its seeds give independent noise:

        E[ da . db ] = ||Delta_true||^2          (cross-product, unbiased)

    where da, db are the mean paired difference over each half. Converting
    that energy to a TV with the observed difference's shape,

        TV_inf ~= 0.5 * ||Dbar||_1 * ||Delta_true||_2 / ||Dbar||_2

    and requiring the null's upper 95% tail (which falls as 1/sqrt(N)) to sit
    below it gives N = (tail_factor * a_null / TV_inf)^2.
    """
    Wb = np.asarray(profiles["presets"]["M1"], dtype=np.float64)
    ladder = {k: np.asarray(v, dtype=np.float64) for k, v in profiles["ladder"].items()}
    n_lad = ladder[list(ladder)[0]].shape[0]
    ha, hb = np.arange(n_lad // 2), np.arange(n_lad // 2, n_lad)

    grid = sorted(int(k) for k in b_res["by_N"])
    n_max = str(max(grid))
    null_mean = b_res["by_N"][n_max]["same"]["mean"]
    null_hi = b_res["by_N"][n_max]["verdict"]["same_central95"][1]
    tail_factor = null_hi / null_mean
    a_null = b_res["fits"]["null_sqrt_plus_floor"]["a"]

    per_style, tvs, reqs = {}, [], []
    for k, v in ladder.items():
        D = v[:n_lad] - Wb[:n_lad]
        dbar = D.mean(axis=0)
        da, db = D[ha].mean(axis=0), D[hb].mean(axis=0)
        energy = float(da @ db)                      # unbiased ||Delta_true||^2
        l2 = float(np.linalg.norm(dbar))
        l1 = float(np.abs(dbar).sum())
        tv_inf = 0.5 * l1 * (np.sqrt(energy) / l2) if energy > 0 else 0.0
        req = float((tail_factor * a_null / tv_inf) ** 2) if tv_inf > 0 else float("inf")
        per_style[k] = {
            "true_difference_energy_unbiased": energy,
            "tv_of_32seed_mean_difference": 0.5 * l1,
            "tv_inf_estimate": tv_inf,
            "required_N": req,
        }
        tvs.append(tv_inf)
        reqs.append(req)
    return {
        "method": "disjoint-half cross-product, shape-preserving TV conversion",
        "tail_factor_of_null_at_N": {"N": int(n_max), "value": tail_factor},
        "a_null_per_sqrtN": a_null,
        "tv_inf": safe_stats(tvs),
        "required_N": safe_stats([r for r in reqs if np.isfinite(r)]),
        "required_N_for_the_median_style": float(np.median(
            [r for r in reqs if np.isfinite(r)])),
        "required_N_for_the_weakest_style": float(np.max(
            [r for r in reqs if np.isfinite(r)])),
        "note": "compare with the emulator's N_required: the emulator is "
                "optimistic by roughly the square of the ratio of the two "
                "asymptotes, because its centres are not noise-free.",
        "per_style": per_style,
    }


def part_c(profiles, a_res, b_res):
    W = {k: np.asarray(v, dtype=np.float64) for k, v in profiles["presets"].items()}
    ladder = {k: np.asarray(v, dtype=np.float64) for k, v in profiles["ladder"].items()}

    bs_presets = {k: ResidualBootstrap(v) for k, v in W.items()}
    preset_extrap = extrapolate(
        list(bs_presets.values()),
        [(bs_presets[x], bs_presets[y]) for x, y in combinations(bs_presets, 2)],
        "different shipped presets",
        a_res["by_N"],
    )

    bs_base = ResidualBootstrap(W["M1"])
    bs_lad = {k: ResidualBootstrap(v) for k, v in ladder.items()}
    ladder_extrap = extrapolate(
        [bs_base],
        [(bs_lad[k], bs_base) for k in bs_lad],
        "eps-0.2 ladder perturbation vs base",
        b_res["by_N"],
    )

    spr = float(profiles["meta"]["seconds_per_render"])

    def price(n, n_styles, share_base=True):
        renders = n_styles * n + (n if share_base else n_styles * n)
        return {
            "N": n, "n_styles": n_styles, "renders": int(renders),
            "seconds": float(renders * spr),
            "hours": float(renders * spr / 3600.0),
        }

    n_preset = preset_extrap["N_required"]
    n_ladder = ladder_extrap["N_required"]

    costs = {
        "seconds_per_render_measured": spr,
        "renders_this_run": int(profiles["meta"]["n_renders"]),
        "wall_seconds_this_run": float(profiles["meta"]["wall_seconds_total"]),
        "scoring_240_ladder_samples": {
            f"N={n}": price(n, 240) for n in [1, 8, 32, 128, 512]
        },
        "scoring_1280_corpus": {
            f"N={n}": price(n, 1280) for n in [1, 8, 32, 128, 512]
        },
        "at_required_N": {
            "presets_240": price(n_preset, 240) if n_preset else None,
            "ladder_240": price(n_ladder, 240) if n_ladder else None,
            "ladder_1280": price(n_ladder, 1280) if n_ladder else None,
        },
        "alternatives": {
            "wavlm_probe_pass": {
                "source": "py/results/phase2b_subspace/{generate,embed}.log",
                "render_seconds_per_clip": 1.57,
                "embed_seconds_per_clip": 0.95,
                "total_seconds_per_sample": 2.52,
                "seconds_1280": 3226.0,
                "hours_1280": 0.90,
                "note": "one render per sample, no seed averaging; and it is a "
                        "calibrated-against-held-out-speakers instrument only "
                        "to the extent phase2b established, which was partial.",
            },
            "human_listening_bench": {
                "source": "docs/LISTENING_BENCHES.md",
                "clips_per_bench": 20,
                "renders_per_bench": 20,
                "render_seconds": 31.4,
                "human_minutes_per_bench": "10-20 (the real cost)",
                "note": "does not scale to 240 or 1280 samples at all; it "
                        "answers a different, higher-authority question.",
            },
            "paired_common_random_numbers_at_2to1_margin": {
                "paired_seeds_median_style": float(
                    b_res["paired_alternative"]["N_for_2to1_amplitude_margin"]["median"]
                ),
                "paired_seeds_mean_style": float(
                    b_res["paired_alternative"]["N_for_2to1_amplitude_margin"]["mean"]
                ),
                "renders_per_sample_at_median": float(
                    2 * b_res["paired_alternative"]["N_for_2to1_amplitude_margin"]["median"]
                ),
                "hours_240": float(
                    2 * b_res["paired_alternative"]["N_for_2to1_amplitude_margin"]["median"]
                    * 240 * spr / 3600.0
                ),
                "hours_1280": float(
                    2 * b_res["paired_alternative"]["N_for_2to1_amplitude_margin"]["median"]
                    * 1280 * spr / 3600.0
                ),
                "note": "base and perturbed rendered at the SAME seeds; pairing "
                        "cuts the noise energy by the measured factor in "
                        "part_b.paired_alternative.pairing_variance_reduction.",
            },
            "paired_common_random_numbers_single_seed": {
                "renders_per_sample": 2,
                "seconds_per_sample": float(2 * spr),
                "seconds_240": float(2 * 240 * spr),
                "note": "base and perturbed at the SAME seed. Detection is "
                        "trivial (null is exactly 0); see part_b."
                        "paired_alternative for whether the measured "
                        "difference is stable across seeds.",
            },
        },
    }

    analytic = analytic_required_N(profiles, b_res)
    ladder_extrap["analytic_cross_check"] = analytic
    emu_asym = ladder_extrap["by_N"][str(EMU_GRID[-1])]["different_mean"]
    fit_asym = b_res["fits"]["ladder_bias_plus_noise"]["asymptotic_separation_c"]
    ladder_extrap["asymptote_bias_check"] = {
        "emulator_asymptotic_different_mean": emu_asym,
        "curve_fit_asymptotic_separation": fit_asym,
        "ratio_emu_over_fit": float(emu_asym / fit_asym) if fit_asym else None,
        "implied_optimism_factor_on_N": float((emu_asym / fit_asym) ** 2) if fit_asym else None,
        "note": "N scales as 1/TV^2, so an asymptote overstated by r means "
                "N_required is understated by r^2.",
    }

    n_ladder_corrected = None
    if ladder_extrap["N_required"] and fit_asym:
        n_ladder_corrected = float(
            ladder_extrap["N_required"] * (emu_asym / fit_asym) ** 2
        )
    costs["at_required_N"]["ladder_240_bias_corrected"] = (
        price(int(round(n_ladder_corrected)), 240) if n_ladder_corrected else None
    )
    costs["at_required_N"]["ladder_1280_bias_corrected"] = (
        price(int(round(n_ladder_corrected)), 1280) if n_ladder_corrected else None
    )
    costs["at_required_N"]["ladder_N_bias_corrected"] = n_ladder_corrected

    return {
        "presets_extrapolation": preset_extrap,
        "ladder_extrapolation": ladder_extrap,
        "cost": costs,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true",
                    help="re-analyse the cached profiles.json, render nothing")
    ap.add_argument("--n-seeds-preset", type=int, default=N_SEEDS_PRESET)
    ap.add_argument("--n-seeds-ladder", type=int, default=N_SEEDS_LADDER)
    ap.add_argument("--n-styles", type=int, default=N_LADDER_STYLES)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join(CORPUS, "manifest.json")))
    primary_text = manifest["texts"][PRIMARY_TEXT_IDX]
    z = np.load(os.path.join(CORPUS, "subspace.npz"))
    corpus = {k: z[k] for k in ("base_ttl", "active_rows", "ttl_K64")}

    cache = os.path.join(OUT_DIR, "profiles.json")
    if args.analyze_only:
        profiles = json.load(open(cache))
        print(f"[cache] {cache}: {profiles['meta']['n_renders']} renders", flush=True)
    else:
        profiles = render_phase(args, corpus, primary_text)

    print("[analysis] Part A", flush=True)
    a_res = part_a(profiles)
    write_json("part_a_presets.json", {"task18_reference": TASK18, **a_res})

    print("[analysis] Part B", flush=True)
    b_res = part_b(profiles)
    write_json("part_b_ladder.json", {"task18_reference": TASK18, **b_res})

    print("[analysis] Part C", flush=True)
    c_res = part_c(profiles, a_res, b_res)
    write_json("part_c_cost.json", c_res)

    summary = {
        "meta": profiles["meta"],
        "sanity": profiles["sanity"],
        "task18_reference": TASK18,
        "part_a_curve": a_res["curve"],
        "part_a_fits": a_res["fits"],
        "part_a_noise_free_separation": a_res["noise_free_separation_estimates"][
            "tv_between_full_64_seed_means"],
        "part_a_separates_at_N": next(
            (n for n in a_res["curve"]["N"]
             if a_res["by_N"][str(n)]["verdict"]["separates"]), None),
        "part_b_curve": b_res["curve"],
        "part_b_fits": b_res["fits"],
        "part_b_detection_N_measured": b_res["detection_N_measured"],
        "part_b_noise_energy_check": b_res["noise_energy_check"],
        "part_b_paired": {
            k: v for k, v in b_res["paired_alternative"].items() if k != "per_style"
        },
        "part_c_N_required_presets": c_res["presets_extrapolation"]["N_required"],
        "part_c_N_required_ladder": c_res["ladder_extrapolation"]["N_required"],
        "part_c_ladder_asymptote_bias_check":
            c_res["ladder_extrapolation"]["asymptote_bias_check"],
        "part_c_ladder_analytic_required_N": {
            k: v for k, v in
            c_res["ladder_extrapolation"]["analytic_cross_check"].items()
            if k != "per_style"
        },
        "part_c_cost": c_res["cost"],
    }
    write_json("summary.json", summary)
    print(json.dumps(to_py(summary["part_a_curve"]), indent=1))
    print(json.dumps(to_py(summary["part_b_curve"]), indent=1))


if __name__ == "__main__":
    main()
