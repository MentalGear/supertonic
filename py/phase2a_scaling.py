"""Phase 2a scaling: how does audio-readout recovery (WavLM -> c_realized)
scale with training-set size n_train?

Phase 2b's original probe (see ../new-plan.md and phase2b_subspace_probe.py)
failed at 6144 target dims and n_train=240 because a ridge's predictions
span at most n_train dimensions of the target -- the subspace ladder fixed
that by shrinking the target to K in {4, 16, 64}. This script asks the next
question: at fixed K, how much does recovery improve as n_train grows, and
has it saturated at the largest n_train currently on disk? The answer sizes
how much GPU time a larger corpus is worth buying.

For each K this refits the exact same ridge probe as phase2b_subspace_probe
(WavLM meanstd features -> c_realized), at a sweep of training-set sizes,
subsampling the FIXED training split at random (several reps per n_train,
since small n_train is noisy) while holding the test split completely fixed
across every n_train and every rep, so the curve is not confounded by
test-set changes. At each (K, n_train, rep) it also recomputes the oracle
ceiling exactly as phase2a_ceiling_audit does (orthogonal projection of the
test targets onto row_space(train targets)) -- this is expected to be
rank-starved (< 1.0) at small n_train against a K-dim target and should stop
binding once n_train >> K, which is the point where the achieved curve
starts measuring audio recoverability rather than estimator rank.

Data (see ../new-plan.md and phase2b_subspace_probe.py's docstring for how
it was generated):
  - results/phase2b_subspace/ (eps=0.20): K=4 (n_train=240), K=16
    (n_train=240), K=64 (n_train=480). This script sweeps n_train up to each
    K's full training split.

Reuses (does not reimplement) phase2b_subspace_probe.build_xy (the WavLM /
c_realized id-matching and train/test split), .per_component_r2,
.mean_cosine, .pooled_r2; phase2b_probe.fit_ridge; and
phase2a_ceiling_audit.numerical_rank / .oracle_projection -- the same
statistical machinery used everywhere else in phase 2, so this curve is
directly comparable to phase2b_subspace_probe's single-n_train numbers.

Usage (from py/):
    python3 phase2a_scaling.py [--reps 5] [--out results/phase2a/scaling.json]
"""

import argparse
import json
import os

import numpy as np

from phase2a_ceiling_audit import numerical_rank, oracle_projection
from phase2b_probe import fit_ridge
from phase2b_subspace_probe import build_xy, mean_cosine, per_component_r2, pooled_r2

SUBSPACE_DIR = "results/phase2b_subspace"
OUT_PATH = "results/phase2a/scaling.json"
N_REPS = 5
LAYERS = [3, 4, 5]

# n_train sweep values per K, capped at each K's full training split
# (K=64 has 480 train samples; K=4/K=16 have 240).
N_TRAIN_SWEEP = {
    4: [30, 60, 120, 240],
    16: [30, 60, 120, 240],
    64: [30, 60, 120, 240, 360, 480],
}


def fit_one_rep(Xtr_full, Ytr_full, Xte, Yte, n_train, rng):
    sub = rng.choice(len(Xtr_full), size=n_train, replace=False)
    Xtr, Ytr = Xtr_full[sub], Ytr_full[sub]

    Ypred, alpha = fit_ridge(Xtr, Ytr, Xte)
    comp_r2 = per_component_r2(Yte, Ypred)
    cos = mean_cosine(Yte, Ypred)
    pooled = pooled_r2(Ytr, Yte, Ypred)

    rank_eps, rank_loose, s = numerical_rank(Ytr)
    P_oracle, _ = oracle_projection(Ytr, Yte, rank_eps)
    oracle_r2 = pooled_r2(Ytr, Yte, P_oracle)

    return {
        "ridge_alpha": alpha,
        "per_component_r2_mean": float(np.nanmean(comp_r2)),
        "per_component_r2_median": float(np.nanmedian(comp_r2)),
        "mean_cosine": cos,
        "pooled_r2": pooled,
        "oracle_rank": rank_eps,
        "oracle_r2": oracle_r2,
    }


def aggregate(reps):
    keys = ["per_component_r2_mean", "per_component_r2_median", "mean_cosine",
            "pooled_r2", "oracle_r2", "oracle_rank"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in reps], dtype=np.float64)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
    return out


def log_linear_fit(n_train_vals, y_vals):
    """Least-squares fit of y against log(n_train). Returns slope, intercept,
    and the Pearson r of the fit (not an R^2 against a variance baseline --
    just how linear the achieved-vs-log(n) relationship is)."""
    x = np.log(np.asarray(n_train_vals, dtype=np.float64))
    y = np.asarray(y_vals, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    return {"slope_per_log_n": float(slope), "intercept": float(intercept), "pearson_r": r}


def run_k(K, recs_k, feats, subspace, layers, n_reps):
    X, Y, rms_pre, itr, ite, n_matched, missing = build_xy(K, recs_k, feats, subspace, layers)
    Xtr_full, Ytr_full = X[itr], Y[itr]
    Xte, Yte = X[ite], Y[ite]
    n_train_full = len(itr)

    sweep_vals = [n for n in N_TRAIN_SWEEP[K] if n <= n_train_full]

    sweep = []
    for n_train in sweep_vals:
        reps = []
        for rep in range(n_reps):
            rng = np.random.default_rng(1_000_000 * K + 1_000 * n_train + rep)
            reps.append(fit_one_rep(Xtr_full, Ytr_full, Xte, Yte, n_train, rng))
        agg = aggregate(reps)
        sweep.append({"n_train": n_train, "n_reps": n_reps, "reps": reps, **agg})
        print(f"K={K:<4} n_train={n_train:<5} "
              f"achieved_mean_R2={agg['per_component_r2_mean_mean']:.4f}"
              f"+-{agg['per_component_r2_mean_std']:.4f} "
              f"achieved_median_R2={agg['per_component_r2_median_mean']:.4f} "
              f"oracle_R2={agg['oracle_r2_mean']:.4f}+-{agg['oracle_r2_std']:.4f} "
              f"oracle_rank={agg['oracle_rank_mean']:.1f} "
              f"cosine={agg['mean_cosine_mean']:.4f}", flush=True)

    trend_mean = log_linear_fit([s["n_train"] for s in sweep],
                                 [s["per_component_r2_mean_mean"] for s in sweep])
    trend_median = log_linear_fit([s["n_train"] for s in sweep],
                                   [s["per_component_r2_median_mean"] for s in sweep])

    # Same fit restricted to points where the oracle ceiling is no longer
    # rank-starved (>=0.999): at small n_train against a K-dim target the
    # achieved curve is depressed by estimator rank, not audio, and mixing
    # those points into the trend biases any extrapolation.
    clean = [s for s in sweep if s["oracle_r2_mean"] >= 0.999]
    trend_mean_clean = (log_linear_fit([s["n_train"] for s in clean],
                                        [s["per_component_r2_mean_mean"] for s in clean])
                         if len(clean) >= 2 else None)

    # last-step marginal slope, to check for saturation independent of the
    # global least-squares fit (a flattening tail can hide inside a decent
    # overall pearson_r if the early points are steep).
    last_step = None
    if len(sweep) >= 2:
        n0, n1 = sweep[-2]["n_train"], sweep[-1]["n_train"]
        r0, r1 = sweep[-2]["per_component_r2_mean_mean"], sweep[-1]["per_component_r2_mean_mean"]
        last_step = {
            "from_n_train": n0, "to_n_train": n1,
            "delta_r2_mean": float(r1 - r0),
            "slope_per_log_n": float((r1 - r0) / (np.log(n1) - np.log(n0))),
        }

    ceiling_still_binding = [s["n_train"] for s in sweep if s["oracle_r2_mean"] < 0.999]

    return {
        "K": K,
        "n_train_full": n_train_full,
        "n_test": len(ite),
        "n_matched": n_matched,
        "n_missing_from_feats": missing,
        "sweep": sweep,
        "trend_mean_r2_vs_log_n": trend_mean,
        "trend_median_r2_vs_log_n": trend_median,
        "trend_mean_r2_vs_log_n_rank_clean": trend_mean_clean,
        "last_step": last_step,
        "oracle_ceiling_still_binding_at_n_train": ceiling_still_binding,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=SUBSPACE_DIR)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--subspace", default=None)
    ap.add_argument("--feats", default=None)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--layers", default=",".join(str(x) for x in LAYERS),
                     help="1-based WavLM layers, comma-separated")
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--k", type=int, default=None, help="only run this K")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.json")
    subspace_path = args.subspace or os.path.join(args.out_dir, "subspace.npz")
    feats_path = args.feats or os.path.join(args.out_dir, "wavlm_feats.npz")

    with open(manifest_path) as f:
        meta = json.load(f)
    records = meta["records"]
    subspace = np.load(subspace_path)
    feats = np.load(feats_path)

    k_list = [args.k] if args.k is not None else sorted(N_TRAIN_SWEEP.keys() & {int(r["K"]) for r in records})

    report = {
        "experiment": "phase2a_scaling",
        "manifest": manifest_path,
        "subspace": subspace_path,
        "feats": feats_path,
        "layers_1based": layers,
        "n_reps": args.reps,
        "n_train_sweep_config": N_TRAIN_SWEEP,
        "target": "c_realized (realized subspace coefficients), test split fixed across n_train and reps",
        "k_list": k_list,
        "results": {},
    }

    for K in k_list:
        recs_k = [r for r in records if int(r["K"]) == K]
        res = run_k(K, recs_k, feats, subspace, layers, args.reps)
        report["results"][str(K)] = res

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSummary (mean per-component R^2 across reps, +- std):")
    hdr = f"{'K':>5}{'n_train':>9}{'achieved_R2':>14}{'std':>8}{'median_R2':>12}{'oracle_R2':>12}{'oracle_rank':>13}{'cosine':>9}"
    print(hdr)
    print("-" * len(hdr))
    for K in k_list:
        for s in report["results"][str(K)]["sweep"]:
            print(f"{K:>5}{s['n_train']:>9}"
                  f"{s['per_component_r2_mean_mean']:>14.4f}"
                  f"{s['per_component_r2_mean_std']:>8.4f}"
                  f"{s['per_component_r2_median_mean']:>12.4f}"
                  f"{s['oracle_r2_mean']:>12.4f}"
                  f"{s['oracle_rank_mean']:>13.1f}"
                  f"{s['mean_cosine_mean']:>9.4f}")
        t = report["results"][str(K)]["trend_mean_r2_vs_log_n"]
        tc = report["results"][str(K)]["trend_mean_r2_vs_log_n_rank_clean"]
        ls = report["results"][str(K)]["last_step"]
        binding = report["results"][str(K)]["oracle_ceiling_still_binding_at_n_train"]
        tc_str = (f"slope={tc['slope_per_log_n']:.4f}, pearson_r={tc['pearson_r']:.4f}"
                  if tc is not None else "n/a")
        print(f"  K={K} trend (all pts): slope={t['slope_per_log_n']:.4f} per log(n), "
              f"pearson_r={t['pearson_r']:.4f}; "
              f"trend (rank-clean pts): {tc_str}; "
              f"last step slope={ls['slope_per_log_n']:.4f} "
              f"({ls['from_n_train']}->{ls['to_n_train']}); "
              f"oracle ceiling <0.999 at n_train in {binding}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
