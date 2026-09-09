"""Phase 2b, stage 3: linear (and kernel) probe from speaker embedding -> style_ttl.

See ../new-plan.md Phase 2b. The question is whether an off-the-shelf speaker
embedding linearly predicts the style tensor, and the answer is only readable
next to the intrinsic dimensionality of the style set that was sampled -- a
probe fitted on convex blends of ten presets has to recover at most 9
directions and would score near 1.0 while proving nothing.

For every condition this reports, per feature set (ECAPA, and a dumb MFCC-moment
baseline):

  * intrinsic dimensionality of the sampled style set (designed, and empirical
    effective rank / PCA curve, which is censored by the sample count);
  * the structural ceiling: what fraction of style variance lies in the top-192
    principal directions, since a linear map from a 192-dim input cannot span
    more than 192 output directions;
  * R^2 on a FAMILY-DISJOINT split (whole base presets held out) and on a random
    split, so split-induced inflation is visible;
  * Hewitt & Liang control-task selectivity: the identical probe refitted
    against permuted targets, reported as real minus control;
  * per-active-row R^2, not just the aggregate, and the whole-tensor R^2 which
    the 26 near-constant rows inflate;
  * for the perturbation conditions, R^2 on the WITHIN-FAMILY RESIDUAL (target
    minus its base preset), which is the part of the style that is genuinely
    high-dimensional rather than a 7-way preset identification.

Usage (from py/):
    python3 phase2b_probe.py [--kernel] [--out results/phase2b/probe_report.json]
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results/phase2b"
VOICE_STYLE_DIR = "assets/voice_styles"
ALPHAS = np.logspace(-3, 8, 34)
RNG = np.random.default_rng(7)


def r2_parts(y_true, y_pred, baseline):
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - baseline) ** 2).sum(0)
    return ss_res, ss_tot


def agg_r2(ss_res, ss_tot):
    """Variance-weighted R^2 over the given dims (the pooled, honest aggregate)."""
    tot = ss_tot.sum()
    return float(1.0 - ss_res.sum() / tot) if tot > 0 else float("nan")


def fit_ridge(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=ALPHAS)  # efficient LOO generalized CV, one alpha for all targets
    model.fit(sc.transform(Xtr), Ytr)
    return model.predict(sc.transform(Xte)), float(model.alpha_)


def _sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def fit_kernel(Xtr, Ytr, Xte):
    """RBF kernel ridge, dual form (n x n). Nonlinear probe; alpha by inner split."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    d2 = _sqdist(A, A)
    gamma = 1.0 / np.median(d2[d2 > 0])
    n = len(A)
    cut = int(n * 0.8)
    perm = np.random.default_rng(3).permutation(n)
    itr, iva = perm[:cut], perm[cut:]
    K = np.exp(-gamma * d2)
    best, best_a = -np.inf, ALPHAS[0]
    for a in np.logspace(-6, 3, 19):
        w = np.linalg.solve(K[np.ix_(itr, itr)] + a * np.eye(len(itr)), Ytr[itr])
        pv = K[np.ix_(iva, itr)] @ w
        ss_res, ss_tot = r2_parts(Ytr[iva], pv, Ytr[itr].mean(0))
        s = agg_r2(ss_res, ss_tot)
        if s > best:
            best, best_a = s, a
    w = np.linalg.solve(K + best_a * np.eye(n), Ytr)
    return np.exp(-gamma * _sqdist(B, A)) @ w, float(best_a)


def trivial_baselines(X, Y, itr, ite, active_rows, ndim):
    """Context for a negative R^2: what do predictors with no learned map score?

    `train_mean` predicts the training set's mean style for everything;
    `nearest_neighbour` copies the style of the training clip whose embedding is
    closest in cosine. Both are scored exactly like the probe.
    """
    out = {}
    P = np.repeat(Y[itr].mean(0, keepdims=True), len(ite), axis=0)
    out["train_mean"] = evaluate(Y[itr], Y[ite], P, active_rows, ndim)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    nn = (Xn[ite] @ Xn[itr].T).argmax(1)
    out["nearest_neighbour_embedding"] = evaluate(Y[itr], Y[ite], Y[itr][nn],
                                                  active_rows, ndim)
    return out


def variance_decomposition(Y, groups):
    """How much of the target variance is just 'which base preset was this?'.

    A probe scoring well on a set whose variance is mostly between-preset is
    doing speaker identification, not style recovery.
    """
    tot = ((Y - Y.mean(0)) ** 2).sum()
    between = 0.0
    for g in np.unique(groups):
        m = groups == g
        between += m.sum() * ((Y[m].mean(0) - Y.mean(0)) ** 2).sum()
    return {"between_base_preset_share": float(between / tot),
            "within_base_preset_share": float(1.0 - between / tot)}


def dimensionality(Y):
    """Empirical intrinsic dimensionality of a style set. Censored by n samples."""
    Yc = Y - Y.mean(0)
    n = Yc.shape[0]
    s = np.linalg.svd(Yc, compute_uv=False)
    var = s ** 2
    var = var[var > var.max() * 1e-12]
    frac = var / var.sum()
    cum = np.cumsum(frac)
    return {
        "n_samples": int(n),
        "n_dims": int(Y.shape[1]),
        "effective_rank_participation_ratio": float(var.sum() ** 2 / (var ** 2).sum()),
        "n_pcs_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_pcs_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "n_pcs_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "var_in_top_9_pcs": float(cum[min(8, len(cum) - 1)]),
        "var_in_top_192_pcs": float(cum[min(191, len(cum) - 1)]),
        "note": "rank is capped at n_samples-1; treat effective rank as a lower bound",
    }


def evaluate(Ytr, Yte, Pte, active_rows, n_dim_per_row):
    ss_res, ss_tot_te = r2_parts(Yte, Pte, Yte.mean(0))
    _, ss_tot_tr = r2_parts(Yte, Pte, Ytr.mean(0))
    per_row_res = ss_res.reshape(len(active_rows), n_dim_per_row).sum(1)
    per_row_tot = ss_tot_te.reshape(len(active_rows), n_dim_per_row).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_row = 1.0 - per_row_res / per_row_tot
    # cosine similarity between predicted and true rows, a scale-free companion
    T = Yte.reshape(len(Yte), len(active_rows), n_dim_per_row)
    P = Pte.reshape(len(Pte), len(active_rows), n_dim_per_row)
    cos = (T * P).sum(-1) / (np.linalg.norm(T, axis=-1) * np.linalg.norm(P, axis=-1) + 1e-12)
    return {
        "r2_testmean_baseline": agg_r2(ss_res, ss_tot_te),
        "r2_trainmean_baseline": agg_r2(ss_res, ss_tot_tr),
        "per_row_r2": {int(r): round(float(v), 4) for r, v in zip(active_rows, per_row)},
        "per_row_r2_min": float(np.nanmin(per_row)),
        "per_row_r2_median": float(np.nanmedian(per_row)),
        "per_row_r2_max": float(np.nanmax(per_row)),
        "mean_row_cosine_pred_vs_true": float(cos.mean()),
    }


def shuffled_targets(Y, pool):
    """Hewitt & Liang control task: re-pair inputs and targets at random, within
    the same pool, so the control keeps the real target distribution and only
    the input-target correspondence is destroyed."""
    Yc = Y.copy()
    Yc[pool] = Y[RNG.permutation(pool)]
    return Yc


def run_probe(X, Y, itr, ite, active_rows, ndim, fitter):
    Ptr, alpha = fitter(X[itr], Y[itr], X[ite])
    real = evaluate(Y[itr], Y[ite], Ptr, active_rows, ndim)
    real["alpha"] = alpha
    # Hewitt & Liang control task: identical probe, targets randomly re-paired.
    Yc = shuffled_targets(Y, np.concatenate([itr, ite]))
    Pc, alpha_c = fitter(X[itr], Yc[itr], X[ite])
    ctrl = evaluate(Yc[itr], Yc[ite], Pc, active_rows, ndim)
    ctrl["alpha"] = alpha_c
    return {
        "real": real,
        "control_task": ctrl,
        "selectivity_r2": real["r2_testmean_baseline"] - ctrl["r2_testmean_baseline"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="store_true", help="also fit an RBF kernel-ridge probe")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "probe_report.json"))
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    S = np.load(os.path.join(OUT_DIR, "styles.npz"))
    E = np.load(os.path.join(OUT_DIR, "embeddings.npz"))
    ttl = S["ttl"]                       # (N, 50, 256)
    active_rows = [int(r) for r in S["active_rows"]]
    ndim = ttl.shape[-1]
    N = len(recs)
    assert ttl.shape[0] == N == E["ecapa"].shape[0]

    Y_active = ttl[:, active_rows, :].reshape(N, -1).astype(np.float64)
    Y_full = ttl.reshape(N, -1).astype(np.float64)
    cond = np.array([r["condition"] for r in recs])
    split = np.array([r["split"] for r in recs])
    base = np.array([r["base"] for r in recs])

    from helper import load_voice_style
    preset_ttl = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]).ttl[0]
                  for p in meta["presets_train"] + meta["presets_held_out"]}

    feature_sets = {"ecapa": E["ecapa"].astype(np.float64),
                    "spectral": E["spectral"].astype(np.float64)}
    fitters = [("linear_ridge", fit_ridge)]
    if args.kernel:
        fitters.append(("rbf_kernel_ridge", fit_kernel))

    report = {
        "experiment": "phase2b_style_probe",
        "generation_manifest": "generation_manifest.json",
        "n_total": N,
        "active_rows": active_rows,
        "presets_train": meta["presets_train"],
        "presets_held_out": meta["presets_held_out"],
        "target": f"{len(active_rows)} active rows x {ndim} = {len(active_rows)*ndim} values",
        "structural_note": (
            "a linear map from a 192-dim ECAPA embedding can span at most 192 output "
            "directions, so var_in_top_192_pcs is a hard ceiling on linear R^2"
        ),
        "conditions": {},
    }

    conditions = list(dict.fromkeys(cond.tolist())) + ["pooled"]
    for c in conditions:
        m = np.ones(N, bool) if c == "pooled" else (cond == c)
        idx = np.where(m)[0]
        itr_fam = idx[split[idx] == "train"]
        ite_fam = idx[split[idx] == "test"]
        # matched-size random split, for the inflation comparison
        perm = np.random.default_rng(11).permutation(idx)
        itr_rnd, ite_rnd = perm[: len(itr_fam)], perm[len(itr_fam):]

        eps = next((e for n, e in [(x["name"], x["eps"]) for x in meta["conditions"]]
                    if n == c), None)
        entry = {
            "eps": eps,
            "n": int(len(idx)),
            "n_train_family_disjoint": int(len(itr_fam)),
            "n_test_family_disjoint": int(len(ite_fam)),
            "dimensionality_active_style": dimensionality(Y_active[idx]),
        }
        if c != "preset_affine" and c != "pooled":
            entry["designed_intrinsic_dim"] = (
                f"{len(active_rows)} rows x {ndim - 1} tangent dims = "
                f"{len(active_rows) * (ndim - 1)} continuous dims, plus the discrete base preset"
            )
        elif c == "preset_affine":
            entry["designed_intrinsic_dim"] = (
                f"convex blends of {len(meta['presets_train'])} train presets -> at most "
                f"{len(meta['presets_train']) - 1} dims (test side: "
                f"{len(meta['presets_held_out']) - 1} dims)"
            )

        # within-family residual target: the perturbation itself, base preset removed
        Y_resid = None
        if eps is not None:
            Y_resid = Y_active[idx] - np.stack(
                [preset_ttl[b][active_rows].reshape(-1) for b in base[idx]]
            ).astype(np.float64)
            entry["dimensionality_residual_style"] = dimensionality(Y_resid)

        entry["trivial_baselines_family_disjoint"] = trivial_baselines(
            feature_sets["ecapa"], Y_active, itr_fam, ite_fam, active_rows, ndim)
        if eps is not None:
            entry["variance_decomposition"] = variance_decomposition(
                Y_active[idx], base[idx])
        entry["probes"] = {}
        for fname, fitter in fitters:
            for xname, X in feature_sets.items():
                key = f"{xname}/{fname}"
                res = {
                    "family_disjoint_split": run_probe(X, Y_active, itr_fam, ite_fam,
                                                       active_rows, ndim, fitter),
                    "random_split": run_probe(X, Y_active, itr_rnd, ite_rnd,
                                              active_rows, ndim, fitter),
                }
                # whole-tensor number, kept to show how the 26 dead rows inflate it
                Pfull, _ = fitter(X[itr_fam], Y_full[itr_fam], X[ite_fam])
                ssr, sst = r2_parts(Y_full[ite_fam], Pfull, Y_full[ite_fam].mean(0))
                res["whole_tensor_r2_family_disjoint"] = agg_r2(ssr, sst)
                if Y_resid is not None:
                    pos = {g: i for i, g in enumerate(idx)}
                    for tag, (gtr, gte) in {
                        "within_family_residual": (itr_fam, ite_fam),
                        # seen base presets: isolates recovery of the perturbation
                        # itself from the confound of an unseen speaker
                        "within_family_residual_random_split": (itr_rnd, ite_rnd),
                    }.items():
                        rtr = np.array([pos[i] for i in gtr])
                        rte = np.array([pos[i] for i in gte])
                        Pr, _ = fitter(X[gtr], Y_resid[rtr], X[gte])
                        ev = evaluate(Y_resid[rtr], Y_resid[rte], Pr, active_rows, ndim)
                        Yrc = shuffled_targets(Y_resid, np.arange(len(Y_resid)))
                        Pc, _ = fitter(X[gtr], Yrc[rtr], X[gte])
                        evc = evaluate(Yrc[rtr], Yrc[rte], Pc, active_rows, ndim)
                        res[tag] = {
                            "real": ev, "control_task": evc,
                            "selectivity_r2": ev["r2_testmean_baseline"]
                            - evc["r2_testmean_baseline"],
                        }
                entry["probes"][key] = res
                print(f"[{c}] {key}: family-disjoint R2="
                      f"{res['family_disjoint_split']['real']['r2_testmean_baseline']:.4f} "
                      f"(control {res['family_disjoint_split']['control_task']['r2_testmean_baseline']:.4f}) "
                      f"random-split R2={res['random_split']['real']['r2_testmean_baseline']:.4f} "
                      f"whole-tensor={res['whole_tensor_r2_family_disjoint']:.4f}"
                      + (f" residual R2={res['within_family_residual']['real']['r2_testmean_baseline']:.4f}"
                         if Y_resid is not None else ""), flush=True)
        report["conditions"][c] = entry

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
