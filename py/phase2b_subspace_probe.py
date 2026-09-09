"""Phase 2b subspace, stage 3: audio readout capacity vs subspace dimension K.

See ../new-plan.md and `phase2b_generate_subspace.py`'s docstring for the
design: style_ttl perturbations are confined to a fixed, nested K-dim linear
subspace (K in {4, 16, 64} by default) of the tangent space, precisely so a
ridge probe has the statistical power to express the target -- unlike
phase2b's isotropic 6,120-dim perturbation with only 240 training samples.

For each K this fits a linear ridge probe WavLM features -> c_realized (the
REALIZED subspace coefficients after unit-row renormalization, not c_drawn,
since renormalization can move a sample off its drawn coefficients) and
reports, next to the achieved R^2:

  * the oracle ceiling for this K -- computed exactly as
    `phase2a_ceiling_audit.py` computes it (orthogonal projection of the test
    targets onto row_space(train targets)), reused by import. With
    n_train >> K this should sit close to 1.0; if it does not, the probe
    result below it is not trustworthy evidence of anything.
  * a Hewitt & Liang shuffled-target control (targets randomly re-paired
    across samples, refit, score) -- reused from `phase2b_probe`. Should be
    near 0.
  * a loudness control -- R^2 of predicting the coefficients from `rms_pre`
    alone (a 1-dim feature, from phase2b_subspace_embed.py's output). If this
    is materially above 0, loudness is leaking into the "real" result.

Ridge fitting (`phase2b_probe.fit_ridge`), the R^2 bookkeeping
(`phase2b_probe.r2_parts` / `agg_r2`), the shuffled-target control
(`phase2b_probe.shuffled_targets`), the oracle-ceiling machinery
(`phase2a_ceiling_audit.numerical_rank` / `oracle_projection`), and the
1-based-layer WavLM representation builder (`phase2b_wavlm.rep_from`) are all
imported and reused rather than reimplemented, per this project's convention
of keeping one source of truth for shared statistical machinery.

Default representation: WavLM layers 3, 4, 5 (1-based, same convention as
phase2b_wavlm.rep_from), mean and std concatenated -> 3*1024*2 = 6144 dims.
Override with --layers.

Usage (from py/):
    python3 phase2b_subspace_probe.py [--layers 3,4,5] [--k 16]
    python3 phase2b_subspace_probe.py --manifest OTHER/manifest.json \
        --subspace OTHER/subspace.npz --feats OTHER/wavlm_feats.npz \
        --out OTHER/subspace_probe.json  # smoke test
"""

import argparse
import json
import os

import numpy as np

from phase2a_ceiling_audit import numerical_rank, oracle_projection
from phase2b_probe import RNG, agg_r2, fit_ridge, r2_parts, shuffled_targets
from phase2b_wavlm import rep_from

SUBSPACE_DIR = "results/phase2b_subspace"
OUT_PATH = "results/phase2a/subspace_probe.json"


def parse_layers(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def pooled_r2(Ytr, Yte, Ypred):
    ss_res, ss_tot = r2_parts(Yte, Ypred, Yte.mean(0))
    return agg_r2(ss_res, ss_tot)


def per_component_r2(Yte, Ypred):
    ss_res, ss_tot = r2_parts(Yte, Ypred, Yte.mean(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return r2


def mean_cosine(Yte, Ypred):
    num = (Yte * Ypred).sum(-1)
    den = np.linalg.norm(Yte, axis=-1) * np.linalg.norm(Ypred, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = num / den
    return float(np.nanmean(cos))


def run_one_k(K, recs_k, feats, subspace, layers):
    """recs_k: manifest records with this K, in manifest order."""
    key_to_row = {(int(k), int(i)): n for n, (k, i) in enumerate(zip(feats["K"], feats["idx"]))}

    kept = [r for r in recs_k if (K, int(r["idx"])) in key_to_row]
    missing = len(recs_k) - len(kept)
    feat_rows = np.array([key_to_row[(K, int(r["idx"]))] for r in kept])

    mean_sel = feats["mean"][feat_rows]  # (n, 24, 1024)
    std_sel = feats["std"][feat_rows]
    rms_pre = feats["rms_pre"][feat_rows].astype(np.float64)

    X = rep_from(mean_sel, std_sel, layers, "meanstd")  # (n, len(layers)*2048)

    c_realized = subspace[f"c_realized_K{K}"]  # (n_k_total, K), indexed by local idx
    Y = np.stack([c_realized[int(r["idx"])] for r in kept]).astype(np.float64)

    split = np.array([r["split"] for r in kept])
    itr = np.where(split == "train")[0]
    ite = np.where(split == "test")[0]
    assert len(itr) > 0 and len(ite) > 0, f"K={K}: empty split (train={len(itr)}, test={len(ite)})"

    Xtr, Xte = X[itr], X[ite]
    Ytr, Yte = Y[itr], Y[ite]

    # ---- achieved: ridge probe, WavLM -> c_realized ----
    Ypred, alpha = fit_ridge(Xtr, Ytr, Xte)
    comp_r2 = per_component_r2(Yte, Ypred)
    achieved_pooled = pooled_r2(Ytr, Yte, Ypred)
    cos = mean_cosine(Yte, Ypred)

    # ---- oracle ceiling: best any linear combo of training rows can do ----
    rank_eps, rank_loose, s = numerical_rank(Ytr)
    P_oracle, _ = oracle_projection(Ytr, Yte, rank_eps)
    oracle_r2 = pooled_r2(Ytr, Yte, P_oracle)

    # ---- Hewitt & Liang shuffled-target control ----
    Yc = shuffled_targets(Y, np.arange(len(Y)))
    Ypred_c, alpha_c = fit_ridge(Xtr, Yc[itr], Xte)
    shuffled_r2 = pooled_r2(Yc[itr], Yc[ite], Ypred_c)

    # ---- loudness control: rms_pre alone -> c_realized ----
    Xrms_tr = rms_pre[itr].reshape(-1, 1)
    Xrms_te = rms_pre[ite].reshape(-1, 1)
    Ypred_rms, alpha_rms = fit_ridge(Xrms_tr, Ytr, Xrms_te)
    loudness_r2 = pooled_r2(Ytr, Yte, Ypred_rms)

    return {
        "K": K,
        "n_matched": len(kept),
        "n_missing_from_feats": missing,
        "n_train": int(len(itr)),
        "n_test": int(len(ite)),
        "layers_1based": layers,
        "feature_dim": int(X.shape[1]),
        "ridge_alpha": alpha,
        "per_component_r2": [float(v) for v in comp_r2],
        "per_component_r2_mean": float(np.nanmean(comp_r2)),
        "per_component_r2_median": float(np.nanmedian(comp_r2)),
        "per_component_r2_min": float(np.nanmin(comp_r2)),
        "per_component_r2_max": float(np.nanmax(comp_r2)),
        "pooled_r2_testmean_baseline": achieved_pooled,
        "mean_cosine_pred_vs_true": cos,
        "oracle_ceiling": {
            "numerical_rank_eps": rank_eps,
            "numerical_rank_loose_1e-10": rank_loose,
            "top_5_singvals": [float(x) for x in s[:5]],
            "r2_testmean_baseline": oracle_r2,
        },
        "shuffled_target_control_r2": shuffled_r2,
        "loudness_control_r2": loudness_r2,
        "loudness_control_alpha": alpha_rms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=SUBSPACE_DIR, help="root dir for default manifest/subspace/feats paths")
    ap.add_argument("--manifest", default=None, help="default: <out-dir>/manifest.json")
    ap.add_argument("--subspace", default=None, help="default: <out-dir>/subspace.npz")
    ap.add_argument("--feats", default=None, help="default: <out-dir>/wavlm_feats.npz")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--layers", type=parse_layers, default=[3, 4, 5],
                     help="1-based WavLM transformer layer indices, comma-separated")
    ap.add_argument("--k", type=int, default=None, help="only run this K (default: every K present in the manifest)")
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.json")
    subspace_path = args.subspace or os.path.join(args.out_dir, "subspace.npz")
    feats_path = args.feats or os.path.join(args.out_dir, "wavlm_feats.npz")

    with open(manifest_path) as f:
        meta = json.load(f)
    records = meta["records"]
    subspace = np.load(subspace_path)
    feats = np.load(feats_path)

    k_list = [args.k] if args.k is not None else sorted({int(r["K"]) for r in records})

    report = {
        "experiment": "phase2b_subspace_probe",
        "manifest": manifest_path,
        "subspace": subspace_path,
        "feats": feats_path,
        "layers_1based": args.layers,
        "target": "c_realized (realized subspace coefficients), not c_drawn",
        "k_list": k_list,
        "results": {},
    }

    rows = []
    for K in k_list:
        recs_k = [r for r in records if int(r["K"]) == K]
        res = run_one_k(K, recs_k, feats, subspace, args.layers)
        report["results"][str(K)] = res
        rows.append(res)
        print(f"K={K:<4} n_train={res['n_train']:<5} "
              f"oracle_ceiling_R2={res['oracle_ceiling']['r2_testmean_baseline']:.4f} "
              f"achieved_mean_R2={res['per_component_r2_mean']:.4f} "
              f"achieved_median_R2={res['per_component_r2_median']:.4f} "
              f"shuffled_control={res['shuffled_target_control_r2']:.4f} "
              f"loudness_control={res['loudness_control_r2']:.4f} "
              f"mean_cosine={res['mean_cosine_pred_vs_true']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    hdr = (f"{'K':>5}{'n_train':>9}{'oracle_R2':>12}{'achieved_mean_R2':>18}"
           f"{'median_R2':>11}{'shuffled':>10}{'loudness':>10}{'cosine':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for res in rows:
        print(f"{res['K']:>5}{res['n_train']:>9}"
              f"{res['oracle_ceiling']['r2_testmean_baseline']:>12.4f}"
              f"{res['per_component_r2_mean']:>18.4f}"
              f"{res['per_component_r2_median']:>11.4f}"
              f"{res['shuffled_target_control_r2']:>10.4f}"
              f"{res['loudness_control_r2']:>10.4f}"
              f"{res['mean_cosine_pred_vs_true']:>9.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
