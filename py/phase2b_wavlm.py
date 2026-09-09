"""Phase 2b, stage 3c: WavLM-large probe -> style_ttl, against the ECAPA result.

new-plan.md Phase 2b names "ECAPA/WavLM embedding -> style_ttl" and only ECAPA
was run. ECAPA came back negative on unseen voices (family-disjoint R^2 0.192
for preset blends down to 0.007 at eps 0.80). That test was weak by
construction: ECAPA is 192-dimensional, so a linear map out of it can span at
most 192 of the 24 x 256 = 6,120 target directions. WavLM-large is 1024-dim per
frame per layer -> 2048 with mean+std pooling, more with multi-layer
concatenation, so it is a genuinely fairer test of the same hypothesis, not
merely a different one.

Everything here mirrors `phase2b_probe.py` exactly -- same split logic, same
Hewitt & Liang control task, same per-row reporting, same trivial baselines --
by importing its functions rather than reimplementing them, so the numbers are
directly comparable. ECAPA is refitted inside this script on the identical
indices so the side-by-side is apples to apples.

Reported, per representation and condition:

  * the structural ceiling at min(feature_dim, n_train): ridge cannot span more
    output directions than either the input dimension or the training set size,
    and with 240 training clips per condition the sample count is what binds for
    WavLM. Both numbers are reported so the fairness of the test is visible.
  * family-disjoint R^2 against the TRAIN-mean baseline (M5/F4/F5 held out) --
    the headline, and the number the ECAPA report quotes;
  * random-split R^2 -- reported alongside and labelled as the misleading one;
  * control-task selectivity (shuffled targets), real minus control;
  * per-active-row R^2 and the inflated whole-tensor number;
  * the within-family residual (base preset subtracted), which is the part that
    is genuinely high-dimensional rather than a 7-way preset identification;
  * trivial baselines: constant train-mean, and 1-NN in the representation.

Usage (from py/):
    python3 phase2b_wavlm.py --stage sanity     # does WavLM see these voices?
    python3 phase2b_wavlm.py --stage sweep      # per-layer R^2, all 24 layers
    python3 phase2b_wavlm.py --stage probe      # full protocol + ECAPA side-by-side
"""

import argparse
import json
import os

import numpy as np

import phase2b_probe as P

OUT_DIR = "results/phase2b"
VOICE_STYLE_DIR = "assets/voice_styles"
SWEEP_CONDITIONS = ["preset_affine", "eps0.05", "eps0.20"]


# --------------------------------------------------------------------------
# representations, built off the cached per-layer (mean, std) pooled features
# --------------------------------------------------------------------------

def load_wavlm(path):
    z = np.load(path)
    return z["mean"], z["std"]  # (N, 24, 1024) each


def rep_from(mean, std, layers, pooling):
    """layers: 1-based transformer layer indices. pooling: 'mean' | 'meanstd'."""
    li = [l - 1 for l in layers]
    parts = [mean[:, li, :].reshape(len(mean), -1)]
    if pooling == "meanstd":
        parts.append(std[:, li, :].reshape(len(std), -1))
    return np.concatenate(parts, 1).astype(np.float64)


def var_in_top_k(Y, k):
    """Fraction of the target's variance inside its top-k principal directions.

    A linear probe from a d-dim input fitted on n training samples produces a
    weight matrix of rank at most min(d, n), so this evaluated at
    k = min(d, n_train) is a hard ceiling on its achievable R^2.
    """
    Yc = Y - Y.mean(0)
    s = np.linalg.svd(Yc, compute_uv=False)
    var = s ** 2
    var = var[var > var.max() * 1e-12]
    cum = np.cumsum(var / var.sum())
    return float(cum[min(k - 1, len(cum) - 1)])


# --------------------------------------------------------------------------
# stage 1: is WavLM working at all on this engine's synthetic audio?
# --------------------------------------------------------------------------

def stage_sanity(args):
    """Mirror of the ECAPA control: 1-NN preset ID over 10 presets x 8 sentences.

    ECAPA scored 1.000 there, which is what ruled out domain mismatch as the
    explanation for its null. WavLM has to clear the same bar before its own
    null means anything.
    """
    z = np.load(os.path.join(OUT_DIR, "wavlm_reference_feats.npz"))
    mean, std, lab = z["mean"], z["std"], z["labels"]
    out = {
        "n_clips": int(len(lab)), "chance": 0.1,
        "note": ("raw cosine is not a meaningful metric on WavLM features -- they carry "
                 "a large shared offset, so every pair sits near cos 0.98. The z-scored "
                 "variant (per-dimension mean/std removed, which is exactly what the "
                 "ridge probe's StandardScaler does) is the fair analogue of the ECAPA "
                 "1-NN control, which scored 1.000."),
        "per_layer": {},
    }
    for layer in range(1, 25):
        for pooling in ("mean", "meanstd"):
            X0 = rep_from(mean, std, [layer], pooling)
            for scaling, X in (("raw", X0),
                               ("zscored", (X0 - X0.mean(0)) / (X0.std(0) + 1e-8))):
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                S = Xn @ Xn.T
                same = lab[:, None] == lab[None, :]
                off = ~np.eye(len(lab), dtype=bool)
                nn = np.argmax(np.where(off, S, -np.inf), axis=1)
                out["per_layer"].setdefault(f"L{layer}", {})[f"{pooling}_{scaling}"] = {
                    "same_preset_cosine_mean": float(S[same & off].mean()),
                    "diff_preset_cosine_mean": float(S[~same].mean()),
                    "one_nn_preset_accuracy": float((lab[nn] == lab).mean()),
                }
        r = out["per_layer"][f"L{layer}"]
        print(f"  L{layer:<2} 1-NN preset acc  raw: mean {r['mean_raw']['one_nn_preset_accuracy']:.3f} "
              f"meanstd {r['meanstd_raw']['one_nn_preset_accuracy']:.3f} | "
              f"z-scored: mean {r['mean_zscored']['one_nn_preset_accuracy']:.3f} "
              f"meanstd {r['meanstd_zscored']['one_nn_preset_accuracy']:.3f}  "
              f"(z same-cos {r['meanstd_zscored']['same_preset_cosine_mean']:+.3f} / "
              f"diff {r['meanstd_zscored']['diff_preset_cosine_mean']:+.3f})", flush=True)
    write(out, args.out or os.path.join(OUT_DIR, "wavlm_sanity.json"))


# --------------------------------------------------------------------------
# shared data loading for the probe stages
# --------------------------------------------------------------------------

def load_all():
    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    S = np.load(os.path.join(OUT_DIR, "styles.npz"))
    ttl = S["ttl"]
    active_rows = [int(r) for r in S["active_rows"]]
    ndim = ttl.shape[-1]
    N = len(recs)
    mean, std = load_wavlm(os.path.join(OUT_DIR, "wavlm_feats.npz"))
    assert len(mean) == N, (len(mean), N)
    d = {
        "meta": meta, "recs": recs, "ttl": ttl, "active_rows": active_rows,
        "ndim": ndim, "N": N, "wavlm_mean": mean, "wavlm_std": std,
        "Y_active": ttl[:, active_rows, :].reshape(N, -1).astype(np.float64),
        "Y_full": ttl.reshape(N, -1).astype(np.float64),
        "cond": np.array([r["condition"] for r in recs]),
        "split": np.array([r["split"] for r in recs]),
        "base": np.array([r["base"] for r in recs]),
        "ecapa": np.load(os.path.join(OUT_DIR, "embeddings.npz"))["ecapa"].astype(np.float64),
    }
    return d


def split_indices(d, c):
    m = np.ones(d["N"], bool) if c == "pooled" else (d["cond"] == c)
    idx = np.where(m)[0]
    itr_fam = idx[d["split"][idx] == "train"]
    ite_fam = idx[d["split"][idx] == "test"]
    perm = np.random.default_rng(11).permutation(idx)  # identical to phase2b_probe
    return idx, itr_fam, ite_fam, perm[: len(itr_fam)], perm[len(itr_fam):]


# --------------------------------------------------------------------------
# stage 2: per-layer sweep, so the layer choice is visible rather than hidden
# --------------------------------------------------------------------------

def stage_sweep(args):
    d = load_all()
    out = {"conditions": SWEEP_CONDITIONS, "pooling": "meanstd (2048-dim)",
           "split": "family-disjoint (M5/F4/F5 held out)", "per_layer": {}}
    for layer in range(1, 25):
        X = rep_from(d["wavlm_mean"], d["wavlm_std"], [layer], "meanstd")
        row = {}
        for c in SWEEP_CONDITIONS:
            idx, itr, ite, _, _ = split_indices(d, c)
            Pte, alpha = P.fit_ridge(X[itr], d["Y_active"][itr], X[ite])
            ev = P.evaluate(d["Y_active"][itr], d["Y_active"][ite], Pte,
                            d["active_rows"], d["ndim"])
            row[c] = {"r2": ev["r2_trainmean_baseline"],
                      "r2_testmean_baseline": ev["r2_testmean_baseline"],
                      "mean_row_cosine": ev["mean_row_cosine_pred_vs_true"],
                      "alpha": alpha}
        out["per_layer"][f"L{layer}"] = row
        print("  L%-2d  " % layer + "  ".join(
            f"{c}={row[c]['r2']:+.4f}" for c in SWEEP_CONDITIONS), flush=True)
    best = max(out["per_layer"],
               key=lambda k: np.mean([out["per_layer"][k][c]["r2"]
                                      for c in SWEEP_CONDITIONS]))
    out["best_layer_by_mean_r2"] = best
    print(f"\nBest layer by mean R^2 over {SWEEP_CONDITIONS}: {best}")
    write(out, args.out or os.path.join(OUT_DIR, "wavlm_layer_sweep.json"))


# --------------------------------------------------------------------------
# stage 3: the full protocol, mirroring phase2b_probe.py
# --------------------------------------------------------------------------

def build_reps(d, spec):
    reps = {}
    for name in spec:
        if name == "ecapa":
            reps["ecapa"] = d["ecapa"]
        elif name.startswith("wavlm:"):
            _, layers, pooling = name.split(":")
            ls = [int(x) for x in layers.split(",")]
            key = f"wavlm_L{'+'.join(map(str, ls))}_{pooling}"
            reps[key] = rep_from(d["wavlm_mean"], d["wavlm_std"], ls, pooling)
        else:
            raise SystemExit(f"unknown representation {name}")
    return reps


def stage_probe(args):
    d = load_all()
    meta = d["meta"]
    reps = build_reps(d, args.reps)
    for k, v in reps.items():
        print(f"representation {k}: {v.shape[1]} dims", flush=True)

    from helper import load_voice_style
    preset_ttl = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]).ttl[0]
                  for p in meta["presets_train"] + meta["presets_held_out"]}

    report = {
        "experiment": "phase2b_wavlm_probe",
        "compare_against": "probe_report.json (ECAPA, identical protocol)",
        "n_total": d["N"],
        "active_rows": d["active_rows"],
        "presets_train": meta["presets_train"],
        "presets_held_out": meta["presets_held_out"],
        "target": f"{len(d['active_rows'])} active rows x {d['ndim']} = "
                  f"{len(d['active_rows']) * d['ndim']} values",
        "representation_dims": {k: int(v.shape[1]) for k, v in reps.items()},
        "structural_note": (
            "ridge from a d-dim input fitted on n samples has rank <= min(d, n), so "
            "the achievable R^2 is capped by the target variance inside its top "
            "min(d, n_train) principal directions. Both the top-d and the "
            "top-min(d,n_train) ceilings are reported."
        ),
        "conditions": {},
    }

    conditions = list(dict.fromkeys(d["cond"].tolist())) + ["pooled"]
    if args.conditions:
        conditions = args.conditions
    for c in conditions:
        idx, itr_fam, ite_fam, itr_rnd, ite_rnd = split_indices(d, c)
        eps = next((e for n, e in [(x["name"], x["eps"]) for x in meta["conditions"]]
                    if n == c), None)
        Ya = d["Y_active"]
        entry = {
            "eps": eps, "n": int(len(idx)),
            "n_train_family_disjoint": int(len(itr_fam)),
            "n_test_family_disjoint": int(len(ite_fam)),
            "dimensionality_active_style": P.dimensionality(Ya[idx]),
        }
        Y_resid = None
        if eps is not None:
            Y_resid = Ya[idx] - np.stack(
                [preset_ttl[b][d["active_rows"]].reshape(-1) for b in d["base"][idx]]
            ).astype(np.float64)
            entry["dimensionality_residual_style"] = P.dimensionality(Y_resid)
            entry["variance_decomposition"] = P.variance_decomposition(Ya[idx], d["base"][idx])

        # ceilings, per representation dimensionality
        entry["structural_ceilings"] = {}
        for k, X in reps.items():
            dd, ntr = int(X.shape[1]), int(len(itr_fam))
            ce = {"dim": dd, "n_train": ntr,
                  "ceiling_top_d_pcs": var_in_top_k(Ya[idx], dd),
                  "ceiling_top_min_d_ntrain_pcs": var_in_top_k(Ya[idx], min(dd, ntr))}
            if Y_resid is not None:
                ce["residual_ceiling_top_min_d_ntrain_pcs"] = var_in_top_k(
                    Y_resid, min(dd, ntr))
            entry["structural_ceilings"][k] = ce

        entry["probes"] = {}
        for k, X in reps.items():
            tb = P.trivial_baselines(X, Ya, itr_fam, ite_fam, d["active_rows"], d["ndim"])
            # WavLM features carry a large shared offset, so raw cosine 1-NN (the
            # ECAPA-comparable form, kept above) understates them. Add the z-scored
            # form, standardised on the training split exactly as fit_ridge does.
            mu, sd = X[itr_fam].mean(0), X[itr_fam].std(0) + 1e-8
            Xz = (X - mu) / sd
            Xz /= np.linalg.norm(Xz, axis=1, keepdims=True).clip(min=1e-12)
            nnz = (Xz[ite_fam] @ Xz[itr_fam].T).argmax(1)
            tb["nearest_neighbour_embedding_zscored"] = P.evaluate(
                Ya[itr_fam], Ya[ite_fam], Ya[itr_fam][nnz], d["active_rows"], d["ndim"])
            res = {
                "trivial_baselines_family_disjoint": tb,
                "family_disjoint_split": P.run_probe(X, Ya, itr_fam, ite_fam,
                                                     d["active_rows"], d["ndim"], P.fit_ridge),
                "random_split": P.run_probe(X, Ya, itr_rnd, ite_rnd,
                                            d["active_rows"], d["ndim"], P.fit_ridge),
            }
            for tag in ("family_disjoint_split", "random_split"):
                res[tag]["selectivity_r2_trainmean"] = (
                    res[tag]["real"]["r2_trainmean_baseline"]
                    - res[tag]["control_task"]["r2_trainmean_baseline"])
            Pfull, _ = P.fit_ridge(X[itr_fam], d["Y_full"][itr_fam], X[ite_fam])
            ssr, sst = P.r2_parts(d["Y_full"][ite_fam], Pfull, d["Y_full"][ite_fam].mean(0))
            res["whole_tensor_r2_family_disjoint"] = P.agg_r2(ssr, sst)
            if Y_resid is not None:
                pos = {g: i for i, g in enumerate(idx)}
                for tag, (gtr, gte) in {
                    "within_family_residual": (itr_fam, ite_fam),
                    "within_family_residual_random_split": (itr_rnd, ite_rnd),
                }.items():
                    rtr = np.array([pos[i] for i in gtr])
                    rte = np.array([pos[i] for i in gte])
                    Pr, _ = P.fit_ridge(X[gtr], Y_resid[rtr], X[gte])
                    ev = P.evaluate(Y_resid[rtr], Y_resid[rte], Pr, d["active_rows"], d["ndim"])
                    Yrc = P.shuffled_targets(Y_resid, np.arange(len(Y_resid)))
                    Pc, _ = P.fit_ridge(X[gtr], Yrc[rtr], X[gte])
                    evc = P.evaluate(Yrc[rtr], Yrc[rte], Pc, d["active_rows"], d["ndim"])
                    res[tag] = {"real": ev, "control_task": evc,
                                "selectivity_r2": ev["r2_testmean_baseline"]
                                - evc["r2_testmean_baseline"],
                                "selectivity_r2_trainmean": ev["r2_trainmean_baseline"]
                                - evc["r2_trainmean_baseline"]}
            entry["probes"][k] = res
            fd = res["family_disjoint_split"]
            print(f"[{c}] {k}: family-disjoint R2={fd['real']['r2_trainmean_baseline']:+.4f} "
                  f"(ctrl {fd['control_task']['r2_trainmean_baseline']:+.4f}, "
                  f"sel {fd['selectivity_r2_trainmean']:+.4f}) "
                  f"random={res['random_split']['real']['r2_trainmean_baseline']:+.4f} "
                  f"ceiling={entry['structural_ceilings'][k]['ceiling_top_min_d_ntrain_pcs']:.3f} "
                  f"1nn={tb['nearest_neighbour_embedding_zscored']['r2_trainmean_baseline']:+.4f}"
                  + (f" resid={res['within_family_residual']['real']['r2_trainmean_baseline']:+.4f}"
                     if Y_resid is not None else ""), flush=True)
        report["conditions"][c] = entry

    write(report, args.out or os.path.join(OUT_DIR, "wavlm_report.json"))


# --------------------------------------------------------------------------
# stage 4: the side-by-side table, WavLM vs ECAPA on identical splits
# --------------------------------------------------------------------------

def stage_summary(args):
    W = json.load(open(os.path.join(OUT_DIR, "wavlm_report.json")))
    conds = list(W["conditions"])
    reps = list(W["representation_dims"])
    print("\nHEADLINE -- family-disjoint split (M5/F4/F5 unseen), R^2 vs train-mean baseline")
    print("all numbers from wavlm_report.json; ECAPA refitted on identical indices\n")
    head = f"{'condition':<14}" + "".join(f"{r.replace('wavlm_','w'):>26}" for r in reps)
    print(head)
    print(f"{'':14}" + "".join(f"{'R2 (ceil@min(d,n)) sel':>26}" for r in reps))
    for c in conds:
        e = W["conditions"][c]
        line = f"{c:<14}"
        for r in reps:
            fd = e["probes"][r]["family_disjoint_split"]
            ce = e["structural_ceilings"][r]["ceiling_top_min_d_ntrain_pcs"]
            line += f"{fd['real']['r2_trainmean_baseline']:>10.4f} ({ce:.3f}) {fd['selectivity_r2_trainmean']:>+7.4f}"
        print(line)

    print("\nRANDOM SPLIT (the misleading one -- base preset is in the training set)")
    for c in conds:
        e = W["conditions"][c]
        print(f"{c:<14}" + "".join(
            f"{e['probes'][r]['random_split']['real']['r2_trainmean_baseline']:>26.4f}" for r in reps))

    print("\nWITHIN-FAMILY RESIDUAL (base preset subtracted; family-disjoint split)")
    for c in conds:
        e = W["conditions"][c]
        if "within_family_residual" not in e["probes"][reps[0]]:
            continue
        line = f"{c:<14}"
        for r in reps:
            rr = e["probes"][r]["within_family_residual"]
            ce = e["structural_ceilings"][r].get("residual_ceiling_top_min_d_ntrain_pcs", float("nan"))
            line += f"{rr['real']['r2_trainmean_baseline']:>10.4f} ({ce:.3f}) {rr['selectivity_r2_trainmean']:>+7.4f}"
        print(line)

    print("\nTRIVIAL BASELINES, family-disjoint (R^2 vs train-mean baseline)")
    print(f"{'condition':<14}{'train_mean':>12}" + "".join(f"{'1nn:'+r[:14]:>22}" for r in reps))
    for c in conds:
        tb = W["conditions"][c]["probes"][reps[0]]["trivial_baselines_family_disjoint"]
        line = f"{c:<14}{tb['train_mean']['r2_trainmean_baseline']:>12.4f}"
        for r in reps:
            t = W["conditions"][c]["probes"][r]["trivial_baselines_family_disjoint"]
            line += f"{t['nearest_neighbour_embedding_zscored']['r2_trainmean_baseline']:>22.4f}"
        print(line)

    print("\nPER-ROW R^2 over the 24 active rows, family-disjoint (min / median / max)")
    for c in conds:
        e = W["conditions"][c]
        line = f"{c:<14}"
        for r in reps:
            v = e["probes"][r]["family_disjoint_split"]["real"]
            line += f"  {r[:10]}: {v['per_row_r2_min']:+.3f}/{v['per_row_r2_median']:+.3f}/{v['per_row_r2_max']:+.3f}"
        print(line)

    print("\nrepresentation dims:", W["representation_dims"])


def write(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "sweep", "probe", "summary"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--reps", nargs="*", default=[
        "ecapa", "wavlm:4:meanstd", "wavlm:12:meanstd", "wavlm:24:meanstd",
        "wavlm:4,12,24:meanstd",
    ])
    args = ap.parse_args()
    {"sanity": stage_sanity, "sweep": stage_sweep, "probe": stage_probe,
     "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
