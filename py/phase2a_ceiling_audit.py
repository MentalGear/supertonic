"""Phase 2a ceiling audit: is phase2b's within-family residual R^2 ~ 0 a
consequence of the audio, or a mathematical consequence of only having
n_train=240 training targets to span a 6144-dim output space?

Ridge regression (and RBF kernel ridge in its dual form) predicts the test
set as a linear combination of TRAINING TARGET ROWS. Every such prediction
therefore lies in row_space(Y_train), whose rank is at most n_train. This
script computes the ORACLE ceiling -- the best possible R^2 any linear
combination of Y_train's rows could achieve on Y_test -- by orthogonally
projecting Y_test onto that row space. This is a hard upper bound on any
probe (linear or kernel-ridge-in-that-span) trained on those samples,
independent of what audio features feed it.

Reuses phase2b_probe.evaluate() / r2_parts() / agg_r2() verbatim so the
ceiling is measured on the exact same metric as the reported result. Does
not modify phase2b_probe.py.

Usage (from py/):
    python3 phase2a_ceiling_audit.py
"""

import json
import os

import numpy as np

from phase2b_probe import evaluate, r2_parts, agg_r2, VOICE_STYLE_DIR, OUT_DIR as PHASE2B_DIR

OUT_DIR = "results/phase2a"
OUT_PATH = os.path.join(OUT_DIR, "ceiling_audit.json")


def numerical_rank(M, rtol=1e-10):
    """SVD-based numerical rank, plus the raw singular values for inspection."""
    s = np.linalg.svd(M, compute_uv=False)
    tol = s.max() * max(M.shape) * np.finfo(M.dtype).eps if s.max() > 0 else 0.0
    # also report a looser, more conservative threshold as a cross-check
    rank_eps = int((s > tol).sum())
    rank_loose = int((s > s.max() * rtol).sum()) if s.max() > 0 else 0
    return rank_eps, rank_loose, s


def oracle_projection(Y_train, Y_test, rank):
    """Orthonormal basis Q of row_space(Y_train) (top `rank` right singular
    vectors), and the projection of Y_test onto it."""
    _, _, Vt = np.linalg.svd(Y_train, full_matrices=False)
    Q = Vt[:rank].T  # (D, rank), orthonormal columns
    P = (Y_test @ Q) @ Q.T
    return P, Q


def captured_fraction(Y_test, Q):
    proj = (Y_test @ Q) @ Q.T
    num = float((proj ** 2).sum())
    den = float((Y_test ** 2).sum())
    return num / den if den > 0 else float("nan")


def pull_achieved(report, cond, tag="within_family_residual"):
    """Best (max) achieved real r2_testmean_baseline for `tag`, across every
    representation/probe key in a phase2b-style report, for one condition."""
    entry = report["conditions"].get(cond)
    if entry is None:
        return None, None
    best_key, best_val = None, -np.inf
    for key, res in entry["probes"].items():
        if tag in res:
            v = res[tag]["real"]["r2_testmean_baseline"]
            if v > best_val:
                best_key, best_val = key, v
    if best_key is None:
        return None, None
    return best_key, float(best_val)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(PHASE2B_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    S = np.load(os.path.join(PHASE2B_DIR, "styles.npz"))
    ttl = S["ttl"]
    active_rows = [int(r) for r in S["active_rows"]]
    ndim = ttl.shape[-1]
    N = len(recs)
    assert ttl.shape[0] == N

    Y_active = ttl[:, active_rows, :].reshape(N, -1).astype(np.float64)
    cond = np.array([r["condition"] for r in recs])
    split = np.array([r["split"] for r in recs])
    base = np.array([r["base"] for r in recs])

    from helper import load_voice_style
    preset_ttl = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]).ttl[0]
                  for p in meta["presets_train"] + meta["presets_held_out"]}

    achieved_reports = {}
    for name, fname in [("ecapa_report", "probe_report.json"), ("wavlm_report", "wavlm_report.json")]:
        path = os.path.join(PHASE2B_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                achieved_reports[name] = json.load(f)

    conditions = list(dict.fromkeys(cond.tolist()))
    results = {"target_dims": Y_active.shape[1], "active_rows": active_rows, "conditions": {}}
    table_rows = []

    for c in conditions:
        m = cond == c
        idx = np.where(m)[0]
        itr_fam = idx[split[idx] == "train"]
        ite_fam = idx[split[idx] == "test"]
        eps = next((x["eps"] for x in meta["conditions"] if x["name"] == c), None)

        entry = {"eps": eps, "n_train": int(len(itr_fam)), "n_test": int(len(ite_fam))}

        # ---- full-target (Y_active) oracle ----
        Ytr_full, Yte_full = Y_active[itr_fam], Y_active[ite_fam]
        rank_full_eps, rank_full_loose, s_full = numerical_rank(Ytr_full)
        P_full, Q_full = oracle_projection(Ytr_full, Yte_full, rank_full_eps)
        oracle_full_eval = evaluate(Ytr_full, Yte_full, P_full, active_rows, ndim)
        entry["full_target"] = {
            "numerical_rank_eps": rank_full_eps,
            "numerical_rank_loose_1e-10": rank_full_loose,
            "top_5_singvals": [float(x) for x in s_full[:5]],
            "smallest_kept_singval": float(s_full[rank_full_eps - 1]) if rank_full_eps > 0 else None,
            "oracle_r2_testmean": oracle_full_eval["r2_testmean_baseline"],
            "oracle_r2_trainmean": oracle_full_eval["r2_trainmean_baseline"],
            "oracle_mean_row_cosine": oracle_full_eval["mean_row_cosine_pred_vs_true"],
            "oracle_per_row_r2_median": oracle_full_eval["per_row_r2_median"],
            "captured_fraction": captured_fraction(Yte_full, Q_full),
        }

        # ---- within-family residual oracle (the headline metric) ----
        if eps is not None:
            Y_resid = Y_active[idx] - np.stack(
                [preset_ttl[b][active_rows].reshape(-1) for b in base[idx]]
            ).astype(np.float64)
            pos = {g: i for i, g in enumerate(idx)}
            rtr = np.array([pos[i] for i in itr_fam])
            rte = np.array([pos[i] for i in ite_fam])
            Ytr_r, Yte_r = Y_resid[rtr], Y_resid[rte]

            rank_r_eps, rank_r_loose, s_r = numerical_rank(Ytr_r)
            P_r, Q_r = oracle_projection(Ytr_r, Yte_r, rank_r_eps)
            oracle_r_eval = evaluate(Ytr_r, Yte_r, P_r, active_rows, ndim)
            cap_r = captured_fraction(Yte_r, Q_r)
            predicted_cap = rank_r_eps / Y_active.shape[1]

            entry["within_family_residual"] = {
                "numerical_rank_eps": rank_r_eps,
                "numerical_rank_loose_1e-10": rank_r_loose,
                "top_5_singvals": [float(x) for x in s_r[:5]],
                "oracle_r2_testmean": oracle_r_eval["r2_testmean_baseline"],
                "oracle_r2_trainmean": oracle_r_eval["r2_trainmean_baseline"],
                "oracle_mean_row_cosine": oracle_r_eval["mean_row_cosine_pred_vs_true"],
                "oracle_per_row_r2_median": oracle_r_eval["per_row_r2_median"],
                "captured_fraction": cap_r,
                "predicted_captured_fraction_ntrain_over_dims": predicted_cap,
            }

            achieved = {}
            for rname, report in achieved_reports.items():
                key, val = pull_achieved(report, c, "within_family_residual")
                achieved[rname] = {"best_key": key, "r2_testmean": val}
            entry["achieved_within_family_residual"] = achieved
            best_val = max((v["r2_testmean"] for v in achieved.values() if v["r2_testmean"] is not None),
                           default=None)
            entry["achieved_best_overall"] = best_val

            table_rows.append({
                "condition": c,
                "n_train": int(len(itr_fam)),
                "rank_Ytrain_resid": rank_r_eps,
                "oracle_full_r2": entry["full_target"]["oracle_r2_testmean"],
                "oracle_resid_r2": oracle_r_eval["r2_testmean_baseline"],
                "oracle_captured_fraction": cap_r,
                "predicted_ntrain_over_dims": predicted_cap,
                "achieved_resid_r2_best": best_val,
            })
        else:
            entry["within_family_residual"] = None
            entry["achieved_within_family_residual"] = None
            table_rows.append({
                "condition": c,
                "n_train": int(len(itr_fam)),
                "rank_Ytrain_resid": None,
                "oracle_full_r2": entry["full_target"]["oracle_r2_testmean"],
                "oracle_resid_r2": None,
                "oracle_captured_fraction": None,
                "predicted_ntrain_over_dims": None,
                "achieved_resid_r2_best": None,
            })

        results["conditions"][c] = entry

    results["table"] = table_rows

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    hdr = f"{'condition':<14}{'n_train':>8}{'rank':>6}{'oracle_full_R2':>16}{'oracle_resid_R2':>17}{'captured':>10}{'pred=n/6144':>13}{'achieved_resid_R2':>19}"
    print(hdr)
    print("-" * len(hdr))
    for r in table_rows:
        def f4(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "  n/a"
        print(f"{r['condition']:<14}{r['n_train']:>8}"
              f"{(r['rank_Ytrain_resid'] if r['rank_Ytrain_resid'] is not None else '-'):>6}"
              f"{f4(r['oracle_full_r2']):>16}"
              f"{f4(r['oracle_resid_r2']):>17}"
              f"{f4(r['oracle_captured_fraction']):>10}"
              f"{f4(r['predicted_ntrain_over_dims']):>13}"
              f"{f4(r['achieved_resid_r2_best']):>19}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
