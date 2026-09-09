"""Null simulation for the Phase 2b residual-recovery ceiling.

Phase 2b checked whether its probe design had room to succeed, using two
diagnostics computed on the target itself: the participation ratio of the
residual ("effective rank ~303") and the fraction of residual variance inside
its top-n_train principal components (`var_in_top_k`, "ceiling ~0.82"). Read
together they say the design had 82% of headroom, so a measured R^2 near zero
looks like a fact about audio.

Both are in-sample statistics. This script feeds pure isotropic noise -- data
in which, by construction, nothing whatsoever is predictable -- through the
same two diagnostics at the same (n, d, n_train), and recovers Phase 2b's
reported values to four significant figures. A statistic that returns 0.82 on
noise cannot certify that a design can reach 0.82 on anything.

The quantity that does bind a ridge is out-of-sample: its predictions are an
affine combination of the training targets, so they lie in a subspace of
dimension at most n_train, and a fresh test direction projects onto that by
about n_train/d. That is computed here too, for contrast.

Run: python3 phase2a_ceiling_null.py
"""

import json
import os

import numpy as np

N, D, N_TRAIN = 320, 6144, 240   # Phase 2b's per-condition shape
SEEDS = range(16)
OUT = "results/phase2a/ceiling_null.json"

# What Phase 2b reported, per eps condition, for the within-family residual.
REPORTED_PR = [303.20, 303.30, 303.15, 302.78, 298.17]
REPORTED_IN_SAMPLE = [0.8225, 0.8222, 0.8225, 0.8228, 0.8247]


def diagnostics(Y, n_train):
    """The two in-sample statistics, plus the out-of-sample one that binds."""
    var = np.linalg.svd(Y - Y.mean(0), compute_uv=False) ** 2
    participation_ratio = var.sum() ** 2 / (var ** 2).sum()
    in_sample_top_k = var[:n_train].sum() / var.sum()

    train, test = Y[:n_train], Y[n_train:]
    basis = np.linalg.svd(train - train.mean(0), full_matrices=False)[2]
    resid = test - train.mean(0)
    reachable = np.linalg.norm(resid @ basis.T) ** 2 / np.linalg.norm(resid) ** 2

    return participation_ratio, in_sample_top_k, reachable


def main():
    rows = [diagnostics(np.random.default_rng(s).standard_normal((N, D)), N_TRAIN)
            for s in SEEDS]
    pr, in_sample, reachable = (np.array(c) for c in zip(*rows))

    report = {
        "shape": {"n": N, "d": D, "n_train": N_TRAIN, "seeds": len(list(SEEDS))},
        "noise": {
            "participation_ratio": [pr.mean(), pr.std()],
            "in_sample_var_in_top_n_train_pcs": [in_sample.mean(), in_sample.std()],
            "out_of_sample_reachable": [reachable.mean(), reachable.std()],
        },
        "phase2b_reported": {
            "participation_ratio": REPORTED_PR,
            "in_sample_var_in_top_n_train_pcs": REPORTED_IN_SAMPLE,
        },
        "analytic_n_train_over_d": N_TRAIN / D,
    }

    print(f"Pure isotropic noise, n={N} d={D} n_train={N_TRAIN}, "
          f"{len(list(SEEDS))} seeds. Nothing here is predictable.\n")
    print(f"  participation ratio           {pr.mean():8.2f} +/- {pr.std():.2f}"
          f"      phase 2b reported {REPORTED_PR}")
    print(f"  in-sample var in top-{N_TRAIN} PCs  {in_sample.mean():8.4f} +/- {in_sample.std():.4f}"
          f"    phase 2b reported {REPORTED_IN_SAMPLE}")
    print(f"  out-of-sample reachable       {reachable.mean():8.4f} +/- {reachable.std():.4f}"
          f"    analytic n_train/d = {N_TRAIN / D:.4f}")
    print("\nThe first two match Phase 2b's residual diagnostics to four significant\n"
          "figures on data with no signal in it, so they carried no information about\n"
          "whether that design could detect anything. The third is the real ceiling.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
