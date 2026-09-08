"""Phase 2b: render probe_report.json / audio_sanity.json as readable tables.

Usage (from py/):
    python3 phase2b_summarize.py
"""
import json
import os

OUT_DIR = "results/phase2b"


def main():
    with open(os.path.join(OUT_DIR, "probe_report.json")) as f:
        rep = json.load(f)

    print("=" * 118)
    print("PHASE 2b -- does an off-the-shelf speaker embedding linearly predict style_ttl?")
    print(f"target: {rep['target']}  |  held-out families: {', '.join(rep['presets_held_out'])}")
    print("=" * 118)
    hdr = (f"{'condition':<15}{'eps':>6}{'effRank':>9}{'PC90':>6}{'top192':>8}"
           f"{'R2 fam':>9}{'ctrl':>8}{'select':>8}{'R2 rand':>9}{'whole':>8}{'resid':>8}{'cos':>7}")
    for probe in ("ecapa/linear_ridge", "spectral/linear_ridge",
                  "ecapa/rbf_kernel_ridge", "spectral/rbf_kernel_ridge"):
        first = True
        for cname, c in rep["conditions"].items():
            if probe not in c["probes"]:
                continue
            if first:
                print(f"\n--- probe: {probe} " + "-" * (100 - len(probe)))
                print(hdr)
                first = False
            p = c["probes"][probe]
            d = c["dimensionality_active_style"]
            fam = p["family_disjoint_split"]
            rnd = p["random_split"]
            resid = p.get("within_family_residual")
            print(f"{cname:<15}{('-' if c['eps'] is None else c['eps']):>6}"
                  f"{d['effective_rank_participation_ratio']:>9.1f}"
                  f"{d['n_pcs_for_90pct']:>6d}{d['var_in_top_192_pcs']:>8.3f}"
                  f"{fam['real']['r2_testmean_baseline']:>9.3f}"
                  f"{fam['control_task']['r2_testmean_baseline']:>8.3f}"
                  f"{p['family_disjoint_split']['selectivity_r2']:>8.3f}"
                  f"{rnd['real']['r2_testmean_baseline']:>9.3f}"
                  f"{p['whole_tensor_r2_family_disjoint']:>8.3f}"
                  f"{(resid['real']['r2_testmean_baseline'] if resid else float('nan')):>8.3f}"
                  f"{fam['real']['mean_row_cosine_pred_vs_true']:>7.3f}")

    print("\nColumns: effRank = participation-ratio effective rank of the sampled style set")
    print("  (capped by n samples, so a lower bound); PC90 = principal components for 90% of")
    print("  style variance; top192 = fraction of style variance inside the top 192 PCs, the")
    print("  hard ceiling on any linear map from a 192-dim input; R2 fam = family-disjoint")
    print("  held-out R2 over the 24 active rows; ctrl = Hewitt & Liang shuffled-target probe;")
    print("  select = real minus control; R2 rand = same-size random split; whole = R2 over all")
    print("  50 rows; resid = R2 on the within-family residual (base preset removed);")
    print("  cos = mean cosine between predicted and true active rows.")

    print("\nPer-active-row R^2, ECAPA linear ridge, family-disjoint split")
    print(f"{'condition':<15}" + "".join(f"{r:>6}" for r in rep["active_rows"]))
    for cname, c in rep["conditions"].items():
        pr = c["probes"]["ecapa/linear_ridge"]["family_disjoint_split"]["real"]["per_row_r2"]
        print(f"{cname:<15}" + "".join(f"{pr[str(r)]:>6.2f}" for r in rep["active_rows"]))

    sp = os.path.join(OUT_DIR, "audio_sanity.json")
    if os.path.exists(sp):
        with open(sp) as f:
            san = json.load(f)
        print("\nAudio sanity -- is the perturbed audio still voice-like?")
        keys = ["voiced_fraction", "median_f0", "spectral_flatness", "mod_2_8hz_ratio", "wer"]
        print(f"{'condition':<22}" + "".join(f"{k:>18}" for k in keys))
        rows = [("unperturbed presets", san["reference_band_unperturbed_presets"])]
        rows += list(san["by_condition"].items())
        for name, m in rows:
            cells = ""
            for k in keys:
                v = m.get(k)
                cells += f"{v['mean']:>10.3f} [{v['p10']:.2f},{v['p90']:.2f}]".rjust(18) if v else " " * 18
            print(f"{name:<22}{cells}")


if __name__ == "__main__":
    main()
