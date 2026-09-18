"""Phase 3, stage 1: numeric, cheap gate for a named presentation axis.

See the phase-3 task brief (session prompt) for full context. Ten shipped
presets carry a free gender label (M1-M5, F1-F5). This estimates the
presentation direction as the class mean difference

    d = mean(F presets) - mean(M presets)

on the 24 active `style_ttl` rows, restricted to five samples per class in
6,120 effective dimensions (24 rows x 255 tangent dims/row after the radial
per-row component is removed) -- too few samples for anything needing a
covariance estimate (LDA, etc.), so mean difference is the only estimator
that does not fit noise.

Does NOT render audio. Prints and writes results/phase3/presentation_axis.json.
Stage 2 (rendering) only proceeds if the leave-one-out gate here clears
(>7/10 correct).

Usage (from py/):
    python3 phase3_presentation_stage1.py
"""

import json
import os

import numpy as np

from phase2b_generate import ACTIVE_ROWS, PRESETS, VOICE_STYLE_DIR
from phase2b_generate_presetspan import BASE_PRESET, build_presetspan_basis
from helper import load_voice_style

OUT_DIR = "results/phase3"
OUT_PATH = os.path.join(OUT_DIR, "presentation_axis.json")

M_PRESETS = [p for p in PRESETS if p.startswith("M")]
F_PRESETS = [p for p in PRESETS if p.startswith("F")]


def flat_active(ttl, active_rows=ACTIVE_ROWS):
    """(1,50,256) style_ttl -> flattened (len(active_rows)*256,) float64 vector
    over the active rows only."""
    return ttl.astype(np.float64)[0, active_rows, :].reshape(-1)


def class_mean_diff(vecs_by_preset, train_presets):
    """d = mean(F in train) - mean(M in train), flat (6144,) vector, plus the
    two class means and their midpoint."""
    f_train = [p for p in train_presets if p in F_PRESETS]
    m_train = [p for p in train_presets if p in M_PRESETS]
    mean_f = np.mean([vecs_by_preset[p] for p in f_train], axis=0)
    mean_m = np.mean([vecs_by_preset[p] for p in m_train], axis=0)
    d = mean_f - mean_m
    midpoint = (mean_f + mean_m) / 2.0
    return d, mean_f, mean_m, midpoint, f_train, m_train


def leave_one_out(vecs_by_preset):
    """For each of the 10 presets, derive d from the other 9 and project the
    held-out preset (centered at that LOO midpoint) onto it. Returns a list
    of per-preset records and the count correct."""
    records = []
    n_correct = 0
    for held_out in PRESETS:
        train = [p for p in PRESETS if p != held_out]
        d, mean_f, mean_m, midpoint, f_train, m_train = class_mean_diff(vecs_by_preset, train)
        d_norm = np.linalg.norm(d)
        x = vecs_by_preset[held_out]
        raw_proj = float(np.dot(x - midpoint, d))
        signed_dist = raw_proj / d_norm  # interpretable in the same units as d
        true_label = "F" if held_out in F_PRESETS else "M"
        pred_label = "F" if raw_proj > 0 else "M"
        correct = pred_label == true_label
        n_correct += int(correct)
        records.append({
            "held_out": held_out,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": correct,
            "raw_projection": raw_proj,
            "signed_distance": signed_dist,
            "d_norm_train": float(d_norm),
            "n_train_f": len(f_train),
            "n_train_m": len(m_train),
        })
    return records, n_correct


def variance_explained(vecs_by_preset, d_full):
    """Fraction of total between-preset variance (sum of squared deviations
    from the 10-preset centroid, summed over all 6144 dims) captured by
    projecting onto the unit direction d_full."""
    X = np.stack([vecs_by_preset[p] for p in PRESETS], axis=0)  # (10, 6144)
    centroid = X.mean(axis=0)
    Xc = X - centroid
    total_var = float(np.sum(Xc ** 2))  # sum over presets and dims

    d_hat = d_full / np.linalg.norm(d_full)
    proj = Xc @ d_hat  # (10,)
    var_along_d = float(np.sum(proj ** 2))

    return total_var, var_along_d, (var_along_d / total_var if total_var > 0 else float("nan"))


def tangent_project_at_base(vec_flat, P, active_rows=ACTIVE_ROWS, n_cols=256):
    """Project a flat (len(active_rows)*n_cols,) vector into the tangent space
    at base point P (len(active_rows), n_cols): remove, per row, the
    component along that row of P. Same operation build_presetspan_basis uses
    for preset differences, applied here to the mean-difference vector."""
    n_rows = len(active_rows)
    v = vec_flat.reshape(n_rows, n_cols)
    radial = np.einsum("rc,rc->r", v, P)
    v_tan = v - radial[:, None] * P
    return v_tan.reshape(-1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_presets = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]) for p in PRESETS}
    vecs_by_preset = {p: flat_active(all_presets[p].ttl) for p in PRESETS}

    # Sanity: rows should already be unit norm (project invariant).
    for p in PRESETS:
        rows = vecs_by_preset[p].reshape(len(ACTIVE_ROWS), 256)
        norms = np.linalg.norm(rows, axis=-1)
        assert np.abs(norms - 1.0).max() < 1e-4, f"{p}: active rows not unit norm ({norms})"

    print("=== Phase 3 stage 1: presentation-axis leave-one-out gate ===")
    print(f"M presets: {M_PRESETS}")
    print(f"F presets: {F_PRESETS}")

    # --- 1. Leave-one-out separability ---
    loo_records, n_correct = leave_one_out(vecs_by_preset)
    print(f"\nLeave-one-out separability: {n_correct}/10 correct")
    for r in loo_records:
        mark = "OK " if r["correct"] else "FAIL"
        print(f"  [{mark}] {r['held_out']:>2} (true={r['true_label']}) "
              f"signed_dist={r['signed_distance']:+.4f} "
              f"pred={r['pred_label']}")

    gate_passed = n_correct > 7  # strictly above 7/10, per task ("at or below 7 stop")

    # --- Full-data direction (all 10 presets) for questions 2 and 3 ---
    d_full, mean_f_full, mean_m_full, midpoint_full, _, _ = class_mean_diff(vecs_by_preset, PRESETS)
    d_full_norm = float(np.linalg.norm(d_full))
    print(f"\nFull-data d = mean(F) - mean(M), ||d||={d_full_norm:.4f} (all 10 presets)")

    # --- 2. Variance explained ---
    total_var, var_along_d, frac_var = variance_explained(vecs_by_preset, d_full)
    print(f"\nBetween-preset variance (10 presets, 6144 dims, from centroid): {total_var:.4f}")
    print(f"Variance along d: {var_along_d:.4f}  ({frac_var * 100:.2f}% of total)")

    # --- 3. Relationship to the 9-dim preset-difference span (basis A at BASE_PRESET) ---
    base_ttl = all_presets[BASE_PRESET].ttl.astype(np.float32)
    B_a, P, rank, extras = build_presetspan_basis(base_ttl, all_presets)
    print(f"\npreset-difference span (basis A at {BASE_PRESET}): rank={rank}")

    d_full_tan = tangent_project_at_base(d_full, P)
    tan_frac_of_norm = float(np.linalg.norm(d_full_tan) / d_full_norm) if d_full_norm > 0 else float("nan")
    print(f"d tangent-projected at {BASE_PRESET}: ||d_tan||/||d|| = {tan_frac_of_norm:.6f} "
          f"(1.0 would mean d already has no radial component at this base)")

    cos_per_basis_vec = (B_a @ d_full_tan) / np.linalg.norm(d_full_tan)
    proj_coeffs = B_a @ d_full_tan  # (rank,)
    in_span_energy = float(np.sum(proj_coeffs ** 2))
    total_tan_energy = float(np.sum(d_full_tan ** 2))
    in_span_fraction = in_span_energy / total_tan_energy if total_tan_energy > 0 else float("nan")
    print(f"Fraction of tangent-projected d's energy inside the 9-dim preset span: "
          f"{in_span_fraction * 100:.2f}%")
    print("Cosine of d (tangent-projected at M1) against each basis vector of the preset span:")
    for i, (c, sv) in enumerate(zip(cos_per_basis_vec, extras["singular_values"])):
        print(f"  basis[{i}] (singular value {sv:.4f}): cosine={c:+.4f}")

    result = {
        "active_rows": ACTIVE_ROWS,
        "m_presets": M_PRESETS,
        "f_presets": F_PRESETS,
        "leave_one_out": {
            "n_correct": n_correct,
            "n_total": 10,
            "gate_threshold": "strictly greater than 7/10 required to proceed to stage 2",
            "gate_passed": gate_passed,
            "records": loo_records,
        },
        "full_data_direction": {
            "base_preset_for_tangent_projection": BASE_PRESET,
            "d_norm_raw": d_full_norm,
            "d_tangent_norm_over_raw_norm_at_base": tan_frac_of_norm,
        },
        "variance_explained": {
            "total_between_preset_variance": total_var,
            "variance_along_d": var_along_d,
            "fraction": frac_var,
        },
        "preset_span_relationship": {
            "basis_rank": rank,
            "singular_values": extras["singular_values"],
            "cosine_per_basis_vector": [float(c) for c in cos_per_basis_vec],
            "in_span_energy_fraction": in_span_fraction,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")

    print(f"\n=== GATE: {'PASSED' if gate_passed else 'FAILED'} "
          f"({n_correct}/10 correct, threshold >7/10) ===")
    return result


if __name__ == "__main__":
    main()
