"""Phase 2b, stage A': preset-difference-span vs random-control perturbation.

See ../new-plan.md "Phase 2: Amortized Style Inversion" and
`phase2b_generate_subspace.py`'s docstring for the shared machinery this
script reuses (tangent-space projection, `sample_style`, the manifest schema,
the seeding rule).

THE QUESTION
------------
Phase 2a (`phase2b_generate_subspace.py`) perturbed style_ttl along ISOTROPIC
random directions in the 6,120-dim tangent space at a fixed base preset.
Those directions have no reason to sound like voices -- the listening notes on
them were mostly about glitches, not about hearing a different voice. But the
ten shipped presets ARE real voices, and Phase 0 established by ear that
linear blends between presets give clean, distinct intermediate voices. So the
span of preset DIFFERENCES (P_i - P_M1) is a subspace that is on-manifold by
construction, and might be a fundamentally more usable perturbation surface
than an isotropic random one of the same dimensionality and amplitude.

Two conditions, identical in every respect except which subspace:

  A. preset-span.  Basis from the 9 difference directions P_i - P_M1 (i != M1),
     restricted to the 24 active rows, projected into the tangent space at M1
     (the same per-row radial-component subtraction `build_basis` does for a
     random Gaussian draw), then reduced to an orthonormal basis by SVD. The
     basis dimensionality is the NUMERICAL RANK of the projected differences,
     not assumed to be 9 -- if the nine presets are not independent in the
     tangent space the true rank is smaller, and everything downstream (both
     conditions, so the comparison stays matched) uses that rank.
  B. random control.  A random subspace of the SAME dimensionality as A,
     built with `phase2b_generate_subspace.build_basis` -- the exact isotropic
     construction Phase 2a used -- seeded, and then sliced to the matched
     rank (`B[:rank]`). Because that basis is QR-nested, slicing to any rank
     gives a genuine draw from the same distribution over that many
     directions, not something recomputed differently.

Both conditions: base preset M1, the same 8 TEXTS, 320 samples each
(240 train / 80 test, random split, matching phase2b_generate_subspace's
TEST_FRAC), coefficients drawn and rescaled so the perturbation has Frobenius
norm eps * sqrt(24) with eps=0.20 -- identical to the existing K=4/16/64
ladder -- with c_realized recorded alongside c_drawn and the in-subspace
fraction, via `sample_style` verbatim.

SPEED IS PINNED TO 1.05, NOT THIS FORK'S 1.0 DEFAULT, ON PURPOSE.
------------------------------------------------------------------
This fork's `speed` default is 1.0 (see CLAUDE.md); the existing
`phase2b_generate_subspace.py` K=4/16/64 ladder that this experiment must
stay comparable to was rendered at speed=1.05 (`phase2b_generate.SPEED`).
speed=1.05 divides the duration predictor's own estimate and so almost
certainly carries some rate of glitch/artifact that speed=1.0 would not (see
docs/GLITCH_MITIGATION.md). We deliberately render BOTH conditions A and B at
1.05 anyway: whatever artifact rate 1.05 contributes is shared equally by both
conditions (same base preset, same texts, same TOTAL_STEP, same seeding rule),
so it cannot bias preset-span against random or vice versa. Do NOT "fix" this
to 1.0 -- that would only decouple this experiment from the ladder it exists
to be compared against. SPEED is defined locally (not imported from
phase2b_generate) specifically so a future change to that module's default
does not silently change this experiment's number.

Also reported, before generation (see `report_geometry()` and its printed
output, and geometry.json written alongside the corpus):

  * natural scale of a real preset difference: ||P_i - P_M1||_F over active
    rows, for each of the nine, against the eps*sqrt(24) perturbation norm.
  * per-row energy balance of basis A vs basis B: for each basis, the average
    (over basis vectors) fraction of each basis vector's energy that lands on
    each of the 24 active rows, summarized as max-row-share / uniform-share
    (uniform = 1/24).
  * principal angles between subspace A and subspace B (scipy
    `subspace_angles`), to confirm near-orthogonality of a random rank-r
    subspace of a 6,120-dim space against the fixed preset-span subspace.

Usage (from py/):
    python3 phase2b_generate_presetspan.py [--smoke]
    python3 phase2b_generate_presetspan.py --condition preset_span
"""

import argparse
import datetime
import json
import os

import numpy as np
import soundfile as sf
from scipy.linalg import subspace_angles
from scipy.signal import resample_poly

from phase2a_ceiling_audit import numerical_rank
from phase2b_generate import (
    ACTIVE_ROWS,
    EMBED_SR,
    LANG,
    ONNX_DIR,
    PRESETS,
    SEED_BASE,
    TEXTS,
    TOTAL_STEP,
    VOICE_STYLE_DIR,
    Style,
    load_text_to_speech,
    load_voice_style,
    timer,
)
from phase2b_generate_subspace import build_basis, sample_style

# Rendered at 1.05 to stay comparable with the existing K=4/16/64 ladder --
# see the module docstring "SPEED IS PINNED TO 1.05" above. Do not change this
# to match the fork's speed=1.0 default; that would only break comparability.
SPEED = 1.05

BASE_PRESET = "M1"
N_PER_COND = 320
TEST_FRAC = 0.25
EPS = 0.20  # matches phase2b_generate_subspace's eps=0.20 total perturbation energy

# Basis seeds. BASIS_SEED_A is unused for randomness (condition A's basis is
# derived from the presets themselves, not drawn) but kept for symmetry/log
# clarity. BASIS_SEED_B seeds the random-control basis via the exact same
# `build_basis` isotropic construction Phase 2a used -- a *different* seed
# from phase2b_generate_subspace's own BASIS_SEED (=SEED_BASE) so this
# experiment's random control is a fresh draw, not a silent re-use of the
# K-ladder's basis object.
BASIS_SEED_B = SEED_BASE + 5000
SAMPLE_SEED = SEED_BASE + 5001

OUT_ROOT = "results/phase2b_presetspan"
CONDITIONS = ["preset_span", "random_control"]


def build_presetspan_basis(base_ttl: np.ndarray, presets: dict):
    """Basis spanning the tangent-space projection of the 9 preset-difference
    directions P_i - P_{BASE_PRESET}, restricted to the 24 active rows.

    Returns (B, P, rank, extras) where B is (rank, 24*256) with orthonormal
    rows (top `rank` right-singular vectors of the projected difference
    matrix), P is (24, 256) the base point, rank is the numerical rank of the
    projected 9 x 6144 difference matrix, and extras carries the raw
    diagnostics (natural-scale norms, singular values) needed for reporting.
    """
    P = base_ttl[0, ACTIVE_ROWS, :].astype(np.float64)  # (24, 256)
    n_rows, n_cols = P.shape

    other_presets = [p for p in PRESETS if p != BASE_PRESET]
    raw_diffs = []
    natural_scale = {}
    for p in other_presets:
        ttl_p = presets[p].ttl.astype(np.float64)[0, ACTIVE_ROWS, :]  # (24, 256)
        diff = ttl_p - P
        natural_scale[p] = float(np.linalg.norm(diff))
        raw_diffs.append(diff)
    G = np.stack(raw_diffs, axis=0)  # (9, 24, 256)

    # Project into the tangent space at P: subtract, per row, the radial
    # component unit_rows would otherwise delete on the next renormalization.
    # Identical operation to `build_basis`'s projection step, applied here to
    # the 9 preset-difference vectors instead of a random Gaussian draw.
    radial = np.einsum("jrc,rc->jr", G, P)  # (9, 24)
    G -= radial[:, :, None] * P[None, :, :]

    G_flat = G.reshape(len(other_presets), n_rows * n_cols)  # (9, 6144)

    rank_eps, rank_loose, s = numerical_rank(G_flat)
    rank = rank_eps

    U, S, Vt = np.linalg.svd(G_flat, full_matrices=False)  # Vt: (9, 6144), orthonormal rows
    B = Vt[:rank].astype(np.float64)  # (rank, 6144)

    extras = {
        "other_presets": other_presets,
        "natural_scale_frobenius": natural_scale,
        "singular_values": [float(x) for x in s],
        "rank_eps": rank_eps,
        "rank_loose_1e-10": rank_loose,
    }
    return B, P, rank, extras


def row_energy_balance(B: np.ndarray, n_rows: int = 24, n_cols: int = 256):
    """For a (K, n_rows*n_cols) orthonormal-row basis, the average (over basis
    vectors) fraction of energy landing on each of the n_rows active rows.
    Returns (per_row_mean_fraction (n_rows,), max_over_uniform_ratio)."""
    K = B.shape[0]
    Br = B.reshape(K, n_rows, n_cols)
    energy = (Br ** 2).sum(axis=-1)  # (K, n_rows), each basis row sums to 1 over n_rows
    per_row_mean = energy.mean(axis=0)  # (n_rows,)
    uniform = 1.0 / n_rows
    return per_row_mean, float(per_row_mean.max() / uniform)


def report_geometry(B_a, B_b, P, extras, rank, out_dir):
    print(f"\n=== Preset-span geometry (base={BASE_PRESET}) ===")
    print(f"numerical rank of 9 tangent-projected preset differences: "
          f"{extras['rank_eps']} (loose 1e-10 threshold: {extras['rank_loose_1e-10']})")
    print(f"singular values: {[round(x, 4) for x in extras['singular_values']]}")

    target_norm = EPS * np.sqrt(len(ACTIVE_ROWS))
    print(f"\nNatural scale of real preset differences (||P_i - P_{BASE_PRESET}||_F over "
          f"active rows) vs perturbation norm target ({target_norm:.4f}):")
    for p, v in extras["natural_scale_frobenius"].items():
        print(f"  {p}: {v:.4f}  (perturbation is {target_norm / v:.3f}x this)")

    row_a, ratio_a = row_energy_balance(B_a)
    row_b, ratio_b = row_energy_balance(B_b)
    print(f"\nPer-row energy balance (max-row-share / uniform-share, uniform=1/24):")
    print(f"  preset-span basis A:   max/uniform = {ratio_a:.3f}")
    print(f"  random-control basis B: max/uniform = {ratio_b:.3f}")

    angles = subspace_angles(B_a.T, B_b.T)  # radians
    angles_deg = np.degrees(angles)
    print(f"\nPrincipal angles between subspace A (preset-span) and B (random control):")
    print(f"  min={angles_deg.min():.2f} deg, mean={angles_deg.mean():.2f} deg, "
          f"max={angles_deg.max():.2f} deg  (90 deg = orthogonal)")

    geometry = {
        "base_preset": BASE_PRESET,
        "active_rows": ACTIVE_ROWS,
        "rank": rank,
        "rank_loose_1e-10": extras["rank_loose_1e-10"],
        "singular_values": extras["singular_values"],
        "eps": EPS,
        "perturbation_frobenius_norm_target": target_norm,
        "natural_scale_frobenius": extras["natural_scale_frobenius"],
        "row_energy_balance": {
            "preset_span": {"per_row_mean_fraction": row_a.tolist(), "max_over_uniform": ratio_a},
            "random_control": {"per_row_mean_fraction": row_b.tolist(), "max_over_uniform": ratio_b},
        },
        "principal_angles_deg": {
            "min": float(angles_deg.min()),
            "mean": float(angles_deg.mean()),
            "max": float(angles_deg.max()),
            "all": angles_deg.tolist(),
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "geometry.json"), "w") as f:
        json.dump(geometry, f, indent=2)
    return geometry


def generate_condition(cond_name, B, P, rank, base_ttl, dp_ref, tts, sample_rng, idx_global_start,
                        out_dir, smoke):
    audio_dir = os.path.join(out_dir, "audio16k")
    os.makedirs(audio_dir, exist_ok=True)

    n = 4 if smoke else N_PER_COND
    n_test = max(1, int(round(n * TEST_FRAC)))
    n_train = n - n_test
    splits = ["train"] * n_train + ["test"] * n_test
    sample_rng.shuffle(splits)

    c_drawn_all = np.zeros((n, rank), dtype=np.float32)
    c_realized_all = np.zeros((n, rank), dtype=np.float32)
    ttl_all = np.zeros((n, 50, 256), dtype=np.float32)
    manifest_records = []

    idx_global = idx_global_start
    with timer(f"{cond_name} ({n} samples)"):
        for local_idx in range(n):
            rows, c_drawn, c_realized, in_subspace_fraction = sample_style(
                sample_rng, B, P, rank, EPS
            )

            ttl = base_ttl.copy()
            ttl[0, ACTIVE_ROWS, :] = rows.astype(np.float32)

            norms = np.linalg.norm(ttl[0, ACTIVE_ROWS, :], axis=-1)
            assert np.abs(norms - 1.0).max() < 1e-5, f"row norm violated: {np.abs(norms - 1).max()}"
            frozen_rows = [r for r in range(50) if r not in ACTIVE_ROWS]
            assert np.array_equal(ttl[0, frozen_rows, :], base_ttl[0, frozen_rows, :]), (
                "frozen row mutated"
            )

            text_i = int(sample_rng.integers(len(TEXTS)))
            seed = SEED_BASE + idx_global
            np.random.seed(seed)  # vocoder latent RNG, as in phase2b_generate_subspace
            wav, dur = tts(TEXTS[text_i], LANG, Style(ttl, dp_ref.copy()), TOTAL_STEP, SPEED)
            trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)

            fname = f"{cond_name}_{local_idx:05d}.wav"
            if not smoke:
                w16 = resample_poly(trimmed, EMBED_SR, tts.sample_rate).astype(np.float32)
                sf.write(os.path.join(audio_dir, fname), w16, EMBED_SR, subtype="PCM_16")

            manifest_records.append({
                "K": rank, "condition": cond_name, "idx": local_idx, "file": fname,
                "text_idx": text_i, "seed": seed, "split": splits[local_idx],
                "in_subspace_fraction": in_subspace_fraction,
                "duration_sec": float(dur[0].item()),
                "peak": float(np.abs(trimmed).max()),
                "rms": float(np.sqrt((trimmed.astype(np.float64) ** 2).mean())),
                "finite": bool(np.isfinite(trimmed).all()),
                "row_norm_max_dev": float(np.abs(norms - 1.0).max()),
            })

            c_drawn_all[local_idx] = c_drawn
            c_realized_all[local_idx] = c_realized
            ttl_all[local_idx] = ttl[0]

            idx_global += 1
            if (idx_global - idx_global_start) % 50 == 0:
                print(f"  {idx_global - idx_global_start}/{n} rendered ({cond_name})", flush=True)

    return manifest_records, c_drawn_all, c_realized_all, ttl_all, idx_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 samples per condition, no audio written")
    ap.add_argument("--condition", choices=CONDITIONS, default=None,
                     help="run only one condition (default: both)")
    ap.add_argument("--out-dir", default=OUT_ROOT)
    args = ap.parse_args()

    out_dir = args.out_dir
    conditions = [args.condition] if args.condition else CONDITIONS

    all_presets = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]) for p in PRESETS}
    base_style = all_presets[BASE_PRESET]
    base_ttl = base_style.ttl.astype(np.float32)  # (1, 50, 256)
    dp_ref = base_style.dp.copy()

    B_a, P, rank, extras = build_presetspan_basis(base_ttl, all_presets)

    B_full_b, P_b = build_basis(base_ttl, n_dims=256, seed=BASIS_SEED_B)
    assert np.allclose(P, P_b), "base point mismatch between condition A and B geometry"
    B_b = B_full_b[:rank]

    for name, B in (("A (preset-span)", B_a), ("B (random control)", B_b)):
        off_diag_max = float(np.abs(B @ B.T - np.eye(B.shape[0])).max())
        print(f"orthonormality check, basis {name}: max |B B^T - I| = {off_diag_max:.3e}")
        assert off_diag_max < 1e-5, f"basis {name} not orthonormal enough: {off_diag_max}"

    geometry = report_geometry(B_a, B_b, P, extras, rank, out_dir)

    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    sample_rng = np.random.default_rng(SAMPLE_SEED)

    basis_by_cond = {"preset_span": B_a, "random_control": B_b}

    idx_global = 0
    for cond_name in conditions:
        cond_dir = os.path.join(out_dir, cond_name)
        B = basis_by_cond[cond_name]
        records, c_drawn_all, c_realized_all, ttl_all, idx_global = generate_condition(
            cond_name, B, P, rank, base_ttl, dp_ref, tts, sample_rng, idx_global, cond_dir, args.smoke
        )

        npz_payload = {
            "basis": B.astype(np.float32),
            "base_ttl": base_ttl[0],
            "active_rows": np.array(ACTIVE_ROWS),
            f"c_drawn_K{rank}": c_drawn_all,
            f"c_realized_K{rank}": c_realized_all,
            f"ttl_K{rank}": ttl_all,
        }
        np.savez_compressed(os.path.join(cond_dir, "subspace.npz"), **npz_payload)

        fractions = [r["in_subspace_fraction"] for r in records]
        meta = {
            "experiment": "phase2b_generate_presetspan",
            "condition": cond_name,
            "date": datetime.date.today().isoformat(),
            "base_preset": BASE_PRESET,
            "rank": rank,
            "n_total": len(records),
            "eps": EPS,
            "active_rows": ACTIVE_ROWS,
            "texts": TEXTS,
            "lang": LANG, "total_step": TOTAL_STEP, "speed": SPEED,
            "speed_note": (
                "Rendered at 1.05, NOT this fork's speed=1.0 default, to stay "
                "comparable with the existing K=4/16/64 ladder rendered at 1.05. "
                "Both conditions A and B share this, so it cannot bias one "
                "against the other. See module docstring."
            ),
            "style_dp": f"held fixed at {BASE_PRESET}'s for every sample",
            "synthesis_sample_rate": tts.sample_rate,
            "embed_sample_rate": EMBED_SR,
            "seed_rule": "np.random.seed(SEED_BASE + idx_global) immediately before each synthesis, "
                         "idx_global shared/continuing across both conditions",
            "basis_seed_random_control": BASIS_SEED_B,
            "sample_seed": SAMPLE_SEED,
            "in_subspace_fraction_mean": float(np.mean(fractions)) if fractions else None,
            "in_subspace_fraction_min": float(np.min(fractions)) if fractions else None,
            "geometry": geometry,
            "records": records,
        }
        with open(os.path.join(cond_dir, "manifest.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nWrote {len(records)} samples -> {cond_dir}")
        bad = [r for r in records if not r["finite"] or r["peak"] >= 1.0]
        print(f"non-finite or clipped clips ({cond_name}): {len(bad)}")


if __name__ == "__main__":
    main()
