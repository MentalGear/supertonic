"""Phase 2b, stage A: low-dimensional subspace perturbation corpus.

See ../new-plan.md "Phase 2: Amortized Style Inversion". Phase 2b perturbed
style_ttl isotropically across all 24 active rows (24*255 = 6120 tangent
dims -- the ambient count of 24*256=6144 overcounts by exactly the 24 radial
directions that unit-norm renormalization kills). With only 240 training
samples the estimator could never express a target of that rank, so the null
result was uninformative. This script confines perturbations to a fixed,
LOW-dimensional linear subspace of the tangent space, nested across
K in {4, 16, 64}, so the estimator has the rank to express the target and we
can measure how many dimensions of style perturbation the audio carries.

Geometry
--------
Each of the 24 active rows of style_ttl is L2-normalized to unit norm, so the
style lives on a product of 24 unit spheres in R^256. `unit_rows` silently
deletes the radial component of any perturbation before renormalizing, so a
basis drawn in the flat ambient space loses part of itself on every sample.
This script instead draws the basis directly in the tangent space at a single
fixed base point (preset M1) and orthonormalizes it with a QR factorization,
which -- because R is upper triangular -- guarantees the nesting
B[:4] subset B[:16] subset B[:64] exactly, not just approximately.

Usage (from py/):
    python3 phase2b_generate_subspace.py [--smoke] [--k 16]
"""

import argparse
import datetime
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from phase2b_generate import (
    ACTIVE_ROWS,
    EMBED_SR,
    LANG,
    ONNX_DIR,
    SEED_BASE,
    SPEED,
    TEXTS,
    TOTAL_STEP,
    VOICE_STYLE_DIR,
    Style,
    load_text_to_speech,
    load_voice_style,
    timer,
    unit_rows,
)

OUT_DIR = "results/phase2b_subspace"
AUDIO_DIR = os.path.join(OUT_DIR, "audio16k")

BASE_PRESET = "M1"
K_LIST = [4, 16, 64]
N_PER_K = {4: 320, 16: 320, 64: 640}
EPS = 0.20  # matches phase2b's eps=0.20 total perturbation energy
TEST_FRAC = 0.25
BASIS_SEED = SEED_BASE  # deterministic, independent of per-sample draws
SAMPLE_SEED = SEED_BASE + 1


def build_basis(base_ttl: np.ndarray, n_dims: int = 256, seed: int = BASIS_SEED):
    """Orthonormal basis for the tangent space at the active rows of base_ttl.

    Returns (B, P): B is (n_dims, 24*256) with B @ B.T == I (rows orthonormal),
    nested so that B[:K] spans the same subspace as the first K raw tangent
    vectors for every K. P is (24, 256), the base point (unit-norm rows).
    """
    P = base_ttl[0, ACTIVE_ROWS, :].astype(np.float64)  # (24, 256), unit rows
    n_rows, n_cols = P.shape

    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_dims, n_rows, n_cols))  # (256, 24, 256)

    # Project each basis vector into the tangent space: subtract, per row, the
    # radial component that unit_rows would otherwise delete.
    radial = np.einsum("jrc,rc->jr", G, P)  # (256, 24)
    G -= radial[:, :, None] * P[None, :, :]

    G_flat = G.reshape(n_dims, n_rows * n_cols)  # (256, 6144)

    # QR on (6144, 256): R upper triangular => first K columns of Q span the
    # same subspace as the first K columns of the input, for every K. This is
    # what makes the K=4/16/64 ladder nested rather than independently drawn.
    Q, _ = np.linalg.qr(G_flat.T, mode="reduced")  # Q: (6144, 256)
    B = Q.T  # (256, 6144), rows orthonormal: B @ B.T == I

    return B.astype(np.float64), P.astype(np.float64)


def sample_style(rng: np.random.Generator, B: np.ndarray, P: np.ndarray, K: int, eps: float):
    """Draw one perturbed style in the K-dim subspace. Returns (ttl_rows, c_drawn,
    c_realized, in_subspace_fraction), where ttl_rows is (24, 256).
    """
    n_rows, n_cols = P.shape
    c = rng.standard_normal(K)
    d = (c @ B[:K]).reshape(n_rows, n_cols)

    target_norm = eps * np.sqrt(n_rows)
    cur_norm = np.linalg.norm(d)
    scale = target_norm / cur_norm
    d = d * scale
    c_drawn = c * scale  # linear map c -> d, so scaling c scales d identically

    rows = unit_rows(P + d)

    realized = (rows - P).reshape(n_rows * n_cols)
    c_realized = B[:K] @ realized
    realized_energy = float(realized @ realized)
    in_subspace_energy = float(c_realized @ c_realized)
    in_subspace_fraction = in_subspace_energy / realized_energy if realized_energy > 0 else float("nan")

    return rows, c_drawn, c_realized, in_subspace_fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 samples per K, no audio written")
    ap.add_argument("--k", type=int, default=None, choices=K_LIST, help="run a single K")
    args = ap.parse_args()

    k_list = [args.k] if args.k is not None else K_LIST
    os.makedirs(AUDIO_DIR, exist_ok=True)

    base_style = load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{BASE_PRESET}.json")])
    base_ttl = base_style.ttl.astype(np.float32)  # (1, 50, 256)
    dp_ref = base_style.dp.copy()

    B, P = build_basis(base_ttl)
    ortho_err = B @ B.T
    off_diag_max = float(np.abs(ortho_err - np.eye(ortho_err.shape[0])).max())
    print(f"orthonormality check: max |B B^T - I| = {off_diag_max:.3e}")
    assert off_diag_max < 1e-5, f"basis not orthonormal enough: {off_diag_max}"

    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    sample_rng = np.random.default_rng(SAMPLE_SEED)

    manifest_records = []
    npz_payload = {
        "basis": B.astype(np.float32),
        "base_ttl": base_ttl[0],
        "active_rows": np.array(ACTIVE_ROWS),
    }

    idx_global = 0
    for K in k_list:
        n_k = 4 if args.smoke else N_PER_K[K]
        n_test = max(1, int(round(n_k * TEST_FRAC)))
        n_train = n_k - n_test
        splits = ["train"] * n_train + ["test"] * n_test
        sample_rng.shuffle(splits)

        c_drawn_all = np.zeros((n_k, K), dtype=np.float32)
        c_realized_all = np.zeros((n_k, K), dtype=np.float32)
        ttl_all = np.zeros((n_k, 50, 256), dtype=np.float32)

        with timer(f"K={K} ({n_k} samples)"):
            for local_idx in range(n_k):
                rows, c_drawn, c_realized, in_subspace_fraction = sample_style(
                    sample_rng, B, P, K, EPS
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
                np.random.seed(seed)  # vocoder latent RNG, as in phase2b_generate
                wav, dur = tts(TEXTS[text_i], LANG, Style(ttl, dp_ref.copy()), TOTAL_STEP, SPEED)
                trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)

                fname = f"K{K}_{local_idx:05d}.wav"
                if not args.smoke:
                    w16 = resample_poly(trimmed, EMBED_SR, tts.sample_rate).astype(np.float32)
                    sf.write(os.path.join(AUDIO_DIR, fname), w16, EMBED_SR, subtype="PCM_16")

                manifest_records.append({
                    "K": K, "idx": local_idx, "file": fname,
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
                if idx_global % 50 == 0:
                    print(f"  {idx_global} rendered total", flush=True)

        npz_payload[f"c_drawn_K{K}"] = c_drawn_all
        npz_payload[f"c_realized_K{K}"] = c_realized_all
        npz_payload[f"ttl_K{K}"] = ttl_all

    np.savez_compressed(os.path.join(OUT_DIR, "subspace.npz"), **npz_payload)

    fractions = [r["in_subspace_fraction"] for r in manifest_records]
    meta = {
        "experiment": "phase2b_generate_subspace",
        "date": datetime.date.today().isoformat(),
        "base_preset": BASE_PRESET,
        "k_list": k_list,
        "n_per_k": {str(K): (4 if args.smoke else N_PER_K[K]) for K in k_list},
        "n_total": len(manifest_records),
        "eps": EPS,
        "active_rows": ACTIVE_ROWS,
        "texts": TEXTS,
        "lang": LANG, "total_step": TOTAL_STEP, "speed": SPEED,
        "style_dp": f"held fixed at {BASE_PRESET}'s for every sample",
        "synthesis_sample_rate": tts.sample_rate,
        "embed_sample_rate": EMBED_SR,
        "seed_rule": "np.random.seed(SEED_BASE + idx_global) immediately before each synthesis",
        "basis_seed": BASIS_SEED,
        "sample_seed": SAMPLE_SEED,
        "orthonormality_off_diag_max": off_diag_max,
        "in_subspace_fraction_mean": float(np.mean(fractions)) if fractions else None,
        "in_subspace_fraction_min": float(np.min(fractions)) if fractions else None,
        "records": manifest_records,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {len(manifest_records)} samples -> {OUT_DIR}")
    bad = [r for r in manifest_records if not r["finite"] or r["peak"] >= 1.0]
    print(f"non-finite or clipped clips: {len(bad)}")


if __name__ == "__main__":
    main()
