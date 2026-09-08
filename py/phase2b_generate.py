"""Phase 2b, stage 1: synthetic (audio, style) pair generator.

See ../new-plan.md "Phase 2: Amortized Style Inversion", variant 2b.

Phase 2b asks whether an off-the-shelf speaker embedding linearly predicts
`style_ttl`. The published route needs an external style optimizer plus a
speaker corpus; neither is available here. But the engine is its own paired
data generator: synthesizing audio *from* a known style tensor yields a
ground-truth (audio, style) pair for free.

THE DESIGN RISK THIS SCRIPT EXISTS TO CONTROL
---------------------------------------------
The sampling distribution over styles decides whether the eventual R^2 means
anything. Convex blends of the ten shipped presets span at most 9 dimensions,
so a 192-dim ECAPA probe would only have to recover 9 directions and would
score near-perfectly while proving nothing. This script therefore samples at
several *controlled* perturbation magnitudes, from that degenerate
preset-blend condition (kept deliberately, as the artifact baseline) up to
isotropic per-row perturbations whose intrinsic dimensionality is bounded only
by the 24 active rows x 255 tangent dims. The fitting stage reports intrinsic
dimensionality per condition alongside every R^2.

Perturbation model (per condition, `eps`):

    for each active row r:  v_r <- normalize(P_r + eps * u_r),  u_r ~ Unif(S^255)

`eps` is the relative perturbation magnitude, so the angular displacement of a
row is about atan(eps) -- eps=0.1 is ~5.7 deg, eps=0.4 ~21 deg. For scale, the
per-row angle between presets M1 and F1 runs 18 deg mean / 37 deg max
(Phase 0 companion result). Rows are renormalized to unit norm over the last
axis, preserving the project invariant; the 26 inactive rows are held at the
base preset's values.

Usage (from py/):
    python3 phase2b_generate.py [--n-per-condition 320] [--smoke]
"""

import argparse
import datetime
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from helper import Style, load_text_to_speech, load_voice_style, timer

ONNX_DIR = "assets/onnx"
VOICE_STYLE_DIR = "assets/voice_styles"
OUT_DIR = "results/phase2b"
AUDIO_DIR = os.path.join(OUT_DIR, "audio16k")

PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
# Held-out *families*: whole base presets never seen in training, so the probe
# is tested on unseen voices rather than on unseen clips of a seen voice.
HELD_OUT_PRESETS = ["M5", "F4", "F5"]
TRAIN_PRESETS = [p for p in PRESETS if p not in HELD_OUT_PRESETS]

# Phase 0 companion result: 24 of the 50 style_ttl rows carry voice identity.
ACTIVE_ROWS = [0, 2, 5, 6, 7, 8, 9, 13, 15, 16, 18, 19, 20, 22, 23, 27, 31, 32,
               38, 42, 45, 47, 48, 49]

# preset_affine is the degenerate control: convex blends of the presets in the
# split's allowed set, spanning at most (n_presets - 1) dimensions.
CONDITIONS = [
    ("preset_affine", None),
    ("eps0.05", 0.05),
    ("eps0.10", 0.10),
    ("eps0.20", 0.20),
    ("eps0.40", 0.40),
    ("eps0.80", 0.80),
]

# Text is varied across samples so the probe cannot key on text-specific
# artifacts. Eight sentences with different phonetic and prosodic content.
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the sea shore every summer morning.",
    "Please call the doctor before the meeting starts at noon.",
    "How much wood would a woodchuck chuck if it could chuck wood?",
    "The library closes early on Thursday, so bring your books back.",
    "Nine hungry travellers waited quietly under the old stone bridge.",
    "I never thought a single question could change everything so quickly.",
    "Bright yellow flowers grew along the muddy path near the river.",
]

LANG = "en"
TOTAL_STEP = 8
SPEED = 1.05
INCLUDE_DURATION = False
EMBED_SR = 16000
SEED_BASE = 20260908


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def sample_perturbed(rng: np.random.Generator, base_ttl: np.ndarray, eps: float) -> np.ndarray:
    """base_ttl (1,50,256) with the active rows nudged by eps and renormalized."""
    ttl = base_ttl.copy()
    g = rng.standard_normal((len(ACTIVE_ROWS), ttl.shape[-1])).astype(np.float32)
    g = unit_rows(g)
    ttl[0, ACTIVE_ROWS, :] = unit_rows(ttl[0, ACTIVE_ROWS, :] + eps * g)
    return ttl.astype(np.float32)


def sample_preset_blend(rng: np.random.Generator, pool: np.ndarray) -> np.ndarray:
    """Convex blend of the pool's TTLs, rows renormalized. Spans <= len(pool)-1 dims."""
    w = rng.dirichlet(np.ones(pool.shape[0])).astype(np.float32)
    ttl = (w[:, None, None, None] * pool).sum(0)
    return unit_rows(ttl).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=320)
    ap.add_argument("--test-frac", type=float, default=0.25,
                    help="fraction of each condition's samples drawn from held-out presets")
    ap.add_argument("--smoke", action="store_true", help="4 samples per condition, no audio kept")
    args = ap.parse_args()

    n_per = 4 if args.smoke else args.n_per_condition
    os.makedirs(AUDIO_DIR, exist_ok=True)

    styles = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]) for p in PRESETS}
    ttl_by_preset = {p: styles[p].ttl.astype(np.float32) for p in PRESETS}
    dp_ref = styles["M1"].dp  # held fixed: style_dp is not part of the 2b target
    train_pool = np.stack([ttl_by_preset[p] for p in TRAIN_PRESETS])
    test_pool = np.stack([ttl_by_preset[p] for p in HELD_OUT_PRESETS])

    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    rng = np.random.default_rng(SEED_BASE)

    records = []
    ttl_all = []
    idx = 0
    for cond_name, eps in CONDITIONS:
        n_test = max(1, int(round(n_per * args.test_frac)))
        n_train = n_per - n_test
        with timer(f"condition {cond_name} ({n_per} samples)"):
            for k in range(n_per):
                split = "train" if k < n_train else "test"
                allowed = TRAIN_PRESETS if split == "train" else HELD_OUT_PRESETS
                if eps is None:
                    pool = train_pool if split == "train" else test_pool
                    ttl = sample_preset_blend(rng, pool)
                    base_name = "blend:" + "+".join(allowed)
                else:
                    base_name = allowed[rng.integers(len(allowed))]
                    ttl = sample_perturbed(rng, ttl_by_preset[base_name], eps)

                norms = np.linalg.norm(ttl, axis=-1)
                assert np.abs(norms - 1.0).max() < 1e-5, f"row norm violated: {np.abs(norms-1).max()}"

                text_i = int(rng.integers(len(TEXTS)))
                seed = SEED_BASE + idx
                np.random.seed(seed)  # vocoder latent RNG -- audio is a fn of (style, text, seed)
                wav, dur = tts(TEXTS[text_i], LANG, Style(ttl, dp_ref.copy()), TOTAL_STEP, SPEED)
                trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)

                fname = f"{idx:05d}.wav"
                if not args.smoke:
                    w16 = resample_poly(trimmed, EMBED_SR, tts.sample_rate).astype(np.float32)
                    sf.write(os.path.join(AUDIO_DIR, fname), w16, EMBED_SR, subtype="PCM_16")

                records.append({
                    "idx": idx, "file": fname, "condition": cond_name,
                    "eps": eps, "split": split, "base": base_name,
                    "text_idx": text_i, "seed": seed,
                    "duration_sec": float(dur[0].item()),
                    "peak": float(np.abs(trimmed).max()),
                    "rms": float(np.sqrt((trimmed.astype(np.float64) ** 2).mean())),
                    "finite": bool(np.isfinite(trimmed).all()),
                    "row_norm_max_dev": float(np.abs(norms - 1.0).max()),
                })
                ttl_all.append(ttl[0])
                idx += 1
                if idx % 50 == 0:
                    print(f"  {idx} rendered", flush=True)

    ttl_arr = np.stack(ttl_all).astype(np.float32)  # (N, 50, 256)
    np.savez_compressed(os.path.join(OUT_DIR, "styles.npz"),
                        ttl=ttl_arr, active_rows=np.array(ACTIVE_ROWS))
    meta = {
        "experiment": "phase2b_generate",
        "date": datetime.date.today().isoformat(),
        "n_total": len(records),
        "n_per_condition": n_per,
        "conditions": [{"name": c, "eps": e} for c, e in CONDITIONS],
        "presets_train": TRAIN_PRESETS,
        "presets_held_out": HELD_OUT_PRESETS,
        "active_rows": ACTIVE_ROWS,
        "texts": TEXTS,
        "lang": LANG, "total_step": TOTAL_STEP, "speed": SPEED,
        "include_duration": INCLUDE_DURATION,
        "style_dp": "held fixed at M1's for every sample",
        "synthesis_sample_rate": tts.sample_rate,
        "embed_sample_rate": EMBED_SR,
        "seed_rule": "np.random.seed(SEED_BASE + idx) immediately before each synthesis",
        "perturbation_model": (
            "per active row: normalize(P_r + eps * u_r), u_r uniform on the unit sphere "
            "in R^256; the 26 inactive rows stay at the base preset's values"
        ),
        "records": records,
    }
    with open(os.path.join(OUT_DIR, "generation_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {len(records)} samples -> {OUT_DIR}")
    bad = [r for r in records if not r["finite"] or r["peak"] >= 1.0]
    print(f"non-finite or clipped clips: {len(bad)}")


if __name__ == "__main__":
    main()
