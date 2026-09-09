"""Phase 2b follow-up: matched-geometry renders for the prosody-vs-timbre question.

The perturbation ray (`phase2b_perturbation_sweep.py`) is audible to a human
listener, who described the change as per-word emphasis rather than voice
identity. `phase2b_direction_audibility.py` measured direction audibility with
ECAPA -- a speaker-verification embedding trained to be invariant to prosody --
and concluded random directions are 3-5x less audible than preset-aligned ones.
If the two subspaces differ, that ratio is an artifact of the measuring stick.

This renders the missing audio so the same battery can be run on directions of
matched per-row geometry but different orientation:

  * `random`      -- the exact ray direction (default_rng(4242), first draw),
                     already on disk from the sweep; re-rendered here only as a
                     bit-exactness check that the two scripts agree.
  * `toward_F1`   -- normalize(F1_r - M1_r), the preset-difference direction.
  * `presetpca`   -- random, projected onto the 9-dim span of the ten presets.
  * from F1, `toward_M1`, for the M1/F1 asymmetry the listener reported.
  * the same random ray on two further sentences, for content dependence.

Everything is rendered with the ray's own settings -- one sentence unless
stated, style_dp pinned at M1's, total_step 8, speed 1.05, and
np.random.seed(4242) before every call so the vocoder latent is identical and
all audio differences come from style. Clips are therefore sample-aligned.

Usage (from py/):  python3 phase2b_prosody_render.py
"""

import json
import os

import numpy as np
import soundfile as sf

from helper import Style, load_text_to_speech, load_voice_style
from phase2b_generate import ACTIVE_ROWS, TEXTS, unit_rows

RAY_DIR = "results/listening_sets/phase2b_perturbation_ray"
OUT_DIR = "results/listening_sets/phase2b_prosody"
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
EPS = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]
TEXT = TEXTS[0]
SEED = 4242
TOTAL_STEP, SPEED = 8, 1.05


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    styles = {p: load_voice_style([f"assets/voice_styles/{p}.json"]).ttl.astype(np.float32)
              for p in PRESETS}
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp

    def render(ttl, text=TEXT):
        np.random.seed(SEED)
        wav, dur = tts(text, "en", Style(ttl.astype(np.float32), dp_ref.copy()), TOTAL_STEP, SPEED)
        return wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32), tts.sample_rate

    # the ray's direction, reproduced exactly
    rng = np.random.default_rng(SEED)
    d_random = unit_rows(rng.standard_normal((len(ACTIVE_ROWS), 256)).astype(np.float32))

    # 9-dim subspace spanned by the ten shipped presets, in active-row space
    P = np.stack([styles[p][0, ACTIVE_ROWS, :].reshape(-1) for p in PRESETS])
    Q = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][:9]
    g = np.random.default_rng(99).standard_normal(6144).astype(np.float32)
    d_pca = unit_rows((Q.T @ (Q @ g)).reshape(len(ACTIVE_ROWS), 256).astype(np.float32))

    out = []

    def emit(base, name, d, eps_list=EPS, text=TEXT, tag=""):
        A = styles[base][0, ACTIVE_ROWS, :]
        for eps in eps_list:
            ttl = styles[base].copy()
            if eps > 0:
                ttl[0, ACTIVE_ROWS, :] = unit_rows(A + eps * d)
            cos = float((ttl[0, ACTIVE_ROWS] * styles[base][0, ACTIVE_ROWS]).sum(-1).mean())
            w, sr = render(ttl, text)
            fn = f"{tag or base}_{name}_eps{eps:.2f}.wav"
            sf.write(os.path.join(OUT_DIR, fn), w, sr)
            out.append({"file": fn, "base": base, "direction": name, "eps": eps,
                        "text": text,
                        "mean_active_row_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
                        "peak": float(np.abs(w).max()),
                        "n_samples": int(w.size)})
            print(f"  {fn}: angle {out[-1]['mean_active_row_angle_deg']:5.1f} deg  "
                  f"peak {out[-1]['peak']:.3f}", flush=True)

    # bit-exactness check against the existing ray
    print("reproducibility check vs the existing ray", flush=True)
    for b in ("M1", "F1"):
        w, sr = render(styles[b])
        ref, _ = sf.read(os.path.join(RAY_DIR, f"{b}_eps0.00.wav"), dtype="float32")
        n = min(w.size, ref.size)
        print(f"  {b}: max|new-ray| = {np.abs(w[:n] - ref[:n]).max():.3e}  "
              f"len {w.size} vs {ref.size}", flush=True)

    print("\nM1, preset-difference direction toward F1", flush=True)
    emit("M1", "towardF1", unit_rows(styles["F1"][0, ACTIVE_ROWS, :] - styles["M1"][0, ACTIVE_ROWS, :]))
    print("\nM1, random direction inside the preset-spanned subspace", flush=True)
    emit("M1", "presetpca", d_pca)
    print("\nF1, preset-difference direction toward M1", flush=True)
    emit("F1", "towardM1", unit_rows(styles["M1"][0, ACTIVE_ROWS, :] - styles["F1"][0, ACTIVE_ROWS, :]))

    print("\nthe ray's random direction on two further sentences", flush=True)
    for i, ti in enumerate((1, 3)):
        emit("M1", "random", d_random, eps_list=[0.0, 0.20, 0.80, 3.20],
             text=TEXTS[ti], tag=f"M1t{ti}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"experiment": "phase2b_prosody_matched_geometry",
                   "text": TEXT, "lang": "en", "total_step": TOTAL_STEP, "speed": SPEED,
                   "style_dp": "held fixed at M1's", "rng_seed": SEED,
                   "note": "random-direction ray lives in ../phase2b_perturbation_ray",
                   "active_rows": ACTIVE_ROWS, "outputs": out}, f, indent=2)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
