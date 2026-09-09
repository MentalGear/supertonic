"""Phase 2b: which directions in style space are audible at all?

The perturbation ray showed that a large isotropic perturbation of the active
rows -- 72 degrees per row, twice the largest per-row angle between presets M1
and F1 -- leaves F0, voicing, rhythm and WER essentially unchanged. If random
style directions are inaudible, then no encoder of the audio can recover them,
and a null probe result says something about the style->audio map rather than
about the embedding.

This measures that directly. From base M1, three families of per-row unit
direction, applied to the 24 active rows at matched magnitudes so the geometry
is identical and only the direction's orientation differs:

  * `random`       -- isotropic in R^256 per row;
  * `preset_diff`  -- normalize(P_r - M1_r) toward another shipped preset;
  * `preset_pca`   -- random, but projected onto the 9-dimensional subspace
                      spanned by the ten shipped presets before normalizing.

Audibility is measured as mean |delta log-STFT| from the unperturbed base (the
Phase 0 row-locality metric) and as ECAPA cosine distance. The vocoder RNG is
seeded identically for every render, so all audio differences come from style.

Usage (from py/):  python3 phase2b_direction_audibility.py
"""

import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft
from speechbrain.inference.speaker import EncoderClassifier

from helper import Style, load_text_to_speech, load_voice_style
from phase2b_generate import ACTIVE_ROWS, TEXTS, unit_rows

OUT_DIR = "results/phase2b"
BASE = "M1"
OTHERS = ["F1", "F2", "M3", "F5"]
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
EPS = [0.1, 0.2, 0.4, 0.8]
SEED = 1234


def log_spectrum(w, sr):
    _, _, z = stft(w, fs=sr, nperseg=1024, noverlap=768)
    return np.log(np.abs(z) + 1e-6)


def sdist(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return float(np.abs(a[..., :n] - b[..., :n]).mean())


def main():
    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    styles = {p: load_voice_style([f"assets/voice_styles/{p}.json"]).ttl.astype(np.float32)
              for p in PRESETS}
    dp_ref = load_voice_style([f"assets/voice_styles/{BASE}.json"]).dp
    base = styles[BASE]
    A = base[0, ACTIVE_ROWS, :]

    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/home/user/.cache/ecapa",
                                         run_opts={"device": "cpu"})
    enc.eval()
    from scipy.signal import resample_poly

    def render(ttl):
        np.random.seed(SEED)
        wav, dur = tts(TEXTS[0], "en", Style(ttl, dp_ref.copy()), 8, 1.05)
        w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
        w16 = resample_poly(w, 16000, tts.sample_rate).astype(np.float32)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(w16)[None, :]).squeeze().numpy()
        return log_spectrum(w, tts.sample_rate), e / np.linalg.norm(e)

    # 9-dim subspace spanned by the ten shipped presets, in active-row space
    P = np.stack([styles[p][0, ACTIVE_ROWS, :].reshape(-1) for p in PRESETS])
    Q = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][:9]      # (9, 6144)

    rng = np.random.default_rng(SEED)
    directions = []
    for k in range(4):
        directions.append(("random", f"random{k}",
                           unit_rows(rng.standard_normal((len(ACTIVE_ROWS), 256)).astype(np.float32))))
    for p in OTHERS:
        directions.append(("preset_diff", f"toward_{p}",
                           unit_rows(styles[p][0, ACTIVE_ROWS, :] - A)))
    for k in range(4):
        g = rng.standard_normal(6144).astype(np.float32)
        g = (Q.T @ (Q @ g)).reshape(len(ACTIVE_ROWS), 256).astype(np.float32)
        directions.append(("preset_pca", f"presetpca{k}", unit_rows(g)))

    spec0, emb0 = render(base)
    rows = []
    for family, name, d in directions:
        for eps in EPS:
            ttl = base.copy()
            ttl[0, ACTIVE_ROWS, :] = unit_rows(A + eps * d)
            spec, emb = render(ttl)
            rows.append({"family": family, "direction": name, "eps": eps,
                         "spectral_distance": sdist(spec, spec0),
                         "ecapa_cosine_distance": float(1.0 - emb @ emb0)})
            print(f"  {family:<12} {name:<14} eps={eps:<4} "
                  f"dSTFT={rows[-1]['spectral_distance']:.4f}  "
                  f"dECAPA={rows[-1]['ecapa_cosine_distance']:.4f}", flush=True)

    # reference scale: the full M1 -> F1 style swap
    specF, embF = render(styles["F1"])
    ref = {"spectral_distance": sdist(specF, spec0),
           "ecapa_cosine_distance": float(1.0 - embF @ emb0)}
    print(f"\nreference, full M1->F1 swap: dSTFT={ref['spectral_distance']:.4f} "
          f"dECAPA={ref['ecapa_cosine_distance']:.4f}")

    print(f"\n{'eps':>5}" + "".join(f"{f:>26}" for f in ("random", "preset_pca", "preset_diff")))
    summary = {}
    for eps in EPS:
        line = f"{eps:>5}"
        for fam in ("random", "preset_pca", "preset_diff"):
            v = [r for r in rows if r["family"] == fam and r["eps"] == eps]
            ds = np.mean([x["spectral_distance"] for x in v])
            de = np.mean([x["ecapa_cosine_distance"] for x in v])
            summary.setdefault(fam, {})[eps] = {"spectral_distance": float(ds),
                                                "ecapa_cosine_distance": float(de)}
            line += f"   dSTFT {ds:.4f} dECAPA {de:.4f}"
        print(line)

    with open(os.path.join(OUT_DIR, "direction_audibility.json"), "w") as f:
        json.dump({"base": BASE, "text": TEXTS[0], "seed": SEED, "eps": EPS,
                   "reference_full_M1_to_F1": ref, "per_direction": rows,
                   "family_means": summary}, f, indent=2)
    print(f"\nWrote {OUT_DIR}/direction_audibility.json")


if __name__ == "__main__":
    main()
