"""Phase 2b, stage 4: listen to what the probe recovers.

R^2 on a 6,144-dim target is hard to hear. This refits the family-disjoint
linear ridge probe for one condition, then renders, for a few held-out test
samples, the TRUE style next to the style the probe PREDICTED from that clip's
ECAPA embedding -- same text, same vocoder seed, so the only difference is the
style tensor. Inactive rows are taken from the true tensor so the comparison
isolates the probe's error on the 24 rows it actually predicts.

Usage (from py/):
    python3 phase2b_render_predictions.py --condition eps0.20 --n 4
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from helper import Style, load_text_to_speech, load_voice_style

OUT_DIR = "results/phase2b"
LISTEN_DIR = "results/listening_sets/phase2b_probe_recovery"
ONNX_DIR = "assets/onnx"
VOICE_STYLE_DIR = "assets/voice_styles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="eps0.20")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--features", default="ecapa", choices=["ecapa", "spectral"])
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    S = np.load(os.path.join(OUT_DIR, "styles.npz"))
    E = np.load(os.path.join(OUT_DIR, "embeddings.npz"))
    ttl, active = S["ttl"], [int(r) for r in S["active_rows"]]
    X = E[args.features].astype(np.float64)

    sel = np.array([r["condition"] == args.condition for r in recs])
    tr = np.where(sel & np.array([r["split"] == "train" for r in recs]))[0]
    te = np.where(sel & np.array([r["split"] == "test" for r in recs]))[0]
    Y = ttl[:, active, :].reshape(len(recs), -1).astype(np.float64)

    sc = StandardScaler().fit(X[tr])
    model = RidgeCV(alphas=np.logspace(-3, 8, 34)).fit(sc.transform(X[tr]), Y[tr])
    P = model.predict(sc.transform(X[te]))

    os.makedirs(LISTEN_DIR, exist_ok=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    dp_ref = load_voice_style([os.path.join(VOICE_STYLE_DIR, "M1.json")]).dp

    entries = []
    for k in range(min(args.n, len(te))):
        i = te[k]
        r = recs[i]
        text = meta["texts"][r["text_idx"]]
        true_ttl = ttl[i][None].astype(np.float32).copy()
        pred_ttl = true_ttl.copy()
        rows = P[k].reshape(len(active), -1).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=-1, keepdims=True).clip(min=1e-8)
        pred_ttl[0, active, :] = rows

        cos = float((true_ttl[0, active] * pred_ttl[0, active]).sum(-1).mean())
        for tag, t in (("true", true_ttl), ("probe_pred", pred_ttl)):
            np.random.seed(r["seed"])
            wav, dur = tts(text, meta["lang"], Style(t, dp_ref.copy()),
                           meta["total_step"], meta["speed"])
            w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
            fn = f"{args.condition}_idx{i:05d}_{tag}.wav"
            sf.write(os.path.join(LISTEN_DIR, fn), w, tts.sample_rate)
            entries.append({"file": fn, "kind": tag, "idx": int(i), "base": r["base"],
                            "condition": args.condition, "text": text, "seed": r["seed"],
                            "mean_active_row_cosine_pred_vs_true": round(cos, 4)})
        print(f"idx {i} ({r['base']}): mean active-row cosine pred-vs-true = {cos:.4f}")

    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump({"experiment": "phase2b_probe_recovery",
                   "features": args.features, "condition": args.condition,
                   "split": "family-disjoint; test presets " + ",".join(meta["presets_held_out"]),
                   "inactive_rows": "taken from the true tensor (the probe predicts active rows only)",
                   "lang": meta["lang"], "total_step": meta["total_step"],
                   "speed": meta["speed"], "outputs": entries}, f, indent=2)
    print(f"\nWrote {LISTEN_DIR}")


if __name__ == "__main__":
    main()
