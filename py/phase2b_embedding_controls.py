"""Phase 2b, stage 3b: two controls on the embedding side of the probe.

A null probe result is only about the two spaces if the embedding itself is
behaving. So:

  1. Does ECAPA separate the ten shipped presets on THIS engine's synthetic
     audio at all? (same-speaker vs different-speaker cosine, and 1-NN speaker
     identification over the 10 presets x 8 sentences reference renders). If it
     cannot, the null is domain mismatch, not a statement about style space.
  2. Does the style perturbation move the embedding? Per condition, the rank
     correlation between pairwise style distance and pairwise embedding
     distance. A style direction the embedding is blind to shows up here as a
     correlation near zero even though the styles differ.

Usage (from py/):  python3 phase2b_embedding_controls.py
"""

import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.stats import spearmanr
from speechbrain.inference.speaker import EncoderClassifier

OUT_DIR = "results/phase2b"
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]


def main():
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/home/user/.cache/ecapa",
                                         run_opts={"device": "cpu"})
    enc.eval()
    embs, labels = [], []
    for p in PRESETS:
        for ti in range(8):
            fp = os.path.join(OUT_DIR, "reference16k", f"{p}_t{ti}.wav")
            w, _ = sf.read(fp, dtype="float32")
            with torch.no_grad():
                embs.append(enc.encode_batch(torch.from_numpy(w)[None, :]).squeeze().numpy())
            labels.append(p)
    E = np.array(embs)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    lab = np.array(labels)
    S = E @ E.T
    same = np.array([[lab[i] == lab[j] for j in range(len(lab))] for i in range(len(lab))])
    off = ~np.eye(len(lab), dtype=bool)
    nn = np.argmax(np.where(off, S, -np.inf), axis=1)
    ctrl1 = {
        "same_preset_cosine_mean": float(S[same & off].mean()),
        "diff_preset_cosine_mean": float(S[~same].mean()),
        "one_nn_preset_accuracy": float((lab[nn] == lab).mean()),
        "n_clips": int(len(lab)),
    }
    print("ECAPA on this engine's synthetic audio: "
          f"same-preset cos {ctrl1['same_preset_cosine_mean']:.3f}, "
          f"diff-preset cos {ctrl1['diff_preset_cosine_mean']:.3f}, "
          f"1-NN preset accuracy {ctrl1['one_nn_preset_accuracy']:.3f} "
          f"(chance 0.1) over {ctrl1['n_clips']} clips")

    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    ttl = np.load(os.path.join(OUT_DIR, "styles.npz"))["ttl"]
    active = meta["active_rows"]
    X = np.load(os.path.join(OUT_DIR, "embeddings.npz"))["ecapa"]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cond = np.array([r["condition"] for r in recs])
    base = np.array([r["base"] for r in recs])

    ctrl2 = {}
    rng = np.random.default_rng(0)
    for c in dict.fromkeys(cond.tolist()):
        idx = np.where(cond == c)[0]
        # within a single base preset, so the correlation is about the
        # perturbation and not about which voice it started from
        for scope, sel in (("all_bases", idx),
                           ("within_one_base", idx[base[idx] == base[idx][0]])):
            if len(sel) < 30:
                continue
            sel = rng.choice(sel, size=min(120, len(sel)), replace=False)
            Y = ttl[sel][:, active, :].reshape(len(sel), -1)
            ds = np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
            de = 1.0 - Xn[sel] @ Xn[sel].T
            iu = np.triu_indices(len(sel), 1)
            r = spearmanr(ds[iu], de[iu]).statistic
            ctrl2.setdefault(c, {})[scope] = {"spearman_style_vs_embedding_distance": float(r),
                                              "n_pairs": int(len(iu[0]))}
            print(f"  {c:<14} {scope:<16} spearman(style dist, ECAPA dist) = {r:+.3f}")

    with open(os.path.join(OUT_DIR, "embedding_controls.json"), "w") as f:
        json.dump({"preset_separability": ctrl1, "distance_correlation": ctrl2}, f, indent=2)
    print(f"\nWrote {OUT_DIR}/embedding_controls.json")


if __name__ == "__main__":
    main()
