"""Phase 2b: does averaging the embedding over several utterances rescue the probe?

The single-utterance probe is limited by a nuisance the design can measure: two
renders of the SAME preset on different sentences sit at ECAPA cosine distance
~0.27, while a style perturbation of eps=0.2 moves the embedding only ~0.06.
Text variation is roughly 4x larger than the style signal, so a single-utterance
embedding may simply be too noisy to carry the style, independent of whether the
two spaces are related.

Standard speaker-embedding practice averages many utterances per speaker so that
text and channel cancel while identity persists. This re-renders each style of
one condition on K utterances, averages the L2-normalized embeddings, and refits
the identical probe -- family-disjoint and random splits, both with the Hewitt &
Liang control -- so the single-utterance and K-utterance numbers are directly
comparable.

Usage (from py/):  python3 phase2b_multi_utterance.py --condition eps0.20 --k 4
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier

from helper import Style, load_text_to_speech, load_voice_style
from phase2b_probe import ALPHAS, evaluate, fit_ridge, shuffled_targets

OUT_DIR = "results/phase2b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="eps0.20")
    ap.add_argument("--k", type=int, default=4, help="utterances averaged per style")
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    texts = meta["texts"]
    S = np.load(os.path.join(OUT_DIR, "styles.npz"))
    ttl, active = S["ttl"], [int(r) for r in S["active_rows"]]
    E1 = np.load(os.path.join(OUT_DIR, "embeddings.npz"))["ecapa"].astype(np.float64)

    idx = np.array([i for i, r in enumerate(recs) if r["condition"] == args.condition])
    print(f"{args.condition}: {len(idx)} styles x {args.k} utterances")

    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/home/user/.cache/ecapa",
                                         run_opts={"device": "cpu"})
    enc.eval()
    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp

    cache = os.path.join(OUT_DIR, f"embeddings_multi_{args.condition}_k{args.k}.npz")
    if os.path.exists(cache):
        Emul = np.load(cache)["ecapa_mean"]
        print(f"loaded cached {cache}")
    else:
        Emul = np.zeros((len(idx), 192))
        for j, i in enumerate(idx):
            r = recs[i]
            e = [E1[i] / np.linalg.norm(E1[i])]
            for k in range(1, args.k):
                ti = (r["text_idx"] + k) % len(texts)
                np.random.seed(r["seed"] + 100000 * k)
                wav, dur = tts(texts[ti], meta["lang"],
                               Style(ttl[i][None].astype(np.float32), dp_ref.copy()),
                               meta["total_step"], meta["speed"])
                w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
                w16 = resample_poly(w, 16000, tts.sample_rate).astype(np.float32)
                with torch.no_grad():
                    v = enc.encode_batch(torch.from_numpy(w16)[None, :]).squeeze().numpy()
                e.append(v / np.linalg.norm(v))
            Emul[j] = np.mean(e, axis=0)
            if (j + 1) % 40 == 0:
                print(f"  {j + 1}/{len(idx)} styles", flush=True)
        np.savez_compressed(cache, ecapa_mean=Emul.astype(np.float32), idx=idx)

    Y = ttl[idx][:, active, :].reshape(len(idx), -1).astype(np.float64)
    base = np.array([recs[i]["base"] for i in idx])
    presets = {p: load_voice_style([f"assets/voice_styles/{p}.json"]).ttl[0] for p in set(base)}
    Yres = Y - np.stack([presets[b][active].reshape(-1) for b in base]).astype(np.float64)
    split = np.array([recs[i]["split"] for i in idx])
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    rp = np.random.default_rng(11).permutation(len(idx))
    rtr, rte = rp[: len(tr)], rp[len(tr):]

    out = {"condition": args.condition, "k": args.k, "n": int(len(idx))}
    for xname, X in (("single_utterance", E1[idx]), (f"mean_of_{args.k}", Emul)):
        out[xname] = {}
        for sname, (a, b) in (("family_disjoint", (tr, te)), ("random", (rtr, rte))):
            for tname, T in (("full_target", Y), ("within_family_residual", Yres)):
                P, _ = fit_ridge(X[a], T[a], X[b])
                real = evaluate(T[a], T[b], P, active, ttl.shape[-1])
                Tc = shuffled_targets(T, np.arange(len(T)))
                Pc, _ = fit_ridge(X[a], Tc[a], X[b])
                ctrl = evaluate(Tc[a], Tc[b], Pc, active, ttl.shape[-1])
                out[xname][f"{sname}/{tname}"] = {
                    "r2_testmean": real["r2_testmean_baseline"],
                    "r2_vs_constant_trainmean": real["r2_trainmean_baseline"],
                    "control_r2_testmean": ctrl["r2_testmean_baseline"],
                    "mean_row_cosine": real["mean_row_cosine_pred_vs_true"],
                }
                v = out[xname][f"{sname}/{tname}"]
                print(f"{xname:<16} {sname:<15} {tname:<24} "
                      f"R2(test-mean)={v['r2_testmean']:+.4f}  "
                      f"R2(vs constant)={v['r2_vs_constant_trainmean']:+.4f}  "
                      f"control={v['control_r2_testmean']:+.4f}  "
                      f"cos={v['mean_row_cosine']:.4f}")

    with open(os.path.join(OUT_DIR, f"multi_utterance_{args.condition}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/multi_utterance_{args.condition}.json")


if __name__ == "__main__":
    main()
