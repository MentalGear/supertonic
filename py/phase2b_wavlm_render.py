"""Phase 2b, stage 4b: hear what WavLM recovers, beside what ECAPA recovered.

`phase2b_render_predictions.py` rendered the ECAPA half -- held-out voices, true
style against probe-predicted style, same text and vocoder seed -- into
results/listening_sets/phase2b_probe_recovery. The WavLM run
(`phase2b_wavlm.py`) rendered nothing, so the one number that matters to a
listener was never testable by ear: WavLM is roughly 2x ECAPA on the full target
(family-disjoint R^2 0.142 against 0.088 pooled). Does 2x a small number sound
twice as good, or does it still sound wrong?

This renders the missing third clip per sample, so each held-out sample gives a
TRIPLE differing only in the style tensor:

    true  /  ECAPA-predicted  /  WavLM-predicted

Both probes are refitted here on identical indices with `phase2b_probe.fit_ridge`
(StandardScaler + RidgeCV, the same alpha grid), family-disjoint, active rows
only -- so the side-by-side is the same protocol the reports quote. WavLM uses
the best representation from the layer sweep: layer 3, mean+std pooled, 2048-dim.
Inactive rows are taken from the true tensor for both, so the comparison isolates
the 24 rows the probes actually predict.

Frame alignment holds by construction (same text, same seed, `style_dp` pinned),
so a direct log-mel distance against the true render is a valid instrument
rather than a summary statistic. All three clips are rms level-matched to the
true render, since a probe that mostly predicts the train mean will also predict
its loudness and would otherwise win or lose the A/B on gain alone.

Usage (from py/):  python3 phase2b_wavlm_render.py [--condition eps0.20] [--n 3]
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

import phase2b_probe as P
import phase2b_wavlm as W
from helper import Style, load_text_to_speech, load_voice_style
from phase2b_prosody_analysis import rms
from phase2b_prosody_mel import logmel

OUT_DIR = "results/phase2b_collapse"
LISTEN_DIR = "results/listening_sets/phase2b_probe_recovery_wavlm"
WAVLM_LAYERS, WAVLM_POOLING = [3], "meanstd"
LOUD_DB = 25.0


def to16k(w, sr):
    from scipy.signal import resample_poly
    return resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="eps0.20")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)

    d = W.load_all()
    meta, recs = d["meta"], d["recs"]
    active, ndim = d["active_rows"], d["ndim"]
    idx, itr, ite, _, _ = W.split_indices(d, args.condition)
    Y = d["Y_active"]

    reps = {
        "ecapa": d["ecapa"],
        "wavlm_L3_meanstd": W.rep_from(d["wavlm_mean"], d["wavlm_std"],
                                       WAVLM_LAYERS, WAVLM_POOLING),
    }
    preds, fit_info = {}, {}
    for name, X in reps.items():
        Pte, alpha = P.fit_ridge(X[itr], Y[itr], X[ite])
        ev = P.evaluate(Y[itr], Y[ite], Pte, active, ndim)
        preds[name] = Pte
        fit_info[name] = {"dim": int(X.shape[1]), "alpha": alpha,
                          "n_train": int(len(itr)), "n_test": int(len(ite)),
                          "family_disjoint_r2_trainmean": ev["r2_trainmean_baseline"],
                          "mean_row_cosine_over_test_set": ev["mean_row_cosine_pred_vs_true"]}
        print(f"{name:<18} dim {X.shape[1]:>5}  alpha {alpha:>10.3g}  "
              f"family-disjoint R^2 {ev['r2_trainmean_baseline']:+.4f}  "
              f"mean row cosine {ev['mean_row_cosine_pred_vs_true']:.4f}")

    # The baseline that decides whether either probe is doing anything audible:
    # predict the training mean for every held-out clip, ignoring the audio.
    # The reports put probe cosine at 0.896 against 0.887 for exactly this, so
    # it belongs in the listening set rather than only in a table.
    train_mean = Y[itr].mean(0)

    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp
    ttl_all = d["ttl"]

    def build(pred_row_block, true_ttl):
        t = true_ttl.copy()
        rows = pred_row_block.reshape(len(active), ndim).astype(np.float32)
        rows = rows / np.linalg.norm(rows, axis=-1, keepdims=True).clip(min=1e-8)
        t[0, active, :] = rows
        dev = float(np.abs(np.linalg.norm(t, axis=-1) - 1.0).max())
        return t, dev

    entries, samples, worst_dev = [], [], 0.0
    for k in range(min(args.n, len(ite))):
        i = int(ite[k])
        r = recs[i]
        text = meta["texts"][r["text_idx"]]
        true_ttl = ttl_all[i][None].astype(np.float32).copy()

        variants = {"true": (true_ttl, None)}
        for name, block in list((n_, preds[n_][k]) for n_ in reps) + [("train_mean", train_mean)]:
            t, dev = build(block, true_ttl)
            worst_dev = max(worst_dev, dev)
            variants[name] = (t, float((true_ttl[0, active] * t[0, active]).sum(-1).mean()))

        wavs = {}
        for tag, (t, _) in variants.items():
            np.random.seed(r["seed"])
            wav, dur = tts(text, meta["lang"], Style(t, dp_ref.copy()),
                           meta["total_step"], meta["speed"])
            wavs[tag] = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
        n = min(w.size for w in wavs.values())
        assert len(set(w.size for w in wavs.values())) == 1, "clips not frame-aligned"

        sr = tts.sample_rate
        mel = {t_: logmel(to16k(w, sr)) for t_, w in wavs.items()}
        nf = min(m.shape[1] for m in mel.values())
        env = mel["true"][:, :nf].mean(0)
        loud = env > env.max() - LOUD_DB
        ref_rms = float(np.sqrt((wavs["true"].astype(np.float64) ** 2).mean()))

        row = {"idx": i, "base": r["base"], "text": text, "seed": r["seed"],
               "condition": args.condition, "n_samples": int(n), "probes": {}}
        for tag in ("ecapa", "wavlm_L3_meanstd", "train_mean"):
            D = mel[tag][:, :nf] - mel["true"][:, :nf]
            g = D.mean(0)
            lvl = float(20 * np.log10(np.sqrt((wavs[tag].astype(np.float64) ** 2).mean()) / ref_rms))
            # what the listener actually hears is the level-matched clip, so the
            # distance is quoted with the overall gain removed as well as raw
            Dm = D - lvl
            row["probes"][tag] = {
                "logmel_rms_db_vs_true_level_matched": round(rms(Dm[:, loud]), 3),
                "mean_active_row_cosine_vs_true": round(float(variants[tag][1]), 4),
                "min_active_row_cosine_vs_true": round(float(
                    (true_ttl[0, active] * variants[tag][0][0, active]).sum(-1).min()), 4),
                "logmel_rms_db_vs_true": round(rms(D[:, loud]), 3),
                "logmel_broadband_gain_db": round(rms(g[loud]), 3),
                "logmel_spectral_shape_db": round(rms((D - g)[:, loud]), 3),
                "level_db_vs_true": round(lvl, 3),
            }
        # how far apart are the two PREDICTIONS from each other?
        Dpp = mel["wavlm_L3_meanstd"][:, :nf] - mel["ecapa"][:, :nf]
        row["ecapa_vs_wavlm"] = {
            "style_active_row_cosine": round(float(
                (variants["ecapa"][0][0, active] *
                 variants["wavlm_L3_meanstd"][0][0, active]).sum(-1).mean()), 4),
            "logmel_rms_db": round(rms(Dpp[:, loud]), 3),
        }
        Dwt = mel["wavlm_L3_meanstd"][:, :nf] - mel["train_mean"][:, :nf]
        row["wavlm_vs_trainmean"] = {
            "style_active_row_cosine": round(float(
                (variants["train_mean"][0][0, active] *
                 variants["wavlm_L3_meanstd"][0][0, active]).sum(-1).mean()), 4),
            "logmel_rms_db": round(rms(Dwt[:, loud]), 3),
        }
        samples.append(row)

        for tag, fn_tag in (("true", "true"), ("ecapa", "ecapa_pred"),
                            ("wavlm_L3_meanstd", "wavlm_pred"),
                            ("train_mean", "trainmean_baseline")):
            w = wavs[tag]
            gdb = 20 * np.log10(ref_rms / max(float(np.sqrt((w.astype(np.float64) ** 2).mean())), 1e-12))
            y = (w * 10.0 ** (gdb / 20.0)).astype(np.float32)
            fn = f"{args.condition}_idx{i:05d}_{fn_tag}.wav"
            sf.write(os.path.join(LISTEN_DIR, fn), y, sr)
            entries.append({
                "file": fn, "kind": fn_tag, "idx": i, "base": r["base"],
                "condition": args.condition, "text": text, "seed": r["seed"],
                "applied_gain_db": round(float(gdb), 3),
                "peak": round(float(np.abs(y).max()), 4),
                "mean_active_row_cosine_vs_true":
                    None if tag == "true" else round(float(variants[tag][1]), 4)})

        print(f"\nidx {i} ({r['base']}): {text}")
        for tag in ("ecapa", "wavlm_L3_meanstd", "train_mean"):
            p_ = row["probes"][tag]
            print(f"  {tag:<18} cos {p_['mean_active_row_cosine_vs_true']:.4f} "
                  f"(min row {p_['min_active_row_cosine_vs_true']:.4f})  "
                  f"log-mel {p_['logmel_rms_db_vs_true']:.2f} dB "
                  f"(level-matched {p_['logmel_rms_db_vs_true_level_matched']:.2f}, "
                  f"shape {p_['logmel_spectral_shape_db']:.2f})  "
                  f"level {p_['level_db_vs_true']:+.2f} dB")
        print(f"  ecapa vs wavlm  : style cos {row['ecapa_vs_wavlm']['style_active_row_cosine']:.4f}  "
              f"log-mel {row['ecapa_vs_wavlm']['logmel_rms_db']:.2f} dB")
        print(f"  wavlm vs trainmean: style cos "
              f"{row['wavlm_vs_trainmean']['style_active_row_cosine']:.4f}  "
              f"log-mel {row['wavlm_vs_trainmean']['logmel_rms_db']:.2f} dB")

    assert worst_dev < 1e-5, f"per-row unit-norm invariant violated: {worst_dev:.2e}"
    print(f"\nper-row unit-norm invariant on the predicted tensors: "
          f"worst deviation {worst_dev:.2e}")

    manifest = {
        "experiment": "phase2b_probe_recovery_wavlm",
        "condition": args.condition,
        "split": "family-disjoint; test presets " + ",".join(meta["presets_held_out"]),
        "representations": {"ecapa": "192-dim", "wavlm_L3_meanstd": "layer 3, mean+std, 2048-dim",
                            "train_mean": "constant predictor: the training split's mean style, "
                                          "no audio read at all"},
        "inactive_rows": "taken from the true tensor (the probes predict active rows only)",
        "level_matching": "every clip rms-matched to the TRUE render of its own sample",
        "lang": meta["lang"], "total_step": meta["total_step"], "speed": meta["speed"],
        "style_dp": "held fixed at M1's",
        "fit": fit_info, "per_sample": samples, "outputs": entries,
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, "probe_recovery_triples.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(entries)} clips -> {LISTEN_DIR}")


if __name__ == "__main__":
    main()
