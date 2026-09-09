"""Phase 2b re-examination: does the WavLM-predicted style carry the true
speaker's identity, measured with a speaker-verification embedding instead of
an uncalibrated log-mel/cosine number?

Motivation (see the top of new-plan.md's request for this script, and the
Phase 2b record): the record closes Phase 2b on the claim that both probes'
reconstructions "sit 14-21 dB from the true style, as far as or further than a
different speaker", calibrated against one M1->F1 identity swap at 16.6 dB, and
that a constant train-mean predictor trails WavLM by only 0.017-0.028
active-row cosine. A human listener reports the opposite: the WavLM prediction
sounds like the true voice (clearly on F5, close on M5), while the train-mean
baseline never does. Active-row style cosine and rms log-mel distance are not
speaker-identity metrics -- they were never calibrated against known same- and
different-speaker pairs. This script does that calibration, using ECAPA for
the job it is actually built for (speaker verification), and checks the three
numbers (ECAPA cosine, style cosine, log-mel distance) against each other.

Reuses: fitted-in-place ridge probes via phase2b_probe.fit_ridge (identical to
phase2b_wavlm_render.py's protocol), styles.npz, wavlm_feats.npz,
embeddings.npz-adjacent ECAPA encoder, and the render path from
phase2b_wavlm_render.py. Nothing is re-extracted or regenerated except the
renders needed for this measurement (~380 clips at ~1 s/clip).

Usage (from py/):
    python3 phase2b_speaker_similarity.py [--condition eps0.20] [--n-bench 3]
"""

import argparse
import itertools
import json
import os

import numpy as np
import soundfile as sf
import torch

import phase2b_probe as P
import phase2b_wavlm as W
from helper import Style, load_text_to_speech, load_voice_style
from phase2b_prosody_analysis import rms
from phase2b_prosody_mel import logmel

OUT_DIR = "results/phase2b_speaker_similarity"
LISTEN_DIR = "results/listening_sets/phase2b_speaker_similarity"
VOICE_STYLE_DIR = "assets/voice_styles"
ECAPA_CACHE = "/home/user/.cache/ecapa"
WAVLM_LAYERS, WAVLM_POOLING = [3], "meanstd"
LOUD_DB = 25.0
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
CAL_SEED_BASE = 90000001   # calibration renders: base text, base seed
CAL_SEED_ALT = 90000002    # same style, different vocoder seed
CAL_TEXT0_IDX = 0          # "The quick brown fox ..."
CAL_TEXT1_IDX = 5          # "Nine hungry travellers ..." (different sentence)


def to16k(w, sr):
    from scipy.signal import resample_poly
    return resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w.astype(np.float32)


def ltas(mel):
    """Long-term average spectrum: per-mel-band mean over time, dB domain."""
    return mel.mean(1)


def level_match(w, ref_rms):
    r = float(np.sqrt((w.astype(np.float64) ** 2).mean()))
    g = ref_rms / max(r, 1e-12)
    return (w * g).astype(np.float32), 20 * np.log10(g)


class Ecapa:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=ECAPA_CACHE,
            run_opts={"device": "cpu"})
        self.enc.eval()

    def embed(self, wav16k):
        with torch.no_grad():
            e = self.enc.encode_batch(torch.from_numpy(wav16k)[None, :])
        return e.squeeze().numpy().astype(np.float64)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_variant(pred_row_block, true_ttl, active, ndim):
    t = true_ttl.copy()
    rows = pred_row_block.reshape(len(active), ndim).astype(np.float32)
    rows = rows / np.linalg.norm(rows, axis=-1, keepdims=True).clip(min=1e-8)
    t[0, active, :] = rows
    dev = float(np.abs(np.linalg.norm(t, axis=-1) - 1.0).max())
    return t, dev


def style_cos(a_ttl, b_ttl, active):
    A, B = a_ttl[0, active, :], b_ttl[0, active, :]
    c = (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12)
    return float(c.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="eps0.20")
    ap.add_argument("--n-bench", type=int, default=3, help="best/typical/worst per identity")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of held-out test samples")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)

    print("Loading data / fitting probes ...", flush=True)
    d = W.load_all()
    meta, recs = d["meta"], d["recs"]
    active, ndim = d["active_rows"], d["ndim"]
    idx, itr, ite, _, _ = W.split_indices(d, args.condition)
    if args.limit:
        ite = ite[: args.limit]
    Y = d["Y_active"]

    reps = {"ecapa": d["ecapa"],
            "wavlm": W.rep_from(d["wavlm_mean"], d["wavlm_std"], WAVLM_LAYERS, WAVLM_POOLING)}
    preds, fit_info = {}, {}
    for name, X in reps.items():
        Pte, alpha = P.fit_ridge(X[itr], Y[itr], X[ite])
        ev = P.evaluate(Y[itr], Y[ite], Pte, active, ndim)
        preds[name] = Pte
        fit_info[name] = {"dim": int(X.shape[1]), "alpha": alpha, "n_train": int(len(itr)),
                          "n_test": int(len(ite)),
                          "family_disjoint_r2": ev["r2_trainmean_baseline"],
                          "mean_active_row_cosine": ev["mean_row_cosine_pred_vs_true"]}
        print(f"  {name:<6} R^2 {ev['r2_trainmean_baseline']:+.4f}  "
              f"row-cos {ev['mean_row_cosine_pred_vs_true']:.4f}", flush=True)
    train_mean = Y[itr].mean(0)

    print("Loading TTS engine + ECAPA encoder ...", flush=True)
    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    sr = tts.sample_rate
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp
    ttl_all = d["ttl"]
    ecapa = Ecapa()

    def render(text, style_ttl, seed):
        np.random.seed(seed)
        wav, dur = tts(text, meta["lang"], Style(style_ttl, dp_ref.copy()),
                       meta["total_step"], meta["speed"])
        return wav[0, : int(sr * dur[0].item())].astype(np.float32)

    # ---------------------------------------------------------------- main test
    print(f"\n=== Main test: condition {args.condition}, {len(ite)} held-out samples ===",
          flush=True)
    per_sample = []
    for k, i in enumerate(ite):
        i = int(i)
        r = recs[i]
        text = meta["texts"][r["text_idx"]]
        true_ttl = ttl_all[i][None].astype(np.float32).copy()

        variants = {"true": true_ttl}
        style_cosines = {}
        for name in reps:
            t, _ = build_variant(preds[name][k], true_ttl, active, ndim)
            variants[name] = t
            style_cosines[name] = style_cos(true_ttl, t, active)
        t, _ = build_variant(train_mean, true_ttl, active, ndim)
        variants["train_mean"] = t
        style_cosines["train_mean"] = style_cos(true_ttl, t, active)

        wavs = {tag: render(text, t, r["seed"]) for tag, t in variants.items()}
        assert len(set(w.size for w in wavs.values())) == 1, "clips not frame-aligned"
        emb = {tag: ecapa.embed(to16k(w, sr)) for tag, w in wavs.items()}

        mel = {tag: logmel(to16k(w, sr)) for tag, w in wavs.items()}
        nf = min(m.shape[1] for m in mel.values())
        env = mel["true"][:, :nf].mean(0)
        loud = env > env.max() - LOUD_DB
        ref_rms = float(np.sqrt((wavs["true"].astype(np.float64) ** 2).mean()))

        row = {"idx": i, "base": r["base"], "text": text, "seed": r["seed"]}
        for tag in ("ecapa", "wavlm", "train_mean"):
            lvl_w, gdb = level_match(wavs[tag], ref_rms)
            D = mel[tag][:, :nf] - mel["true"][:, :nf] - gdb
            row[tag] = {
                "ecapa_cosine_vs_true": round(cos(emb[tag], emb["true"]), 4),
                "style_active_row_cosine_vs_true": round(style_cosines[tag], 4),
                "logmel_rms_db_vs_true_level_matched": round(rms(D[:, loud]), 3),
            }
        per_sample.append(row)
        print(f"  [{k+1:3d}/{len(ite)}] idx {i:5d} ({r['base']}): "
              f"ecapa-cos ecapa={row['ecapa']['ecapa_cosine_vs_true']:.3f} "
              f"wavlm={row['wavlm']['ecapa_cosine_vs_true']:.3f} "
              f"trainmean={row['train_mean']['ecapa_cosine_vs_true']:.3f}", flush=True)

    # ---------------------------------------------------------------- calibration
    print("\n=== Calibration anchors: same-/different-speaker pairs (10 presets) ===",
          flush=True)
    preset_ttl = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]).ttl
                  for p in PRESETS}
    text0 = meta["texts"][CAL_TEXT0_IDX]
    text1 = meta["texts"][CAL_TEXT1_IDX]

    cal_wavs = {"base": {}, "seed2": {}, "text2": {}}
    for p in PRESETS:
        cal_wavs["base"][p] = render(text0, preset_ttl[p], CAL_SEED_BASE)
        cal_wavs["seed2"][p] = render(text0, preset_ttl[p], CAL_SEED_ALT)
        cal_wavs["text2"][p] = render(text1, preset_ttl[p], CAL_SEED_BASE)
        print(f"  rendered {p}", flush=True)

    cal_emb = {grp: {p: ecapa.embed(to16k(w, sr)) for p, w in g.items()}
               for grp, g in cal_wavs.items()}
    cal_mel = {grp: {p: logmel(to16k(w, sr)) for p, w in g.items()}
               for grp, g in cal_wavs.items()}

    def frame_aligned_logmel_db(m1, w1, m2, w2):
        nf = min(m1.shape[1], m2.shape[1])
        ref_rms = float(np.sqrt((w1.astype(np.float64) ** 2).mean()))
        lvl_w2, gdb = level_match(w2, ref_rms)
        env = m1[:, :nf].mean(0)
        loud = env > env.max() - LOUD_DB
        D = m2[:, :nf] - m1[:, :nf] - gdb
        return round(rms(D[:, loud]), 3)

    def ltas_db_dist(m1, m2):
        return round(rms(ltas(m1) - ltas(m2)), 3)

    diff_speaker = []  # 45 pairs, same text0/seed_base -> frame-aligned
    for p, q in itertools.combinations(PRESETS, 2):
        diff_speaker.append({
            "pair": f"{p}-{q}",
            "ecapa_cosine": round(cos(cal_emb["base"][p], cal_emb["base"][q]), 4),
            "style_active_row_cosine": round(style_cos(preset_ttl[p], preset_ttl[q], active), 4),
            "logmel_rms_db_frame_aligned": frame_aligned_logmel_db(
                cal_mel["base"][p], cal_wavs["base"][p], cal_mel["base"][q], cal_wavs["base"][q]),
        })

    same_speaker_diff_seed = []  # 10 pairs, same text -> frame-aligned
    for p in PRESETS:
        same_speaker_diff_seed.append({
            "pair": f"{p}(seed A)-{p}(seed B)",
            "ecapa_cosine": round(cos(cal_emb["base"][p], cal_emb["seed2"][p]), 4),
            "logmel_rms_db_frame_aligned": frame_aligned_logmel_db(
                cal_mel["base"][p], cal_wavs["base"][p], cal_mel["seed2"][p], cal_wavs["seed2"][p]),
        })

    same_speaker_diff_text = []  # 10 pairs, different text -> NOT frame-aligned, use LTAS
    for p in PRESETS:
        same_speaker_diff_text.append({
            "pair": f"{p}(text0)-{p}(text1)",
            "ecapa_cosine": round(cos(cal_emb["base"][p], cal_emb["text2"][p]), 4),
            "logmel_ltas_db": ltas_db_dist(cal_mel["base"][p], cal_mel["text2"][p]),
        })

    m1f1 = next(r for r in diff_speaker if r["pair"] == "M1-F1")

    def dist(vals):
        a = np.array(vals, float)
        return {"n": len(a), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                "std": round(float(a.std()), 4)}

    calibration = {
        "different_speaker_pairs_frame_aligned": diff_speaker,
        "same_speaker_diff_seed_pairs_frame_aligned": same_speaker_diff_seed,
        "same_speaker_diff_text_pairs_LTAS": same_speaker_diff_text,
        "m1_f1_reference_pair": m1f1,
        "distributions": {
            "ecapa_cosine": {
                "different_speaker (45 preset pairs)": dist([r["ecapa_cosine"] for r in diff_speaker]),
                "same_speaker_diff_seed (10 pairs)": dist([r["ecapa_cosine"] for r in same_speaker_diff_seed]),
                "same_speaker_diff_text (10 pairs)": dist([r["ecapa_cosine"] for r in same_speaker_diff_text]),
            },
            "logmel_rms_db_frame_aligned": {
                "different_speaker (45 preset pairs)": dist([r["logmel_rms_db_frame_aligned"] for r in diff_speaker]),
                "same_speaker_diff_seed (10 pairs)": dist([r["logmel_rms_db_frame_aligned"] for r in same_speaker_diff_seed]),
            },
            "logmel_ltas_db": {
                "same_speaker_diff_text (10 pairs)": dist([r["logmel_ltas_db"] for r in same_speaker_diff_text]),
            },
            "style_active_row_cosine": {
                "different_speaker (45 preset pairs)": dist([r["style_active_row_cosine"] for r in diff_speaker]),
            },
        },
    }
    print(json.dumps(calibration["distributions"], indent=2))

    # ---------------------------------------------------------------- correlation
    print("\n=== Correlating ECAPA cosine against style cosine and log-mel distance ===",
          flush=True)
    ec_all, sc_all, lm_all, tag_all = [], [], [], []
    for row in per_sample:
        for tag in ("ecapa", "wavlm", "train_mean"):
            ec_all.append(row[tag]["ecapa_cosine_vs_true"])
            sc_all.append(row[tag]["style_active_row_cosine_vs_true"])
            lm_all.append(row[tag]["logmel_rms_db_vs_true_level_matched"])
            tag_all.append(tag)
    for r in diff_speaker:
        ec_all.append(r["ecapa_cosine"]); sc_all.append(r["style_active_row_cosine"])
        lm_all.append(r["logmel_rms_db_frame_aligned"]); tag_all.append("cal_diff_speaker")
    for r in same_speaker_diff_seed:
        ec_all.append(r["ecapa_cosine"]); sc_all.append(1.0)  # identical style tensor
        lm_all.append(r["logmel_rms_db_frame_aligned"]); tag_all.append("cal_same_speaker")

    ec_a, sc_a, lm_a = np.array(ec_all), np.array(sc_all), np.array(lm_all)

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return pearson(ra, rb)

    correlation = {
        "n_pairs": len(ec_all),
        "pearson_ecapa_vs_style_cosine": round(pearson(ec_a, sc_a), 4),
        "spearman_ecapa_vs_style_cosine": round(spearman(ec_a, sc_a), 4),
        "pearson_ecapa_vs_logmel_db": round(pearson(ec_a, lm_a), 4),
        "spearman_ecapa_vs_logmel_db": round(spearman(ec_a, lm_a), 4),
        "pearson_style_cosine_vs_logmel_db": round(pearson(sc_a, lm_a), 4),
        "note": "ecapa_cosine is the speaker-identity ground truth here; the other two "
                "are the metrics the 2b record relied on. A low |correlation| means they "
                "were not tracking speaker identity.",
    }
    print(json.dumps(correlation, indent=2))

    # ---------------------------------------------------------------- headline summary
    def summarize_pred(tag, base_filter=None):
        rows = per_sample if base_filter is None else [r for r in per_sample if r["base"] == base_filter]
        return {
            "n": len(rows),
            "ecapa_cosine_vs_true": dist([r[tag]["ecapa_cosine_vs_true"] for r in rows]),
            "style_active_row_cosine_vs_true": dist([r[tag]["style_active_row_cosine_vs_true"] for r in rows]),
            "logmel_rms_db_vs_true": dist([r[tag]["logmel_rms_db_vs_true_level_matched"] for r in rows]),
        }

    identities = sorted(set(r["base"] for r in per_sample))
    headline = {
        "condition": args.condition,
        "overall": {tag: summarize_pred(tag) for tag in ("ecapa", "wavlm", "train_mean")},
        "per_identity": {ident: {tag: summarize_pred(tag, ident) for tag in ("ecapa", "wavlm", "train_mean")}
                         for ident in identities},
    }
    print("\n=== HEADLINE ===")
    print(json.dumps(headline["overall"], indent=2))
    for ident in identities:
        print(f"\n-- {ident} --")
        print(json.dumps(headline["per_identity"][ident], indent=2))

    report = {
        "experiment": "phase2b_speaker_similarity",
        "condition": args.condition,
        "fit": fit_info,
        "headline": headline,
        "per_sample": per_sample,
        "calibration": calibration,
        "correlation": correlation,
    }
    with open(os.path.join(OUT_DIR, "speaker_similarity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {os.path.join(OUT_DIR, 'speaker_similarity_report.json')}")

    # ---------------------------------------------------------------- listening bench
    print("\n=== Building listening bench (best/typical/worst per identity, by WavLM ECAPA-cos) ===",
          flush=True)
    bench_entries = []
    manifest_rows = []
    for ident in identities:
        rows = sorted([r for r in per_sample if r["base"] == ident],
                      key=lambda r: r["wavlm"]["ecapa_cosine_vs_true"])
        n = len(rows)
        picks = {"worst": rows[0], "typical": rows[n // 2], "best": rows[-1]}
        if args.n_bench > 3:
            # add extra evenly spaced samples
            extra_idx = np.linspace(0, n - 1, args.n_bench).astype(int)
            picks = {f"p{i}": rows[j] for i, j in enumerate(extra_idx)}
        for label, row in picks.items():
            i = row["idx"]
            r = recs[i]
            text = meta["texts"][r["text_idx"]]
            true_ttl = ttl_all[i][None].astype(np.float32).copy()
            k = list(ite).index(i)
            wavlm_t, _ = build_variant(preds["wavlm"][k], true_ttl, active, ndim)
            tm_t, _ = build_variant(train_mean, true_ttl, active, ndim)
            variants = {"true": true_ttl, "wavlm_pred": wavlm_t, "trainmean_baseline": tm_t}
            wavs = {tag: render(text, t, r["seed"]) for tag, t in variants.items()}
            ref_rms = float(np.sqrt((wavs["true"].astype(np.float64) ** 2).mean()))
            for tag, w in wavs.items():
                lvl_w, gdb = level_match(w, ref_rms)
                fn = f"{ident}_{label}_idx{i:05d}_{tag}.wav"
                sf.write(os.path.join(LISTEN_DIR, fn), lvl_w, sr)
                bench_entries.append({"file": fn, "identity": ident, "rank_label": label,
                                      "idx": i, "kind": tag, "text": text,
                                      "ecapa_cosine_vs_true": (None if tag == "true" else
                                          row["wavlm" if tag == "wavlm_pred" else "train_mean"]["ecapa_cosine_vs_true"])})
            manifest_rows.append({"identity": ident, "rank_label": label, "idx": i, "base": ident,
                                  "text": text, "seed": r["seed"],
                                  "wavlm_ecapa_cosine_vs_true": row["wavlm"]["ecapa_cosine_vs_true"],
                                  "trainmean_ecapa_cosine_vs_true": row["train_mean"]["ecapa_cosine_vs_true"],
                                  "wavlm_logmel_db": row["wavlm"]["logmel_rms_db_vs_true_level_matched"],
                                  "trainmean_logmel_db": row["train_mean"]["logmel_rms_db_vs_true_level_matched"]})
            print(f"  {ident}/{label}: idx {i}  wavlm-cos {row['wavlm']['ecapa_cosine_vs_true']:.3f}  "
                  f"trainmean-cos {row['train_mean']['ecapa_cosine_vs_true']:.3f}", flush=True)

    bench_manifest = {
        "experiment": "phase2b_speaker_similarity_bench",
        "condition": args.condition,
        "lang": meta["lang"], "total_step": meta["total_step"], "speed": meta["speed"],
        "style_dp": "held fixed at M1's",
        "level_matching": "every clip rms-matched to its own TRUE render",
        "rows": manifest_rows,
        "outputs": bench_entries,
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(bench_manifest, f, indent=2)
    print(f"\nWrote {len(bench_entries)} bench clips -> {LISTEN_DIR}")


if __name__ == "__main__":
    main()
