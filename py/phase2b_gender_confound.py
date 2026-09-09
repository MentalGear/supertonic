"""Follow-up to phase2b_speaker_similarity.py: separate "recovers gender" (H1)
from "recovers speaker identity beyond gender" (H2).

The coordinator flagged a confound: the train-mean baseline renders as a
male-sounding voice on all three held-out identities, including the two
female ones (F4, F5). So "WavLM-pred sounds like true, train-mean doesn't" on
F4/F5 may be explained entirely by gender (train-mean gets gender wrong),
while the one unconfounded case (M5, where the baseline is also male) is the
one the listener rated weakest. This script adds:

  1. A direct check of the train-mean baseline's gender: ECAPA nearest-preset
     ranking and median F0 against all ten presets.
  2. A same-gender impostor control per held-out sample: sim(render(wrong
     preset of the SAME gender as the true speaker), render(true)), rendered
     with the sample's own text and seed so it is directly comparable to the
     WavLM-pred and train-mean numbers already in
     results/phase2b_speaker_similarity/speaker_similarity_report.json.
  3. A breakdown by true-speaker gender (M5 vs F4+F5 pooled), since the
     baseline's maleness only helps the comparison on the female-true side.

Reuses the WavLM prediction, train-mean prediction and per-sample ECAPA
cosine already computed and cached in speaker_similarity_report.json --
nothing there is re-rendered. Only the TRUE render (needed to score the new
impostor arm) and the impostor render are synthesized here.

Usage (from py/): python3 phase2b_gender_confound.py
"""

import json
import os

import numpy as np

import phase2b_probe as P
import phase2b_wavlm as W
from helper import Style, load_text_to_speech, load_voice_style
from phase2b_prosody_analysis import f0_cents
from phase2b_speaker_similarity import Ecapa, build_variant, cos, to16k

OUT_DIR = "results/phase2b_speaker_similarity"
VOICE_STYLE_DIR = "assets/voice_styles"
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
MALE_IMPOSTORS = ["M1", "M2"]      # wrong-speaker-same-gender controls for M5
FEMALE_IMPOSTORS = ["F1", "F2"]    # wrong-speaker-same-gender controls for F4 / F5
BASELINE_SEEDS = [90000001, 90000003, 90000005]
CONDITION = "eps0.20"


def dist(vals):
    a = np.array(vals, float)
    return {"n": len(a), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "std": round(float(a.std()), 4)}


def main():
    with open(os.path.join(OUT_DIR, "speaker_similarity_report.json")) as f:
        prior = json.load(f)
    prior_by_idx = {row["idx"]: row for row in prior["per_sample"]}

    d = W.load_all()
    meta, recs = d["meta"], d["recs"]
    active, ndim = d["active_rows"], d["ndim"]
    idx, itr, ite, _, _ = W.split_indices(d, CONDITION)
    Y = d["Y_active"]
    ttl_all = d["ttl"]

    print("Loading TTS engine + ECAPA encoder ...", flush=True)
    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    sr = tts.sample_rate
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp
    m1_ttl = load_voice_style(["assets/voice_styles/M1.json"]).ttl
    ecapa = Ecapa()

    X = W.rep_from(d["wavlm_mean"], d["wavlm_std"], [3], "meanstd")
    Pte, _ = P.fit_ridge(X[itr], Y[itr], X[ite])
    train_mean = Y[itr].mean(0)

    def render(text, ttl, seed):
        np.random.seed(seed)
        wav, dur = tts(text, meta["lang"], Style(ttl, dp_ref.copy()), meta["total_step"], meta["speed"])
        return wav[0, : int(sr * dur[0].item())].astype(np.float32)

    preset_ttl = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]).ttl for p in PRESETS}

    # ---------------------------------------------------------------- 1. baseline gender check
    print("\n=== 1. Train-mean baseline: is it male-leaning? ===", flush=True)
    text0 = meta["texts"][0]
    tm_ttl, _ = build_variant(train_mean, m1_ttl.copy(), active, ndim)
    tm_embs, tm_f0 = [], []
    for s in BASELINE_SEEDS:
        w = render(text0, tm_ttl, s)
        w16 = to16k(w, sr)
        tm_embs.append(ecapa.embed(w16))
        f0 = f0_cents(w16)
        tm_f0.append(float(np.nanmedian(f0)))
    tm_emb = np.mean(tm_embs, axis=0)

    preset_embs, preset_f0 = {}, {}
    for p in PRESETS:
        w = render(text0, preset_ttl[p], BASELINE_SEEDS[0])
        w16 = to16k(w, sr)
        preset_embs[p] = ecapa.embed(w16)
        preset_f0[p] = float(np.nanmedian(f0_cents(w16)))
        print(f"  rendered preset {p} for baseline check", flush=True)

    ranked = sorted([(p, round(cos(tm_emb, e), 4)) for p, e in preset_embs.items()], key=lambda x: -x[1])
    male_f0 = np.mean([preset_f0[p] for p in PRESETS if p.startswith("M")])
    female_f0 = np.mean([preset_f0[p] for p in PRESETS if p.startswith("F")])
    baseline_gender = {
        "train_mean_ecapa_cosine_to_each_preset_ranked_desc": ranked,
        "train_mean_nearest_preset": ranked[0][0],
        "train_mean_median_f0_cents_re_55hz_across_3_seeds": [round(x, 1) for x in tm_f0],
        "preset_median_f0_cents_re_55hz": {p: round(v, 1) for p, v in preset_f0.items()},
        "male_preset_mean_f0_cents": round(float(male_f0), 1),
        "female_preset_mean_f0_cents": round(float(female_f0), 1),
        "note": "cents relative to 55 Hz (librosa.pyin median, voiced frames only); "
                "lower cents = lower F0 = more male-typical register",
    }
    print(json.dumps(baseline_gender, indent=2))

    # ---------------------------------------------------------------- 2. per-sample impostor control
    print(f"\n=== 2. Same-gender impostor control ({CONDITION}, {len(ite)} samples) ===", flush=True)
    rows = []
    for k, i in enumerate(ite):
        i = int(i)
        r = recs[i]
        text = meta["texts"][r["text_idx"]]
        gender = "M" if r["base"].startswith("M") else "F"
        impostors = MALE_IMPOSTORS if gender == "M" else FEMALE_IMPOSTORS
        impostor_preset = impostors[k % len(impostors)]

        true_ttl = ttl_all[i][None].astype(np.float32).copy()
        true_w = render(text, true_ttl, r["seed"])
        imp_w = render(text, preset_ttl[impostor_preset], r["seed"])
        true_emb = ecapa.embed(to16k(true_w, sr))
        imp_emb = ecapa.embed(to16k(imp_w, sr))

        p = prior_by_idx[i]
        row = {
            "idx": i, "base": r["base"], "gender": gender, "impostor_preset": impostor_preset,
            "ecapa_cosine_wavlm_vs_true": p["wavlm"]["ecapa_cosine_vs_true"],
            "ecapa_cosine_trainmean_vs_true": p["train_mean"]["ecapa_cosine_vs_true"],
            "ecapa_cosine_impostor_vs_true": round(cos(imp_emb, true_emb), 4),
        }
        row["wavlm_minus_trainmean"] = round(row["ecapa_cosine_wavlm_vs_true"] - row["ecapa_cosine_trainmean_vs_true"], 4)
        row["wavlm_minus_impostor"] = round(row["ecapa_cosine_wavlm_vs_true"] - row["ecapa_cosine_impostor_vs_true"], 4)
        rows.append(row)
        print(f"  [{k+1:3d}/{len(ite)}] idx {i:5d} ({r['base']}, impostor {impostor_preset}): "
              f"wavlm={row['ecapa_cosine_wavlm_vs_true']:.3f} "
              f"trainmean={row['ecapa_cosine_trainmean_vs_true']:.3f} "
              f"impostor={row['ecapa_cosine_impostor_vs_true']:.3f}  "
              f"(wavlm-trainmean={row['wavlm_minus_trainmean']:+.3f}, "
              f"wavlm-impostor={row['wavlm_minus_impostor']:+.3f})", flush=True)

    # ---------------------------------------------------------------- 3. gender-split summary
    print("\n=== 3. Breakdown by true-speaker gender ===", flush=True)
    groups = {
        "M5 (male-true, unconfounded)": [r for r in rows if r["base"] == "M5"],
        "F4+F5 (female-true, baseline-gender-confounded)": [r for r in rows if r["base"] in ("F4", "F5")],
        "all": rows,
    }
    summary = {}
    for name, g in groups.items():
        summary[name] = {
            "n": len(g),
            "ecapa_cosine_wavlm_vs_true": dist([r["ecapa_cosine_wavlm_vs_true"] for r in g]),
            "ecapa_cosine_trainmean_vs_true": dist([r["ecapa_cosine_trainmean_vs_true"] for r in g]),
            "ecapa_cosine_impostor_vs_true": dist([r["ecapa_cosine_impostor_vs_true"] for r in g]),
            "wavlm_minus_trainmean": dist([r["wavlm_minus_trainmean"] for r in g]),
            "wavlm_minus_impostor": dist([r["wavlm_minus_impostor"] for r in g]),
            "frac_wavlm_beats_impostor": round(float(np.mean([r["wavlm_minus_impostor"] > 0 for r in g])), 3),
            "frac_wavlm_beats_trainmean": round(float(np.mean([r["wavlm_minus_trainmean"] > 0 for r in g])), 3),
        }
    print(json.dumps(summary, indent=2))

    out = {
        "experiment": "phase2b_gender_confound",
        "condition": CONDITION,
        "hypotheses": {
            "H1_weak": "probe recovers coarse gender only; train-mean loses mainly because it "
                       "happens to land male, not because wavlm recovers identity",
            "H2_strong": "probe recovers speaker identity beyond gender",
            "diagnostic": "if wavlm's advantage over a SAME-GENDER impostor is small/zero, that is H1; "
                          "if wavlm reliably beats a same-gender impostor too, that supports H2",
        },
        "baseline_gender_check": baseline_gender,
        "per_sample": rows,
        "gender_split_summary": summary,
    }
    with open(os.path.join(OUT_DIR, "gender_confound_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.join(OUT_DIR, 'gender_confound_report.json')}")


if __name__ == "__main__":
    main()
