"""Phase 2a -- is sibilant over-drive controllable through `style_ttl`, and if
so, is there a de-essing direction?

`phase2a_sibilance.py` computed a suggestive, not validated, measure of the
artifact: the ratio of peak sample amplitude inside sibilant frames to the
whole clip's peak. The three clips it labels flagged do rank 1st-3rd of 20 at
0.9751-1.0000, but the margin is thin, not a clean separation: the highest
strict-clean clip (pool_M4_t4) scores 0.9587, so flagged-min beats clean-max
by only 0.0164, not the ~0.33 gap "clean clips top out at 0.646" implied. The
flagged/clean labels are also not the listener's own sibilance remarks --
they are a 3-flagged/14-clean partition of the 20 clips (3 excluded), and at
least one clip with a sibilance remark (clip07, rated clean overall) falls
outside the flagged set. This script reuses that measure exactly
(`phase2a_sibilance.analyze_clip`, unmodified) and
asks whether it moves under `style_ttl` perturbation, and if so, whether a
ridge-fitted linear direction can lower it without turning the preset into a
different speaker.

Worst case per the brief: preset F3, "The library closes early on Thursday,
so bring your books back.", speed=1.0 (this fork's default), TOTAL_STEP=8.

Method (see module docstring sections below for the corresponding step):
  1. Render N_STEP1 renders of F3 perturbed along random directions drawn in
     the tangent space at F3 (`phase2b_generate_subspace.build_basis` +
     `sample_style`, called with F3's own ttl as the base point, not M1).
     Measure the sibilant ratio on each. Report the spread.
  2. Fit a ridge model of ratio ~ perturbation coefficients
     (`phase2b_probe.fit_ridge`/`ALPHAS`), report held-out R^2. Only proceed
     if it generalizes.
  3. Step along the negative gradient at several magnitudes; at each, render
     with a FIXED vocoder seed and FIXED style_dp (so duration/alignment is
     pinned), measure ratio, absolute sibilant-frame peak, whole-clip peak,
     RMS, and ECAPA cosine to unperturbed F3 across several seeds.
  4. Report explicitly whether the numerator (sibilant peak) or the
     denominator (whole-clip peak) moved.
  5. Apply the best direction unchanged to F5/seashells and to F3 on a
     different sentence; report both transfer results.

Usage (from py/):
    python3 phase2a_deessing_direction.py [--smoke]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper import Style, load_text_to_speech, load_voice_style  # noqa: E402
from phase2a_sibilance import analyze_clip  # noqa: E402 -- reused unmodified
from phase2b_generate import ACTIVE_ROWS, LANG, ONNX_DIR, TEXTS, VOICE_STYLE_DIR  # noqa: E402
from phase2b_generate_subspace import build_basis, sample_style  # noqa: E402
from phase2b_probe import ALPHAS, agg_r2, fit_ridge, r2_parts  # noqa: E402
from phase2b_speaker_similarity import Ecapa, cos  # noqa: E402

OUT_DIR = "results/phase2a_deessing"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
REPORT_JSON = os.path.join(OUT_DIR, "report.json")
CALIBRATION_JSON = "results/phase2b_speaker_similarity/speaker_similarity_report.json"

SPEED = 1.0  # this fork's default; passed explicitly per the brief
TOTAL_STEP = 8
BASE_PRESET = "F3"
LIBRARY_TEXT_IDX = 4
assert TEXTS[LIBRARY_TEXT_IDX].startswith("The library closes early")
ALT_SENTENCE_TEXT_IDX = 2  # "Please call the doctor ... starts at noon." (has sibilants)
assert "starts" in TEXTS[ALT_SENTENCE_TEXT_IDX]
SEASHELLS_TEXT_IDX = 1
assert TEXTS[SEASHELLS_TEXT_IDX].startswith("She sells seashells")

K = 64                      # subspace dimension for the ridge probe
EPS = 0.15                  # "modest" perturbation magnitude for step 1
N_STEP1 = 120
TEST_FRAC = 0.25
BASIS_SEED = 20270301
SAMPLE_SEED = 20270302
DEESSING_VOCODER_SEED_BASE = 20270310  # step-1 renders: BASE + i

LADDER_MAGNITUDES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
LADDER_SEED = 20270310          # == DEESSING_VOCODER_SEED_BASE + 0 (i=0's seed);
                                 # reused deliberately so ladder mag=0.00 IS step-1's
                                 # very first (unperturbed-direction-scaled-to-0) render
ANCHOR_SEEDS = [20270310, 20270311, 20270312, 20270313]  # unperturbed F3, 4 vocoder seeds

R2_GO_THRESHOLD = 0.10           # held-out R^2 must clear this to trust the direction
SPEAKER_COSINE_FLOOR = 0.70      # below the calibrated same-speaker-diff-seed min (0.7843)
                                  # is treated as "no longer reads as F3"


def unit_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def apply_direction(P, direction_unit_flat, magnitude, n_rows=24):
    """Same scaling convention as phase2b_generate_subspace.sample_style:
    target_norm = magnitude * sqrt(n_rows), applied deterministically (no RNG)."""
    d = direction_unit_flat.reshape(P.shape) * (magnitude * np.sqrt(n_rows))
    return unit_rows(P + d)


def build_ttl(base_ttl, rows):
    ttl = base_ttl.copy()
    ttl[0, ACTIVE_ROWS, :] = rows.astype(np.float32)
    return ttl


def render(tts, text, ttl, dp, seed):
    np.random.seed(seed)
    wav, dur = tts(text, LANG, Style(ttl, dp.copy()), TOTAL_STEP, SPEED)
    trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
    return trimmed, tts.sample_rate


def write_and_analyze(model, trimmed, sr, stem):
    path = os.path.join(AUDIO_DIR, f"{stem}.wav")
    sf.write(path, trimmed, sr, subtype="PCM_16")
    result = analyze_clip(model, path)  # reused, unmodified
    ratio = result["sibilant_frames"]["peak"] / max(result["whole_clip"]["peak"], 1e-12)
    rms = float(np.sqrt((trimmed.astype(np.float64) ** 2).mean()))
    return dict(
        path=path, ratio=float(ratio),
        sibilant_peak=result["sibilant_frames"]["peak"],
        other_peak=result["other_frames"]["peak"],
        whole_peak=result["whole_clip"]["peak"],
        rms=rms,
        n_sibilant_frames=result["n_sibilant_frames"],
        n_words=len(result["words"]),
    )


def fit_full_gradient(X, y):
    """Same StandardScaler+RidgeCV mechanics as phase2b_probe.fit_ridge, but
    returns the fitted gradient in raw coefficient space instead of held-out
    predictions -- fit_ridge itself has no hook for that."""
    sc = StandardScaler().fit(X)
    model = RidgeCV(alphas=ALPHAS)
    model.fit(sc.transform(X), y)
    grad = model.coef_ / sc.scale_
    return grad, float(model.alpha_)


def to16k(w, sr):
    from scipy.signal import resample_poly
    return resample_poly(w, 16000, sr).astype(np.float32) if sr != 16000 else w.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny N for a pipeline smoke test")
    args = ap.parse_args()

    n1 = 6 if args.smoke else N_STEP1
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("Loading faster-whisper tiny.en ...", flush=True)
    from faster_whisper import WhisperModel
    wmodel = WhisperModel("Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8",
                           local_files_only=True)

    print("Loading TTS engine + F3 style ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    base_style = load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{BASE_PRESET}.json")])
    base_ttl = base_style.ttl.astype(np.float32)
    dp_ref = base_style.dp.copy()

    B, P = build_basis(base_ttl, n_dims=K, seed=BASIS_SEED)
    off_diag = float(np.abs(B @ B.T - np.eye(K)).max())
    print(f"basis orthonormality max|BB^T-I| = {off_diag:.2e}")

    library_text = TEXTS[LIBRARY_TEXT_IDX]

    # =================================================================== 1
    print(f"\n=== Step 1: {n1} random-direction renders, eps={EPS} ===", flush=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    step1 = []
    for i in range(n1):
        rows, c_drawn, c_realized, frac = sample_style(rng, B, P, K, EPS)
        ttl = build_ttl(base_ttl, rows)
        seed = DEESSING_VOCODER_SEED_BASE + i
        wav, sr = render(tts, library_text, ttl, dp_ref, seed)
        stem = f"s1_{i:04d}_F3_seed{seed}"
        m = write_and_analyze(wmodel, wav, sr, stem)
        m.update(i=i, seed=seed, c_drawn=c_drawn.tolist(), in_subspace_fraction=frac)
        step1.append(m)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n1}] ratio={m['ratio']:.4f} sib_peak={m['sibilant_peak']:.4f}", flush=True)

    ratios1 = np.array([r["ratio"] for r in step1])
    spread = dict(
        n=len(ratios1), mean=float(ratios1.mean()), std=float(ratios1.std()),
        min=float(ratios1.min()), max=float(ratios1.max()),
        p10=float(np.percentile(ratios1, 10)), p50=float(np.percentile(ratios1, 50)),
        p90=float(np.percentile(ratios1, 90)),
    )
    print(f"Step-1 ratio spread: {spread}")

    report = dict(
        experiment="phase2a_deessing_direction",
        date=datetime.date.today().isoformat(),
        base_preset=BASE_PRESET, text=library_text, speed=SPEED, total_step=TOTAL_STEP,
        eps=EPS, k=K, n_step1=n1,
        step1_spread=spread,
    )

    # gate: does it move at all? use a simple, pre-declared bar -- a spread
    # (max-min) below 0.05 (an order of magnitude below the flagged/clean gap
    # of ~0.33-0.6 in phase2a_sibilance) is treated as "does not move".
    moves_at_all = (spread["max"] - spread["min"]) > 0.05
    report["moves_at_all"] = moves_at_all
    if not moves_at_all:
        report["verdict"] = ("style_ttl perturbation at eps=%.2f does not move the sibilant "
                              "ratio materially (spread %.4f); NOT controllable through this "
                              "surface at this magnitude -- stopping here." % (EPS, spread["max"] - spread["min"]))
        print("\n" + report["verdict"])
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        return

    # =================================================================== 2
    print("\n=== Step 2: ridge fit, ratio ~ perturbation coefficients ===", flush=True)
    X = np.array([r["c_drawn"] for r in step1])
    y = ratios1
    perm = np.random.default_rng(SAMPLE_SEED + 1).permutation(n1)
    n_test = max(1, int(round(n1 * TEST_FRAC)))
    ite, itr = perm[:n_test], perm[n_test:]
    Pte, alpha = fit_ridge(X[itr], y[itr], X[ite])
    ss_res, ss_tot = r2_parts(y[ite], Pte, y[itr].mean(0))
    r2_heldout = agg_r2(ss_res, ss_tot)
    print(f"held-out R^2 (train-mean baseline) = {r2_heldout:.4f} (alpha={alpha:.3g}, "
          f"n_train={len(itr)}, n_test={len(ite)})")
    report["ridge_fit"] = dict(r2_heldout_trainmean_baseline=float(r2_heldout), alpha=alpha,
                                n_train=int(len(itr)), n_test=int(len(ite)))

    if r2_heldout <= R2_GO_THRESHOLD:
        report["direction_usable"] = False
        report["verdict"] = ("style_ttl moves the ratio (spread %.4f) but the ridge fit does "
                              "not generalize (held-out R^2=%.4f <= %.2f threshold): the "
                              "'direction' would be noise. Stopping -- no direction is used."
                              % (spread["max"] - spread["min"], r2_heldout, R2_GO_THRESHOLD))
        print("\n" + report["verdict"])
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        return

    report["direction_usable"] = True
    grad, alpha_full = fit_full_gradient(X, y)
    direction = -grad / np.linalg.norm(grad)  # unit vector in c-space, descent direction
    direction_flat = direction @ B[:K]        # unit vector in the 24x256 tangent space
    direction_flat = direction_flat / np.linalg.norm(direction_flat)
    report["gradient_fit_alpha_full_data"] = alpha_full

    # =================================================================== 3
    print("\n=== Step 3: ladder + speaker-identity controls ===", flush=True)
    anchors = []
    for seed in ANCHOR_SEEDS:
        stem = f"anchor_F3_seed{seed}"
        if seed == LADDER_SEED:
            continue  # rendered as ladder mag=0.00 below; avoid duplicate render
        wav, sr = render(tts, library_text, base_ttl, dp_ref, seed)
        m = write_and_analyze(wmodel, wav, sr, stem)
        m.update(seed=seed)
        anchors.append(m)

    ecapa = Ecapa()
    anchor_embs = []

    ladder = []
    for mag in LADDER_MAGNITUDES:
        rows = apply_direction(P, direction_flat, mag)
        ttl = base_ttl.copy() if mag == 0.0 else build_ttl(base_ttl, rows)
        seed = LADDER_SEED
        wav, sr = render(tts, library_text, ttl, dp_ref, seed)
        stem = f"ladder_mag{mag:.3f}_F3_seed{seed}"
        m = write_and_analyze(wmodel, wav, sr, stem)
        emb = ecapa.embed(to16k(wav, sr))
        m.update(mag=mag, seed=seed)
        ladder.append((m, emb, wav, sr))
        if mag == 0.0:
            anchors.append(dict(seed=seed, ratio=m["ratio"], sibilant_peak=m["sibilant_peak"],
                                 other_peak=m["other_peak"], whole_peak=m["whole_peak"],
                                 rms=m["rms"], path=m["path"]))
            anchor_embs.append(emb)

    for a in anchors:
        if "path" in a and len(anchor_embs) < len(ANCHOR_SEEDS):
            w, sr2 = sf.read(a["path"], dtype="float32")
            anchor_embs.append(ecapa.embed(to16k(w, sr2)))

    ladder_report = []
    for m, emb, wav, sr in ladder:
        cosines = [cos(emb, ae) for ae in anchor_embs]
        entry = dict(m)
        entry["ecapa_cosine_to_anchors_mean"] = float(np.mean(cosines))
        entry["ecapa_cosine_to_anchors_min"] = float(np.min(cosines))
        entry["ecapa_cosine_to_anchors_all"] = [float(c) for c in cosines]
        ladder_report.append(entry)
        print(f"  mag={m['mag']:.2f} ratio={m['ratio']:.4f} sib_peak={m['sibilant_peak']:.4f} "
              f"other_peak={m['other_peak']:.4f} rms={m['rms']:.4f} "
              f"ecapa_cos_mean={entry['ecapa_cosine_to_anchors_mean']:.4f}", flush=True)

    report["anchors_n"] = len(anchor_embs)
    report["ladder"] = ladder_report

    with open(CALIBRATION_JSON) as f:
        cal = json.load(f)["calibration"]["distributions"]["ecapa_cosine"]
    report["calibration_anchors"] = cal

    baseline = next(e for e in ladder_report if e["mag"] == 0.0)
    # pick the best magnitude: lowest ratio among those whose speaker cosine
    # stays above SPEAKER_COSINE_FLOOR and whose sibilant peak (not just the
    # whole-clip peak) actually decreased vs baseline.
    candidates = [e for e in ladder_report
                  if e["mag"] > 0.0
                  and e["ecapa_cosine_to_anchors_mean"] >= SPEAKER_COSINE_FLOOR
                  and e["sibilant_peak"] < baseline["sibilant_peak"]]
    best = min(candidates, key=lambda e: e["ratio"]) if candidates else None
    report["best_magnitude"] = best["mag"] if best else None
    report["numerator_or_denominator"] = dict(
        note="numerator = sibilant-frame peak, denominator = whole-clip peak",
        baseline_sibilant_peak=baseline["sibilant_peak"],
        baseline_whole_peak=baseline["whole_peak"],
        baseline_rms=baseline["rms"],
        per_magnitude=[dict(mag=e["mag"], sibilant_peak=e["sibilant_peak"],
                             whole_peak=e["whole_peak"], other_peak=e["other_peak"],
                             rms=e["rms"]) for e in ladder_report],
    )

    if best is None:
        report["verdict"] = ("A ridge direction exists and generalizes (R^2=%.4f), but no "
                              "step magnitude both lowered the sibilant peak and kept the "
                              "ECAPA cosine to F3 anchors >= %.2f. No usable de-essing "
                              "direction survives the speaker-identity control."
                              % (r2_heldout, SPEAKER_COSINE_FLOOR))
        print("\n" + report["verdict"])
        with open(REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        return

    print(f"\nBest magnitude: {best['mag']} "
          f"(ratio {baseline['ratio']:.4f} -> {best['ratio']:.4f}, "
          f"ecapa_cos_mean {best['ecapa_cosine_to_anchors_mean']:.4f})")

    # =================================================================== 5
    print("\n=== Step 5: generalization ===", flush=True)
    generalization = {}

    # F5 / seashells, direction applied UNCHANGED (same 24x256 delta vector)
    f5_style = load_voice_style([os.path.join(VOICE_STYLE_DIR, "F5.json")])
    f5_ttl = f5_style.ttl.astype(np.float32)
    f5_dp = f5_style.dp.copy()
    f5_P = f5_ttl[0, ACTIVE_ROWS, :].astype(np.float64)
    seashells_text = TEXTS[SEASHELLS_TEXT_IDX]
    gen_seed = 20270320
    for tag, mag in [("baseline", 0.0), ("corrected", best["mag"])]:
        rows = f5_P if mag == 0.0 else apply_direction(f5_P, direction_flat, mag)
        ttl = f5_ttl.copy() if mag == 0.0 else build_ttl(f5_ttl, rows)
        wav, sr = render(tts, seashells_text, ttl, f5_dp, gen_seed)
        stem = f"gen_{tag}_F5_seashells_mag{mag:.3f}_seed{gen_seed}"
        m = write_and_analyze(wmodel, wav, sr, stem)
        generalization[f"F5_seashells_{tag}"] = m
    print(f"  F5/seashells: ratio {generalization['F5_seashells_baseline']['ratio']:.4f} -> "
          f"{generalization['F5_seashells_corrected']['ratio']:.4f}")

    # F3 / a different sentence, same base point P
    alt_text = TEXTS[ALT_SENTENCE_TEXT_IDX]
    gen_seed2 = 20270321
    for tag, mag in [("baseline", 0.0), ("corrected", best["mag"])]:
        rows = P if mag == 0.0 else apply_direction(P, direction_flat, mag)
        ttl = base_ttl.copy() if mag == 0.0 else build_ttl(base_ttl, rows)
        wav, sr = render(tts, alt_text, ttl, dp_ref, gen_seed2)
        stem = f"gen_{tag}_F3_altsent_mag{mag:.3f}_seed{gen_seed2}"
        m = write_and_analyze(wmodel, wav, sr, stem)
        generalization[f"F3_altsentence_{tag}"] = m
    print(f"  F3/alt-sentence: ratio {generalization['F3_altsentence_baseline']['ratio']:.4f} -> "
          f"{generalization['F3_altsentence_corrected']['ratio']:.4f}")

    report["generalization"] = generalization
    report["direction_flat_npy"] = None  # saved separately below

    np.save(os.path.join(OUT_DIR, "direction_flat.npy"), direction_flat)
    np.save(os.path.join(OUT_DIR, "P_f3.npy"), P)

    report["verdict"] = (
        "Usable de-essing direction found: ridge held-out R^2=%.4f, best magnitude=%.2f, "
        "F3/library ratio %.4f -> %.4f with ECAPA cosine to F3 anchors %.4f (vs calibrated "
        "same-speaker floor ~%.2f, different-speaker ceiling ~%.2f)."
        % (r2_heldout, best["mag"], baseline["ratio"], best["ratio"],
           best["ecapa_cosine_to_anchors_mean"],
           cal["same_speaker_diff_seed (10 pairs)"]["min"],
           cal["different_speaker (45 preset pairs)"]["max"])
    )
    print("\n" + report["verdict"])

    report["step1_records"] = step1  # kept for the bench / re-analysis
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
