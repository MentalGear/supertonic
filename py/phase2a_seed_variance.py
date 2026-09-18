"""Phase 2a follow-up: is the bench-6 glitch a property of the style or of the
vocoder seed?

Bench 6 ("Phase 2a -- Capacity", see ../docs/LISTENING_BENCHES.md #6)
collected nine triples of (true style, probe prediction, control). Three
carried notes, and two of those three describe an audible hiccup in the
CONTROL clip -- which is M1 completely unperturbed, the shipped preset:

  K4_typical  (woodchuck, seed 20261069): "the 'chuck' after woodchuck sounds
    condensed/hiccuped in control"
  K16_worst   (seashells, seed 20261449): "control has the most prominent case
    where the final word 'morning' is pronounced too quickly so it sounds
    like a hiccup"
  K16_best    (woodchuck, seed 20261295): "control sounds normal and the
    best"

Since the CONTROL clip is the *same* unperturbed M1 style in every triple,
the only thing that varies between a glitchy control and a clean one is the
vocoder seed (`np.random.seed(...)` immediately before
`sample_noisy_latent`, see helper.py). This script renders the same
(text, style) pairs at 12 different seeds each -- including the four exact
bench-6 seeds -- to see whether the glitch tracks the seed, the style, or
both.

Styles rendered, per text (CLAUDE.md: style_dp is held fixed at M1's for
every style here, so durations are pinned and frames align across styles):

  woodchuck ("How much wood would a woodchuck chuck if it could chuck
  wood?"):
    control          -- M1 unperturbed
    true_K4_typical  -- true style of the triple that reported control glitch
    true_K16_best    -- true style of a same-text triple that reported clean

  seashells ("She sells seashells by the sea shore every summer morning."):
    control          -- M1 unperturbed
    true_K16_worst   -- true style of the triple that reported control glitch
    true_K64_typical -- true style of a same-text triple with no glitch note

Detection: log-mel spectrograms (16 kHz, matching phase2b_prosody_analysis's
NFFT/HOP), frame-to-frame L2 flux in log-mel space. A hiccup is a short,
sharp spectral break, which shows up as an isolated spike in this flux
signal. The flag threshold is the 99th percentile of the flux distribution
pooled across all 72 renders -- not hand-picked per clip.

Usage (from py/):
    python3 phase2a_seed_variance.py
"""

import datetime
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from phase2b_generate import (
    LANG,
    ONNX_DIR,
    SEED_BASE,
    SPEED,
    TEXTS,
    TOTAL_STEP,
    VOICE_STYLE_DIR,
    Style,
    load_text_to_speech,
    load_voice_style,
    timer,
)
from phase2b_prosody_analysis import HOP, NFFT, Canvas
from phase2b_prosody_mel import DIV, SEQ, cmap

SUBSPACE_DIR = "results/phase2b_subspace"
OUT_DIR = "results/phase2a"
FIG_DIR = os.path.join(OUT_DIR, "seed_variance_figs")
LISTEN_DIR = "results/listening_sets/phase2a_seed_variance"
REPORT_PATH = os.path.join(OUT_DIR, "seed_variance.json")

BASE_PRESET = "M1"
VOICE_STYLE_DIR_M1 = os.path.join(VOICE_STYLE_DIR, f"{BASE_PRESET}.json")

SR = 16000  # analysis sample rate, matches phase2b_prosody_analysis
N_MELS, FMIN_MEL, FMAX_MEL = 80, 60.0, 7800.0

WOODCHUCK_TEXT_IDX = 3
SEASHELLS_TEXT_IDX = 1
assert TEXTS[WOODCHUCK_TEXT_IDX].startswith("How much wood")
assert TEXTS[SEASHELLS_TEXT_IDX].startswith("She sells seashells")

# The four exact bench-6 seeds, plus 8 more spread across a disjoint range so
# there is no chance of colliding with a phase2b_generate_subspace corpus
# seed (SEED_BASE + idx_global, idx_global < 1280).
BENCH_SEEDS = [20261069, 20261295, 20261449, 20262075]
EXTRA_SEEDS = [SEED_BASE + 10000 + i for i in range(8)]
SEEDS = sorted(BENCH_SEEDS + EXTRA_SEEDS)

# What bench 6 (results/listening_sets/phase2a_capacity/manifest.json) said
# about the CONTROL clip in the triple that used each named seed.
BENCH_NOTE = {
    20261069: dict(triple="K4_typical", text="woodchuck", verdict="glitch",
                   quote="'chuck' after woodchuck sounds condensed/hiccuped in control"),
    20261295: dict(triple="K16_best", text="woodchuck", verdict="clean",
                   quote="control sounds normal and the best"),
    20261449: dict(triple="K16_worst", text="seashells", verdict="glitch",
                   quote="control has the most prominent case where the final word "
                         "'morning' is pronounced too quickly so it sounds like a hiccup"),
    20262075: dict(triple="K64_typical", text="seashells", verdict="clean",
                   quote="no note recorded (implicit: not flagged)"),
}

# (text_key, text_idx, style_key, K, sample_idx_in_ttl_K, description)
STYLE_SPECS = [
    dict(text_key="woodchuck", text_idx=WOODCHUCK_TEXT_IDX, style_key="control",
         K=None, sample_idx=None,
         desc="M1 unperturbed -- the shipped preset, identical to bench 6's CONTROL clip"),
    dict(text_key="woodchuck", text_idx=WOODCHUCK_TEXT_IDX, style_key="true_K4_typical",
         K=4, sample_idx=161,
         desc="true style of K4_typical -- bench 6 listener flagged this triple's control as glitched"),
    dict(text_key="woodchuck", text_idx=WOODCHUCK_TEXT_IDX, style_key="true_K16_best",
         K=16, sample_idx=67,
         desc="true style of K16_best -- bench 6 listener called this triple's control clean"),
    dict(text_key="seashells", text_idx=SEASHELLS_TEXT_IDX, style_key="control",
         K=None, sample_idx=None,
         desc="M1 unperturbed -- the shipped preset, identical to bench 6's CONTROL clip"),
    dict(text_key="seashells", text_idx=SEASHELLS_TEXT_IDX, style_key="true_K16_worst",
         K=16, sample_idx=221,
         desc="true style of K16_worst -- bench 6 listener flagged this triple's control as glitched"),
    dict(text_key="seashells", text_idx=SEASHELLS_TEXT_IDX, style_key="true_K64_typical",
         K=64, sample_idx=527,
         desc="true style of K64_typical -- bench 6 recorded no glitch note for this triple"),
]


def logmel(w, sr=SR):
    import librosa
    m = librosa.feature.melspectrogram(
        y=w, sr=sr, n_fft=NFFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN_MEL, fmax=FMAX_MEL, center=True, power=2.0,
    )
    return 10.0 * np.log10(m + 1e-10)


def spectral_flux(M):
    """Frame-to-frame L2 norm of the log-mel delta. M: (n_mels, T) -> (T-1,)."""
    return np.linalg.norm(np.diff(M, axis=1), axis=0)


def diff_figure(path, title, M_a, M_b, label_a, label_b, caption_lines):
    """Two log-mel panels plus their signed diff, hand-rolled PNG (no matplotlib)."""
    n = min(M_a.shape[1], M_b.shape[1])
    M_a, M_b = M_a[:, :n], M_b[:, :n]
    D = M_a - M_b
    sym = float(np.percentile(np.abs(D), 99))
    PH, GAP, T, L, R, B = 160, 40, 90, 100, 24, 70
    W = 1180
    H = T + 3 * (PH + GAP) + B
    cv = Canvas(W, H)
    x0, x1 = L, W - R
    panels = [(f"{label_a}", M_a, False), (f"{label_b}", M_b, False),
              (f"diff ({label_a} - {label_b})", D, True)]
    vmin, vmax = float(np.percentile(np.concatenate([M_a, M_b]), 2)), float(max(M_a.max(), M_b.max()))
    for i, (lab, arr, is_diff) in enumerate(panels):
        y0 = T + i * (PH + GAP)
        h, w = PH, x1 - x0
        a = np.asarray(arr, float)
        if is_diff:
            z = 0.5 + 0.5 * np.clip(a / max(sym, 1e-9), -1, 1)
            anchors = DIV
        else:
            z = (a - vmin) / max(vmax - vmin, 1e-9)
            anchors = SEQ
        ri = np.clip((np.arange(h) / h * a.shape[0]).astype(int), 0, a.shape[0] - 1)[::-1]
        ci = np.clip((np.arange(w) / w * a.shape[1]).astype(int), 0, a.shape[1] - 1)
        cv.px[int(y0):int(y0) + h, int(x0):int(x0) + w] = cmap(z[np.ix_(ri, ci)], anchors)
        cv.text(L, y0 - 16, lab[:90], (30, 30, 34), 2)
    cv.text(L, 16, title[:100], (20, 20, 24), 2)
    yb = T + 3 * (PH + GAP) - GAP + 10
    for j, ln in enumerate(caption_lines):
        cv.text(L, yb + j * 16, ln[:130], (70, 70, 76), 2)
    cv.save(path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)

    subspace = np.load(os.path.join(SUBSPACE_DIR, "subspace.npz"))
    base_ttl_full = subspace["base_ttl"].astype(np.float32)  # (50, 256), M1

    dp_ref = load_voice_style([VOICE_STYLE_DIR_M1]).dp  # held fixed for every style below

    print("Loading TTS engine ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    native_sr = tts.sample_rate

    def style_ttl_for(spec):
        if spec["K"] is None:
            return base_ttl_full.copy()
        return subspace[f"ttl_K{spec['K']}"][spec["sample_idx"]].astype(np.float32)

    def render(text, ttl_50_256, seed):
        np.random.seed(seed)  # vocoder latent RNG, see helper.py sample_noisy_latent
        ttl = ttl_50_256[None].astype(np.float32)
        wav, dur = tts(text, LANG, Style(ttl, dp_ref.copy()), TOTAL_STEP, SPEED)
        trimmed = wav[0, : int(native_sr * dur[0].item())].astype(np.float32)
        return trimmed, float(dur[0].item())

    all_records = []
    flux_by_key = {}      # (text_key, style_key, seed) -> flux array
    wav16k_by_key = {}    # kept only for the 4 named-seed control renders (figures)
    n_frames_pooled = []

    with timer(f"rendering {len(STYLE_SPECS)} styles x {len(SEEDS)} seeds = "
               f"{len(STYLE_SPECS) * len(SEEDS)} renders"):
        for spec in STYLE_SPECS:
            text = TEXTS[spec["text_idx"]]
            ttl = style_ttl_for(spec)
            for seed in SEEDS:
                wav, dur = render(text, ttl, seed)
                w16 = resample_poly(wav, SR, native_sr).astype(np.float32)
                M = logmel(w16)
                flux = spectral_flux(M)

                stem = f"{spec['text_key']}_{spec['style_key']}_seed{seed}"
                sf.write(os.path.join(LISTEN_DIR, f"{stem}.wav"), wav, native_sr, subtype="PCM_16")

                key = (spec["text_key"], spec["style_key"], seed)
                flux_by_key[key] = flux
                n_frames_pooled.append(flux.shape[0])
                if spec["style_key"] == "control" and seed in BENCH_SEEDS:
                    wav16k_by_key[key] = (w16, M)

                all_records.append({
                    "text_key": spec["text_key"], "style_key": spec["style_key"],
                    "K": spec["K"], "sample_idx": spec["sample_idx"], "seed": seed,
                    "is_bench_seed": seed in BENCH_SEEDS,
                    "duration_sec": dur, "n_frames": int(flux.shape[0]),
                    "file": f"{stem}.wav",
                })

    # --- calibrate the flag threshold on the pooled distribution across all 72 renders ---
    pooled = np.concatenate(list(flux_by_key.values()))
    thresh = float(np.percentile(pooled, 99))
    med = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - med)))
    print(f"pooled flux: n={pooled.size} median={med:.3f} MAD={mad:.3f} "
          f"p99={thresh:.3f} (threshold) p99.5={np.percentile(pooled, 99.5):.3f} "
          f"p95={np.percentile(pooled, 95):.3f}")

    def flag_frames(flux):
        idx = np.flatnonzero(flux > thresh)
        return idx

    # --- per-style, per-seed flag counts, and fraction-of-seeds-flagged summary ---
    style_summary = {}
    for spec in STYLE_SPECS:
        rows = []
        n_seed_flagged = 0
        for seed in SEEDS:
            key = (spec["text_key"], spec["style_key"], seed)
            flux = flux_by_key[key]
            idx = flag_frames(flux)
            times_ms = (idx * HOP / SR * 1000.0).round(1).tolist()
            flagged = len(idx) > 0
            n_seed_flagged += int(flagged)
            rows.append({
                "seed": seed, "is_bench_seed": seed in BENCH_SEEDS,
                "n_flagged_frames": int(len(idx)),
                "flagged_times_ms": times_ms,
                "max_flux": float(flux.max()),
            })
        style_summary[f"{spec['text_key']}::{spec['style_key']}"] = {
            "text_key": spec["text_key"], "style_key": spec["style_key"],
            "K": spec["K"], "sample_idx": spec["sample_idx"], "desc": spec["desc"],
            "n_seeds": len(SEEDS),
            "n_seeds_flagged": n_seed_flagged,
            "fraction_seeds_flagged": n_seed_flagged / len(SEEDS),
            "per_seed": rows,
        }

    # --- detector validation against the listener's 3 (4) judgments ---
    # Bench-6 CONTROL == M1 unperturbed at the triple's own seed.
    validation = []
    for seed, note in BENCH_NOTE.items():
        text_key = note["text"]
        key = (text_key, "control", seed)
        flux = flux_by_key[key]
        idx = flag_frames(flux)
        predicted = "glitch" if len(idx) > 0 else "clean"
        validation.append({
            "seed": seed, "triple": note["triple"], "text_key": text_key,
            "listener_verdict": note["verdict"], "listener_quote": note["quote"],
            "detector_verdict": predicted, "match": predicted == note["verdict"],
            "n_flagged_frames": int(len(idx)),
            "flagged_times_ms": (idx * HOP / SR * 1000.0).round(1).tolist(),
        })
        print(f"validate seed={seed} triple={note['triple']:12s} listener={note['verdict']:6s} "
              f"detector={predicted:6s} match={predicted == note['verdict']} "
              f"n_flagged={len(idx)}")

    n_match = sum(v["match"] for v in validation)
    detector_reliable = n_match == len(validation)

    # --- figures (woodchuck text only, per instructions) ---
    # (a) unperturbed-at-glitchy-seed vs unperturbed-at-clean-seed: isolates the
    #     seed's effect on the SAME style/text -- durations are pinned (style_dp
    #     fixed at M1's throughout), so frames line up 1:1.
    w_glitchy, M_glitchy = wav16k_by_key[("woodchuck", "control", 20261069)]
    w_clean, M_clean = wav16k_by_key[("woodchuck", "control", 20261295)]
    diff_figure(
        os.path.join(FIG_DIR, "woodchuck_control_seed20261069_vs_seed20261295.png"),
        "Woodchuck, M1 control: seed 20261069 (bench-6 glitch) vs seed 20261295 (bench-6 clean)",
        M_glitchy, M_clean,
        "seed 20261069 (glitchy)", "seed 20261295 (clean)",
        ["Same text, same style (M1 unperturbed), same style_dp -> durations pinned, frames aligned.",
         "Blue/red in the diff panel = seed-only spectral difference; a sharp isolated column is the seed's hiccup.",
         f"Detector threshold (p99 of pooled flux, all 72 renders) = {thresh:.2f} dB/frame (log-mel L2 flux)."],
    )

    # (b) true-vs-control at a fixed seed: isolates the style's effect, holding
    #     the seed (and therefore the vocoder noise draw) fixed.
    fixed_seed = 20261069
    ttl_true = subspace["ttl_K4"][161].astype(np.float32)
    wav_true, _ = render(TEXTS[WOODCHUCK_TEXT_IDX], ttl_true, fixed_seed)
    w16_true = resample_poly(wav_true, SR, native_sr).astype(np.float32)
    M_true = logmel(w16_true)
    diff_figure(
        os.path.join(FIG_DIR, "woodchuck_true_K4_typical_vs_control_seed20261069.png"),
        f"Woodchuck, fixed seed {fixed_seed}: true style (K4_typical) vs M1 control",
        M_true, M_glitchy,
        "true style (K4_typical, idx161)", "control (M1)",
        ["Same text, same seed, same style_dp -> durations pinned, frames aligned.",
         "Isolates what the STYLE changes, independent of the seed-driven glitch shown in the other figure.",
         f"Detector threshold = {thresh:.2f} dB/frame."],
    )

    print(f"Wrote figures -> {FIG_DIR}")

    report = {
        "experiment": "phase2a_seed_variance",
        "date": datetime.date.today().isoformat(),
        "question": "Is the bench-6 control-clip glitch explained by the vocoder seed, the style, or both?",
        "base_preset": BASE_PRESET,
        "texts": {"woodchuck": TEXTS[WOODCHUCK_TEXT_IDX], "seashells": TEXTS[SEASHELLS_TEXT_IDX]},
        "seeds": SEEDS,
        "bench_seeds": BENCH_SEEDS,
        "extra_seeds": EXTRA_SEEDS,
        "lang": LANG, "total_step": TOTAL_STEP, "speed": SPEED,
        "style_dp": "held fixed at M1's for every style/seed -- durations pinned across the whole run",
        "analysis_sample_rate": SR,
        "mel": {"n_fft": NFFT, "hop": HOP, "n_mels": N_MELS, "fmin": FMIN_MEL, "fmax": FMAX_MEL},
        "detector": "frame-to-frame L2 norm of the log-mel delta (spectral flux in log-mel space)",
        "threshold_calibration": {
            "method": "99th percentile of per-frame flux pooled across all 72 renders",
            "pooled_n_frames": int(pooled.size),
            "pooled_median": med, "pooled_mad": mad,
            "threshold": thresh,
            "p95": float(np.percentile(pooled, 95)),
            "p99_5": float(np.percentile(pooled, 99.5)),
        },
        "style_summary": style_summary,
        "detector_validation_against_listener": validation,
        "detector_validation_n_match": n_match,
        "detector_validation_n_total": len(validation),
        "detector_reliable": detector_reliable,
        "figures": [
            "results/phase2a/seed_variance_figs/woodchuck_control_seed20261069_vs_seed20261295.png",
            "results/phase2a/seed_variance_figs/woodchuck_true_K4_typical_vs_control_seed20261069.png",
        ],
        "records": all_records,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report -> {REPORT_PATH}")

    print("\n=== fraction of seeds flagged, per (text, style) ===")
    for k, v in style_summary.items():
        print(f"  {k:35s} {v['n_seeds_flagged']:2d}/{v['n_seeds']}  ({v['fraction_seeds_flagged']:.2f})  "
              f"K={v['K']}")


if __name__ == "__main__":
    main()
