"""Phase 2a follow-up, stage 0: engine baseline artifact rate.

Bench 6's CONTROL clips are stock Supertonic -- the shipped preset,
unmodified `style_ttl`/`style_dp`, the ordinary pipeline, nothing of this
fork's perturbations in them. Two of those control clips were reported as
glitchy (a "hiccup") and one as clean, all three being the SAME unperturbed
style rendered at different vocoder seeds (see phase2a_seed_variance.py's
docstring for the exact bench-6 quotes). That means whatever produces the
glitch is a property of the stock engine's own rendering, not of anything
this project has added -- and nobody has ever measured how often stock
Supertonic does this on its own. Without that number there is no floor to
compare any perturbed style against.

This script renders all 10 shipped presets (M1-M5, F1-F5), completely
unperturbed (each preset's own style_ttl AND style_dp -- the literal
out-of-the-box pipeline, not the style_dp-pinned setup phase2a_seed_variance
uses for its style comparison), on all 8 corpus TEXTS from
phase2b_generate.py, at 6 vocoder seeds each: 10 x 8 x 6 = 480 renders.

It computes the same frame-to-frame log-mel L2 flux detector used in
phase2a_seed_variance.py over every render and reports:
  - the overall baseline flag rate (fraction of stock renders with >=1
    flagged frame, at a threshold calibrated on this 480-render pool);
  - that rate broken down per preset and per text;
  - the seed-to-seed variability of peak flux for each fixed (preset, text)
    cell -- the quantity that says how many seeds a listening bench needs
    per condition to be trustworthy.

Audio is not retained -- this is a calibration/characterization run, not a
listening set; only the report JSON is kept.

Usage (from py/):
    python3 phase2a_baseline_artifact_rate.py
"""

import datetime
import json
import os

import numpy as np

from phase2b_generate import (
    LANG,
    ONNX_DIR,
    PRESETS,
    SEED_BASE,
    SPEED,
    TEXTS,
    TOTAL_STEP,
    VOICE_STYLE_DIR,
    load_text_to_speech,
    load_voice_style,
    timer,
)
from phase2a_seed_variance import HOP, NFFT, N_MELS, FMIN_MEL, FMAX_MEL, SR, logmel, spectral_flux  # noqa: E402

OUT_DIR = "results/phase2a"
REPORT_PATH = os.path.join(OUT_DIR, "baseline_artifact_rate.json")

N_SEEDS = 6
BASELINE_SEED_BASE = SEED_BASE + 100000  # disjoint from every other seed pool this project uses


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.signal import resample_poly

    print("Loading TTS engine ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    native_sr = tts.sample_rate

    styles = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]) for p in PRESETS}

    records = []
    flux_by_key = {}  # (preset, text_idx, seed) -> flux array
    counter = 0

    with timer(f"rendering {len(PRESETS)} presets x {len(TEXTS)} texts x {N_SEEDS} seeds = "
               f"{len(PRESETS) * len(TEXTS) * N_SEEDS} renders"):
        for p_i, preset in enumerate(PRESETS):
            style = styles[preset]
            for t_i, text in enumerate(TEXTS):
                for s_i in range(N_SEEDS):
                    seed = BASELINE_SEED_BASE + counter
                    counter += 1
                    np.random.seed(seed)
                    wav, dur = tts(text, LANG, style, TOTAL_STEP, SPEED)
                    trimmed = wav[0, : int(native_sr * dur[0].item())].astype(np.float32)
                    w16 = resample_poly(trimmed, SR, native_sr).astype(np.float32)
                    M = logmel(w16)
                    flux = spectral_flux(M)

                    key = (preset, t_i, seed)
                    flux_by_key[key] = flux
                    records.append({
                        "preset": preset, "text_idx": t_i, "seed": seed,
                        "duration_sec": float(dur[0].item()),
                        "max_flux": float(flux.max()), "mean_flux": float(flux.mean()),
                        "median_flux": float(np.median(flux)), "n_frames": int(flux.shape[0]),
                    })
                    if counter % 60 == 0:
                        print(f"  {counter} rendered", flush=True)

    pooled = np.concatenate(list(flux_by_key.values()))
    thresholds = {f"p{q}": float(np.percentile(pooled, q)) for q in (90, 95, 99, 99.5, 99.9)}
    print(f"pooled baseline flux: n={pooled.size} " +
          " ".join(f"{k}={v:.2f}" for k, v in thresholds.items()))

    def flag_rate_at(thresh):
        n_flag = sum(1 for flux in flux_by_key.values() if flux.max() > thresh)
        return n_flag / len(flux_by_key)

    overall_rate = {k: flag_rate_at(v) for k, v in thresholds.items()}

    # per-preset, per-text breakdown at the p99 threshold (reported as the
    # primary threshold; other percentiles included in `thresholds` for
    # sensitivity, and picked apart in phase2a_seed_variance's validation)
    primary = thresholds["p99"]

    def flagged(key):
        return flux_by_key[key].max() > primary

    per_preset = {}
    for preset in PRESETS:
        keys = [k for k in flux_by_key if k[0] == preset]
        n_flag = sum(flagged(k) for k in keys)
        per_preset[preset] = {"n": len(keys), "n_flagged": n_flag, "rate": n_flag / len(keys)}

    per_text = {}
    for t_i, text in enumerate(TEXTS):
        keys = [k for k in flux_by_key if k[1] == t_i]
        n_flag = sum(flagged(k) for k in keys)
        per_text[str(t_i)] = {"text": text, "n": len(keys), "n_flagged": n_flag, "rate": n_flag / len(keys)}

    # seed-to-seed variability of peak flux, per (preset, text) cell -- the
    # quantity that tells us how many seeds a bench needs per condition
    cell_variance = []
    for preset in PRESETS:
        for t_i, text in enumerate(TEXTS):
            keys = [k for k in flux_by_key if k[0] == preset and k[1] == t_i]
            maxes = np.array([flux_by_key[k].max() for k in keys])
            cv = float(maxes.std() / maxes.mean()) if maxes.mean() > 0 else float("nan")
            cell_variance.append({
                "preset": preset, "text_idx": t_i,
                "max_flux_mean": float(maxes.mean()), "max_flux_std": float(maxes.std()),
                "coefficient_of_variation": cv,
                "n_flagged_of_n_seeds": int(sum(m > primary for m in maxes)),
                "n_seeds": len(maxes),
            })
    cv_values = [c["coefficient_of_variation"] for c in cell_variance if np.isfinite(c["coefficient_of_variation"])]

    report = {
        "experiment": "phase2a_baseline_artifact_rate",
        "date": datetime.date.today().isoformat(),
        "purpose": "stock-engine artifact-rate floor, unperturbed presets, for comparison against "
                   "phase2a_seed_variance's perturbed-style flag rates",
        "presets": PRESETS,
        "texts": TEXTS,
        "n_seeds_per_cell": N_SEEDS,
        "n_total_renders": len(records),
        "analysis_sample_rate": SR,
        "mel": {"n_fft": NFFT, "hop": HOP, "n_mels": N_MELS, "fmin": FMIN_MEL, "fmax": FMAX_MEL},
        "detector": "frame-to-frame L2 norm of the log-mel delta (spectral flux in log-mel space); "
                    "flag = render's max flux exceeds threshold",
        "threshold_candidates": thresholds,
        "primary_threshold": primary,
        "overall_baseline_flag_rate_by_threshold": overall_rate,
        "per_preset_flag_rate_at_p99": per_preset,
        "per_text_flag_rate_at_p99": per_text,
        "seed_variance_per_cell": cell_variance,
        "seed_variance_cv_summary": {
            "mean": float(np.mean(cv_values)) if cv_values else None,
            "median": float(np.median(cv_values)) if cv_values else None,
            "min": float(np.min(cv_values)) if cv_values else None,
            "max": float(np.max(cv_values)) if cv_values else None,
        },
        "records": records,
        "caveat": "This detector has NOT been shown to track the perceptual glitch bench 6 reported -- "
                  "see phase2a_seed_variance.py's validation section. These rates characterize the "
                  "flux-max distribution of stock renders, not a confirmed audible-artifact rate, "
                  "until a validated detector exists.",
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report -> {REPORT_PATH}")
    print(f"overall baseline flag rate @ p99: {overall_rate['p99']:.3f}")
    print("per-preset @ p99:")
    for p, v in per_preset.items():
        print(f"  {p}: {v['n_flagged']}/{v['n']} = {v['rate']:.2f}")
    print("per-text @ p99:")
    for t, v in per_text.items():
        print(f"  [{t}] {v['text'][:40]:40s} {v['n_flagged']}/{v['n']} = {v['rate']:.2f}")
    print(f"seed-variance CV (max-flux, per preset x text cell): "
          f"mean={report['seed_variance_cv_summary']['mean']:.3f} "
          f"median={report['seed_variance_cv_summary']['median']:.3f}")


if __name__ == "__main__":
    main()
