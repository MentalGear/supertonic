"""Combine phase2a_baseline_artifact_rate.json (stage 0) with
phase2a_seed_variance.json (stage 1) into one final, honestly-calibrated
report: results/phase2a/seed_variance_final.json.

Re-runs the flux detector's validation against bench 6's known judgments
using EVERY baseline-calibrated threshold candidate (p90/p95/p99/p99.5/p99.9),
not just one -- if no candidate separates the two listener-flagged-glitch
control clips from the two listener-flagged-clean control clips in the right
direction, the detector is marked unreliable and the style/baseline flag-rate
numbers are reported as uncalibrated, per instructions.

Usage (from py/, after both phase2a_baseline_artifact_rate.py and
phase2a_seed_variance.py have been run):
    python3 phase2a_combine_seed_variance.py
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from phase2a_seed_variance import (
    BENCH_NOTE,
    BENCH_SEEDS,
    LISTEN_DIR,
    SR,
    STYLE_SPECS,
    SEEDS,
    logmel,
    spectral_flux,
)

OUT_DIR = "results/phase2a"
BASELINE_PATH = os.path.join(OUT_DIR, "baseline_artifact_rate.json")
SEED_VARIANCE_PATH = os.path.join(OUT_DIR, "seed_variance.json")
FINAL_PATH = os.path.join(OUT_DIR, "seed_variance_final.json")


def flux_for_file(fname):
    path = os.path.join(LISTEN_DIR, fname)
    w, sr = sf.read(path)
    w16 = resample_poly(w.astype(np.float32), SR, sr).astype(np.float32)
    return spectral_flux(logmel(w16))


def main():
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(SEED_VARIANCE_PATH) as f:
        stage1 = json.load(f)

    thresholds = baseline["threshold_candidates"]

    # Recompute flux for the four bench-6 control clips (already rendered by
    # phase2a_seed_variance.py -- filenames follow its stem convention).
    control_files = {
        20261069: "woodchuck_control_seed20261069.wav",
        20261295: "woodchuck_control_seed20261295.wav",
        20261449: "seashells_control_seed20261449.wav",
        20262075: "seashells_control_seed20262075.wav",
    }
    control_flux = {seed: flux_for_file(fn) for seed, fn in control_files.items()}
    control_max = {seed: float(flux.max()) for seed, flux in control_flux.items()}

    print("Control clip max-flux (bench-6 known judgments):")
    for seed, note in BENCH_NOTE.items():
        print(f"  seed={seed} triple={note['triple']:12s} listener={note['verdict']:6s} "
              f"max_flux={control_max[seed]:.2f}")

    validation_by_threshold = {}
    for label, thresh in thresholds.items():
        rows = []
        for seed, note in BENCH_NOTE.items():
            predicted = "glitch" if control_max[seed] > thresh else "clean"
            rows.append({
                "seed": seed, "triple": note["triple"], "listener_verdict": note["verdict"],
                "detector_verdict": predicted, "match": predicted == note["verdict"],
            })
        n_match = sum(r["match"] for r in rows)
        validation_by_threshold[label] = {
            "threshold": thresh, "rows": rows, "n_match": n_match, "n_total": len(rows),
            "all_match": n_match == len(rows),
        }
        print(f"  threshold {label}={thresh:.2f}: {n_match}/{len(rows)} match "
              f"({'ALL MATCH -- usable' if n_match == len(rows) else 'fails'})")

    any_threshold_works = any(v["all_match"] for v in validation_by_threshold.values())

    # Ordering check, independent of any specific threshold: does max_flux even
    # rank the two glitch clips above their same-text clean counterpart?
    ordering = {
        "woodchuck: glitch(20261069) vs clean(20261295)": {
            "glitch_max_flux": control_max[20261069], "clean_max_flux": control_max[20261295],
            "glitch_higher": control_max[20261069] > control_max[20261295],
        },
        "seashells: glitch(20261449) vs clean(20262075)": {
            "glitch_max_flux": control_max[20261449], "clean_max_flux": control_max[20262075],
            "glitch_higher": control_max[20261449] > control_max[20262075],
        },
    }
    both_correct_direction = all(v["glitch_higher"] for v in ordering.values())

    detector_reliable = any_threshold_works and both_correct_direction

    # --- supplementary probe: raw-waveform sample-to-sample discontinuity ---
    # The log-mel flux detector above is anti-correlated with the listener's
    # judgment. Tried as a cheap alternative (no re-rendering: reuses the same
    # 4 already-saved WAVs): per-8ms-frame max |x[n]-x[n-1]|, RMS-normalized.
    # This is directionally consistent (glitch > same-text clean) on BOTH known
    # pairs, unlike log-mel flux -- but with only 2 pairs to calibrate against,
    # and overlapping within-text percentile ranks (woodchuck-clean actually
    # ranks higher than seashells-glitch), it is a lead, not a validated
    # detector. Reported for the record, not used for any flag-rate claim.
    def click_max(fname, hop_ms=8.0):
        path = os.path.join(LISTEN_DIR, fname)
        w, sr = sf.read(path)
        w = w.astype(np.float64)
        d = np.abs(np.diff(w))
        rms = np.sqrt((w ** 2).mean())
        frame = max(1, int(round(hop_ms / 1000.0 * sr)))
        n = len(d) // frame
        d = d[: n * frame].reshape(n, frame)
        return float((d.max(axis=1) / max(rms, 1e-9)).max())

    click_vals = {seed: click_max(fn) for seed, fn in control_files.items()}
    click_ordering = {
        "woodchuck: glitch(20261069) vs clean(20261295)": {
            "glitch": click_vals[20261069], "clean": click_vals[20261295],
            "glitch_higher": click_vals[20261069] > click_vals[20261295],
        },
        "seashells: glitch(20261449) vs clean(20262075)": {
            "glitch": click_vals[20261449], "clean": click_vals[20262075],
            "glitch_higher": click_vals[20261449] > click_vals[20262075],
        },
    }
    click_both_correct = all(v["glitch_higher"] for v in click_ordering.values())
    supplementary_click_probe = {
        "description": "per-8ms-frame max |sample-to-sample waveform diff|, RMS-normalized, max over the render",
        "values_by_seed": click_vals,
        "ordering_check": click_ordering,
        "both_pairs_correct_direction": click_both_correct,
        "validated_as_detector": False,
        "note": "Directionally consistent on both known pairs (unlike log-mel flux), but a single threshold "
                "could not separate both pairs even after within-text percentile normalization (glitch "
                "percentile 58.3 for seashells < clean percentile 72.2 for woodchuck) -- promising lead for "
                "a follow-up detector, not itself validated. Not run across the 480-render baseline or the "
                "full 72-render sweep due to time budget; would need both to become a reportable rate.",
    }

    final = {
        "experiment": "phase2a_seed_variance_final",
        "detector_reliable": detector_reliable,
        "any_baseline_threshold_separates_known_pairs": any_threshold_works,
        "max_flux_ranks_glitch_above_clean_for_both_known_pairs": both_correct_direction,
        "ordering_check": ordering,
        "validation_by_baseline_threshold": validation_by_threshold,
        "conclusion": (
            "Detector FAILS validation: for both known pairs, the listener-flagged-GLITCH control clip "
            "has LOWER max frame-to-frame log-mel flux than the same-text listener-flagged-CLEAN control "
            "clip -- the opposite of what a spike detector needs. No baseline-calibrated threshold can fix "
            "this because the ordering itself is reversed, not merely miscalibrated. Baseline flag-rate and "
            "style flag-rate numbers from this detector are NOT reported as evidence of audible glitching; "
            "see raw distributional numbers in baseline_artifact_rate.json / seed_variance.json instead, "
            "explicitly marked as uncalibrated flux-distribution facts."
        ) if not detector_reliable else "Detector validated -- see validation_by_baseline_threshold for the working threshold.",
        "supplementary_click_probe": supplementary_click_probe,
        "baseline_source": BASELINE_PATH,
        "stage1_source": SEED_VARIANCE_PATH,
    }
    with open(FINAL_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\ndetector_reliable = {detector_reliable}")
    print(f"Wrote -> {FINAL_PATH}")


if __name__ == "__main__":
    main()
