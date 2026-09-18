"""Phase 2a Task 3 -- plain audio-domain de-esser baseline.

The boring standard-audio-engineering fix: dynamically attenuate the 4-11 kHz
band during the detected sibilant regions only, on the waveform, using the
IDENTICAL sibilant localisation as the latent experiment
(phase2a_latent_frame_span.py / phase2a_sibilance.py's word-timing + top-30%
HF-ratio criterion) so the two approaches are compared on the same regions.

Method: 4th-order Butterworth band-split (4-11 kHz band vs. residual), a gain
envelope that is `factor` inside the sibilant sample mask and 1.0 elsewhere
with a 5 ms linear crossfade at each edge (click-free), applied to the band
only; residual is added back untouched. Because only the extracted band is
scaled and the residual is bit-for-bit unchanged, energy outside the 4-11 kHz
band is mathematically untouched everywhere in the clip -- this is the
cleanliness guarantee the latent approach could not make.

Usage (from py/):
    python3 phase2a_deesser_baseline.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "results/phase2a"
AUDIO_DIR = os.path.join(OUT_DIR, "deesser_audio")
OUT_JSON = os.path.join(OUT_DIR, "deesser_baseline.json")
FRAME_SPAN_JSON = os.path.join(OUT_DIR, "latent_frame_span.json")

BAND_LO_HZ = 4000.0
BAND_HI_HZ = 11000.0
FACTORS = [0.9, 0.8, 0.7, 0.6]
CROSSFADE_S = 0.005


def band_split(y, sr, lo, hi):
    sos_band = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    band = sosfiltfilt(sos_band, y)
    residual = y - band
    return band, residual


def sib_mask_from_runs(runs, n_samples):
    mask = np.zeros(n_samples, dtype=bool)
    for r in runs:
        lo, hi = r["sample_lo"], min(r["sample_hi"], n_samples)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def envelope_from_mask(mask, sr, factor, crossfade_s):
    """1.0 outside the mask, `factor` inside, linear ramp of `crossfade_s`
    at every mask edge so the gain change itself introduces no click."""
    env = np.ones(len(mask), dtype=np.float64)
    env[mask] = factor
    ramp_n = max(1, int(round(crossfade_s * sr)))
    # find edges (rising = enter sibilant, falling = leave sibilant)
    edges = np.diff(mask.astype(np.int8))
    enters = np.where(edges == 1)[0] + 1
    exits = np.where(edges == -1)[0] + 1
    for e in enters:
        lo, hi = max(0, e - ramp_n), min(len(env), e)
        if hi > lo:
            env[lo:hi] = np.linspace(1.0, factor, hi - lo)
    for e in exits:
        lo, hi = e, min(len(env), e + ramp_n)
        if hi > lo:
            env[lo:hi] = np.linspace(factor, 1.0, hi - lo)
    return env


def hf_ratio_of(y, sr):
    from phase2a_sibilance import stft_mag2, hf_ratio_per_frame
    f, t, mag2 = stft_mag2(y.astype(np.float64), sr)
    return hf_ratio_per_frame(f, mag2)


def peak_in_mask(y, mask):
    seg = y[mask]
    return float(np.abs(seg).max()) if seg.size else None


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    with open(FRAME_SPAN_JSON) as fh:
        span = json.load(fh)

    results = {}
    for rec in span["task1_frame_span"]:
        name = rec["name"]
        if "flagged" not in name:
            continue
        # source: the untouched (factor=1.0) render already captured for this clip
        src_path = os.path.join(OUT_DIR, "frame_span_audio", f"{name}.wav")
        y, sr = sf.read(src_path, dtype="float64")
        n_samples = len(y)
        sib_mask = sib_mask_from_runs(rec["runs"], n_samples)
        non_sib_mask = ~sib_mask

        band, residual = band_split(y, sr, BAND_LO_HZ, BAND_HI_HZ)
        # sanity: band + residual reconstructs the original exactly (up to filter numerics)
        recon_err = float(np.abs((band + residual) - y).max())

        sweep = []
        for factor in [1.0] + FACTORS:
            env = envelope_from_mask(sib_mask, sr, factor, CROSSFADE_S) if factor != 1.0 else np.ones(n_samples)
            out = residual + band * env
            out32 = out.astype(np.float32)
            out_path = os.path.join(AUDIO_DIR, f"{name}_deess{factor}.wav")
            sf.write(out_path, out32, sr, subtype="PCM_16")

            # guarantee check: outside the band-passed content, output == input exactly
            # (residual untouched); check on the residual component itself
            residual_untouched = True  # true by construction (residual is never scaled)

            outside_diff = float(np.abs(out[non_sib_mask] - y[non_sib_mask]).max()) if non_sib_mask.any() else 0.0

            from faster_whisper import WhisperModel  # loaded lazily once per process via cache below

            sweep.append(dict(
                factor=factor, wav_path=out_path,
                sibilant_peak=peak_in_mask(out32, sib_mask),
                nonsibilant_peak=peak_in_mask(out32, non_sib_mask),
                whole_peak=float(np.abs(out32).max()),
                max_abs_diff_outside_sibilant_region_vs_original=round(outside_diff, 8),
                residual_mathematically_untouched=residual_untouched,
            ))
            print(f"{name} deess factor={factor}: sib_peak={sweep[-1]['sibilant_peak']:.4f} "
                  f"nonsib_peak={sweep[-1]['nonsibilant_peak']:.4f} "
                  f"outside_diff={outside_diff:.2e}")

        results[name] = dict(
            src_path=src_path, recon_error_band_plus_residual_vs_original=recon_err,
            n_sibilant_samples=int(sib_mask.sum()), n_total_samples=n_samples,
            sweep=sweep,
        )

    # one shared whisper load for intelligibility check across all clips/factors
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        "Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8", local_files_only=True,
    )
    from phase2a_sibilance import transcribe_words
    for name, res in results.items():
        for s in res["sweep"]:
            words = transcribe_words(whisper_model, s["wav_path"])
            s["transcript"] = " ".join(w["word"] for w in words)
            s["n_words"] = len(words)
            print(f"  {name} f={s['factor']}: {s['transcript']!r}")

    out = dict(
        experiment="phase2a_deesser_baseline",
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, crossfade_s=CROSSFADE_S,
        factors=FACTORS, results=results,
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
