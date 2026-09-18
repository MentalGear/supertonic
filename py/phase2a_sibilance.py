"""Phase 2a -- sibilant over-drive: is it clipping, aliasing, or under-resolved
denoising?

Bench 7 (`phase2a_baseline_artifact_rate.py` / `benches/phase2a_baseline_bench.py`,
see `docs/LISTENING_BENCHES.md` #7) drew 16 unperturbed stock clips blind, one
per (preset, text, seed) cell, plus 4 hidden repeats of bench-6's M1 control
clips. Every clip that was flagged for an audible artifact was a female
preset (F1, F3 "yes clearly", F5 twice), 0 of 8 male clips flagged (Fisher
one-sided p=0.038), and the clear-flag quote and two "maybe" quotes all
independently describe a sharp/hissing 's'. The listener's own guess --
"amplitude peaks" -- plus "over-drive" being a distortion word, suggested a
single mechanism: sibilant frames push samples toward full scale and clip,
and female presets carry more high-frequency energy so they hit it first.

This script tests that hypothesis directly against the one instrument this
project trusts (the bench-7 verdicts), on the WAVs bench 7 already rendered
(all 10 presets, no new rendering needed for the main measurement) plus one
cheap extra render (TOTAL_STEP 8 vs 32 on the flagged F3 clip) for the
denoising-resolution question.

Method
------
1. Locate sibilant frames per clip: faster-whisper tiny.en word timings
   restrict candidate windows to words containing the letters 's' or 'z'
   (a grapheme proxy for /s,z,sh,zh/), then within each candidate window a
   per-clip-adaptive high-frequency-energy criterion (STFT energy >= 4 kHz
   relative to total, top 30% of frames within the window) picks out the
   actual fricative frames rather than the whole word.
2. Peak/clipping stats, computed separately for sibilant-core frames and for
   every other frame in the same clip: peak |sample|, and counts of samples
   >= 0.99 / 0.95 / 0.90.
3. Per-preset comparison of sibilant-frame peak and HF-energy ratio across
   all 10 presets represented in the bench-7 set (2 clips each, 4 for M1).
4. Near-Nyquist spectral energy (20-22.05 kHz of 44.1 kHz native) as an
   aliasing probe, flagged vs clean.
5. TOTAL_STEP 8 vs 32 on the flagged F3 library clip, same seed, comparing
   high-frequency spectral structure -- one new render, done here.

Discipline (CLAUDE.md): every measure is checked against bench 7's verdicts
before being cited as evidence. If a measure does not separate flagged from
clean clips, this script says so plainly instead of presenting it as support.

Usage (from py/):
    python3 phase2a_sibilance.py
"""
import glob
import json
import os
import re

import numpy as np
import soundfile as sf
from scipy.signal import stft

DATA_DIR = "results/listening_sets/phase2a_baseline"
OUT_DIR = "results/phase2a"
OUT_JSON = os.path.join(OUT_DIR, "sibilance.json")

NFFT = 1024
HOP = 256
HF_CUTOFF_HZ = 4000.0
NYQUIST_BAND_LO_HZ = 20000.0  # native sr is 44100 -> Nyquist 22050

# ---------------------------------------------------------------------------
# Ground truth from bench 7 (docs/LISTENING_BENCHES.md #7, docs/GLITCH_MITIGATION.md).
# The clip stems match results/listening_sets/phase2a_baseline/*.wav.
# F1's flagged clip was not individually identified in the writeup (only "F1"
# named among "F1, F3, F5 twice" = 4 flagged clips total); both F1 pool clips
# are kept but marked ambiguous and excluded from the strict flagged-vs-clean
# split below.
# ---------------------------------------------------------------------------
VERDICTS = {
    "pool_F3_t4_seed20361268": dict(
        verdict="flagged", strength="yes_clearly", family="sibilant",
        quote="there is a strong sharp 's' over-drive resulting in a sharp hissing"),
    "pool_F5_t1_seed20361347": dict(
        verdict="flagged", strength="maybe", family="sibilant",
        quote="maybe the 's'-es are a bit sharp, but that might be normal "
              "accumulation (amplitude) peaks given how many there are in "
              "quick succession"),
    "pool_F5_t5_seed20361373": dict(
        verdict="flagged", strength="maybe", family="sibilant",
        quote="not individually quoted; inferred from 'F5 twice' in the 4/8 count"),
    "pool_F1_t3_seed20361166": dict(
        verdict="ambiguous", strength="maybe?", family="sibilant",
        quote="one of the two F1 pool clips was the flagged 'maybe' -- which "
              "one is not stated in the writeup"),
    "pool_F1_t4_seed20361174": dict(
        verdict="ambiguous", strength="maybe?", family="sibilant",
        quote="one of the two F1 pool clips was the flagged 'maybe' -- which "
              "one is not stated in the writeup"),
    "check_seashells_seed20261449": dict(
        verdict="clean_for_sibilance", strength="n/a", family="time_compression",
        quote="flagged for a DIFFERENT, already-explained family (time-condensed "
              "final word / speed default); the same note also said 'maybe "
              "slightly too sharp s-es pronounciation' as a secondary remark, "
              "kept out of the strict split since its primary flag is timing"),
}
# Every other clip in the manifest (M2 x2, M3 x2, M4 x2, M5 x2, F2 x2, F4 x1,
# check_woodchuck x2, check_seashells_20262075) was clean / not flagged.


def preset_of(stem: str) -> str:
    m = re.search(r"_(M[1-5]|F[1-5])_", "_" + stem + "_")
    if m:
        return m.group(1)
    if "check_" in stem:
        return "M1"  # all 4 check clips are M1
    raise ValueError(stem)


def gender_of(preset: str) -> str:
    return "female" if preset.startswith("F") else "male"


def load_clip(path):
    y, sr = sf.read(path, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def stft_mag2(y, sr):
    f, t, Z = stft(y, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    return f, t, (np.abs(Z) ** 2)


def hf_ratio_per_frame(f, mag2):
    hf_mask = f >= HF_CUTOFF_HZ
    total = mag2.sum(axis=0).clip(min=1e-12)
    hf = mag2[hf_mask].sum(axis=0)
    return hf / total


def nyquist_band_ratio(f, mag2, sr):
    band_mask = f >= NYQUIST_BAND_LO_HZ
    total = mag2.sum(axis=0).clip(min=1e-12)
    band = mag2[band_mask].sum(axis=0)
    return float((band / total).mean())


def frame_to_sample_range(frame_idx, hop, n_samples):
    lo = max(0, frame_idx * hop - hop // 2)
    hi = min(n_samples, frame_idx * hop + hop // 2 + 1)
    return lo, hi


def clip_stats(samples):
    a = np.abs(samples)
    if a.size == 0:
        return dict(peak=0.0, n=0, n_ge_099=0, n_ge_095=0, n_ge_090=0)
    return dict(
        peak=float(a.max()),
        n=int(a.size),
        n_ge_099=int((a >= 0.99).sum()),
        n_ge_095=int((a >= 0.95).sum()),
        n_ge_090=int((a >= 0.90).sum()),
    )


def transcribe_words(model, wav_path):
    segments, _info = model.transcribe(wav_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def analyze_clip(model, path):
    stem = os.path.basename(path)[:-4]
    y, sr = load_clip(path)
    n_samples = len(y)

    f, t, mag2 = stft_mag2(y, sr)
    hf_ratio = hf_ratio_per_frame(f, mag2)
    n_frames = mag2.shape[1]

    words = transcribe_words(model, path)
    sib_words = [w for w in words if re.search(r"[sSzZ]", w["word"])]

    PAD = 0.03  # 30ms padding around each candidate word
    sib_frame_mask = np.zeros(n_frames, dtype=bool)
    candidate_frame_mask = np.zeros(n_frames, dtype=bool)
    for w in sib_words:
        f_lo = max(0, int(round((w["start"] - PAD) * sr / HOP)))
        f_hi = min(n_frames, int(round((w["end"] + PAD) * sr / HOP)) + 1)
        if f_hi <= f_lo:
            continue
        candidate_frame_mask[f_lo:f_hi] = True
        window_ratio = hf_ratio[f_lo:f_hi]
        if window_ratio.size == 0:
            continue
        thresh = np.percentile(window_ratio, 70)  # top 30% of the window
        local_sib = window_ratio >= thresh
        sib_frame_mask[f_lo:f_hi] |= local_sib

    # sample-level masks from frame masks
    sib_sample_mask = np.zeros(n_samples, dtype=bool)
    for fi in np.nonzero(sib_frame_mask)[0]:
        lo, hi = frame_to_sample_range(fi, HOP, n_samples)
        sib_sample_mask[lo:hi] = True
    other_sample_mask = ~sib_sample_mask

    sib_stats = clip_stats(y[sib_sample_mask])
    other_stats = clip_stats(y[other_sample_mask])
    whole_stats = clip_stats(y)

    sib_hf_ratio_mean = float(hf_ratio[sib_frame_mask].mean()) if sib_frame_mask.any() else None
    other_hf_ratio_mean = float(hf_ratio[~candidate_frame_mask].mean()) if (~candidate_frame_mask).any() else None

    nyq_ratio = nyquist_band_ratio(f, mag2, sr)

    v = VERDICTS.get(stem, dict(verdict="clean", strength="n/a", family="n/a", quote=None))
    preset = preset_of(stem)

    return dict(
        stem=stem, path=path, sr=sr, n_samples=n_samples,
        preset=preset, gender=gender_of(preset),
        verdict=v["verdict"], strength=v["strength"], family=v["family"], quote=v["quote"],
        words=[w["word"] for w in words],
        sibilant_words=[w["word"] for w in sib_words],
        n_sibilant_frames=int(sib_frame_mask.sum()),
        n_candidate_frames=int(candidate_frame_mask.sum()),
        n_total_frames=n_frames,
        whole_clip=whole_stats,
        sibilant_frames=sib_stats,
        other_frames=other_stats,
        sibilant_hf_ratio_mean=sib_hf_ratio_mean,
        non_candidate_hf_ratio_mean=other_hf_ratio_mean,
        nyquist_band_20_22khz_ratio_mean=nyq_ratio,
    )


def mine_phase2b_generate_manifest():
    """Free evidence: 1920 renders already characterized with peak/finite,
    across all 10 presets, at eps-perturbation conditions (base = single
    preset, not a blend). Reports peak distribution per preset."""
    path = os.path.join(OUT_DIR, "..", "phase2b", "generation_manifest.json")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    from collections import defaultdict
    by_preset = defaultdict(list)
    for r in d["records"]:
        base = r["base"]
        if base.startswith("blend:"):
            continue
        by_preset[base].append(r["peak"])
    out = {}
    all_peaks = []
    for p, arr in by_preset.items():
        arr = np.array(arr)
        all_peaks.append(arr)
        out[p] = dict(
            n=int(arr.size), mean=float(arr.mean()), median=float(np.median(arr)),
            max=float(arr.max()), p95=float(np.percentile(arr, 95)),
            n_ge_090=int((arr >= 0.90).sum()), n_ge_099=int((arr >= 0.99).sum()),
        )
    all_peaks = np.concatenate(all_peaks)
    n_non_finite = sum(1 for r in d["records"] if not r["finite"])
    return dict(
        source=path, n_total_records=len(d["records"]), n_eps_condition_records=int(all_peaks.size),
        per_preset=out, overall_max_peak=float(all_peaks.max()),
        overall_n_ge_099=int((all_peaks >= 0.99).sum()), overall_n_ge_090=int((all_peaks >= 0.90).sum()),
        n_non_finite=n_non_finite,
    )


def render_step_comparison():
    """TOTAL_STEP 8 vs 32 on the flagged F3 library clip, same seed and style.
    One new render (step-32); step-8 is re-rendered too rather than reusing
    the WAV on disk, so both are compared from the identical code path."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from helper import load_text_to_speech, load_voice_style
    from phase2b_generate import LANG, ONNX_DIR, SPEED, TEXTS, VOICE_STYLE_DIR

    SEED = 20361268
    TEXT = TEXTS[4]
    assert TEXT.startswith("The library closes early"), TEXT

    print("Loading TTS engine for step-count comparison ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    style = load_voice_style([os.path.join(VOICE_STYLE_DIR, "F3.json")])

    results = {}
    for total_step in (8, 32):
        np.random.seed(SEED)
        wav, dur = tts(TEXT, LANG, style, total_step, SPEED)
        trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
        out_path = os.path.join(OUT_DIR, f"f3_library_seed{SEED}_step{total_step}.wav")
        sf.write(out_path, trimmed, tts.sample_rate, subtype="PCM_16")

        f, t, mag2 = stft_mag2(trimmed.astype(np.float64), tts.sample_rate)
        hf_ratio = hf_ratio_per_frame(f, mag2)
        nyq = nyquist_band_ratio(f, mag2, tts.sample_rate)
        results[total_step] = dict(
            path=out_path, peak=float(np.abs(trimmed).max()),
            hf_ratio_mean=float(hf_ratio.mean()), hf_ratio_max=float(hf_ratio.max()),
            nyquist_band_ratio_mean=nyq,
        )
    return results


def summarize_flagged_vs_clean(clips):
    strict = [c for c in clips if c["verdict"] in ("flagged", "clean")]
    flagged = [c for c in strict if c["verdict"] == "flagged"]
    clean = [c for c in strict if c["verdict"] == "clean"]

    def arr(cs, key_path):
        vals = []
        for c in cs:
            v = c
            for k in key_path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return np.array(vals)

    checks = {}
    for name, key_path in [
        ("sibilant_frame_peak", ("sibilant_frames", "peak")),
        ("sibilant_frame_n_ge_090", ("sibilant_frames", "n_ge_090")),
        ("sibilant_hf_ratio_mean", ("sibilant_hf_ratio_mean",)),
        ("whole_clip_peak", ("whole_clip", "peak")),
        ("nyquist_band_ratio", ("nyquist_band_20_22khz_ratio_mean",)),
    ]:
        fv = arr(flagged, key_path)
        cv = arr(clean, key_path)
        if fv.size == 0 or cv.size == 0:
            checks[name] = dict(separates=None, note="insufficient data")
            continue
        checks[name] = dict(
            flagged_mean=float(fv.mean()), flagged_min=float(fv.min()), flagged_max=float(fv.max()),
            clean_mean=float(cv.mean()), clean_min=float(cv.min()), clean_max=float(cv.max()),
            overlap=bool(fv.min() <= cv.max() and cv.min() <= fv.max()),
            flagged_gt_clean_mean=bool(fv.mean() > cv.mean()),
        )
    return dict(n_flagged=len(flagged), n_clean=len(clean), checks=checks)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from faster_whisper import WhisperModel

    print("Loading faster-whisper tiny.en (local, int8, cpu) ...", flush=True)
    model = WhisperModel(
        "Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8", local_files_only=True,
    )

    wav_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    print(f"Analyzing {len(wav_paths)} clips from {DATA_DIR} ...")
    clips = []
    for p in wav_paths:
        c = analyze_clip(model, p)
        clips.append(c)
        print(f"  {c['stem']:35s} preset={c['preset']} verdict={c['verdict']:20s} "
              f"sib_peak={c['sibilant_frames']['peak']:.4f} sib_hf={c['sibilant_hf_ratio_mean']}")

    print("\nMining phase2b_generate's existing 1920-render manifest for free peak evidence ...")
    manifest_evidence = mine_phase2b_generate_manifest()

    print("\nRendering TOTAL_STEP 8 vs 32 comparison on F3 library, seed 20361268 ...")
    step_comparison = render_step_comparison()

    calibration = summarize_flagged_vs_clean(clips)

    # per-preset aggregate over the bench-7 set
    from collections import defaultdict
    per_preset = defaultdict(list)
    for c in clips:
        per_preset[c["preset"]].append(c)
    per_preset_summary = {}
    for p, cs in per_preset.items():
        sib_peaks = [c["sibilant_frames"]["peak"] for c in cs]
        sib_hf = [c["sibilant_hf_ratio_mean"] for c in cs if c["sibilant_hf_ratio_mean"] is not None]
        per_preset_summary[p] = dict(
            gender=gender_of(p), n_clips=len(cs),
            mean_sibilant_peak=float(np.mean(sib_peaks)) if sib_peaks else None,
            mean_sibilant_hf_ratio=float(np.mean(sib_hf)) if sib_hf else None,
            any_flagged=any(c["verdict"] == "flagged" for c in cs),
        )

    out = dict(
        experiment="phase2a_sibilance",
        hypothesis="sibilant frames push samples toward full scale and clip; female "
                   "presets carry more HF energy and hit it first",
        data_dir=DATA_DIR,
        nfft=NFFT, hop=HOP, hf_cutoff_hz=HF_CUTOFF_HZ, nyquist_band_lo_hz=NYQUIST_BAND_LO_HZ,
        clips=clips,
        per_preset_summary=per_preset_summary,
        phase2b_generate_manifest_evidence=manifest_evidence,
        step_count_comparison=step_comparison,
        flagged_vs_clean_calibration=calibration,
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")

    print("\n=== Calibration summary (flagged vs clean) ===")
    for name, res in calibration["checks"].items():
        print(f"  {name}: {res}")


if __name__ == "__main__":
    main()
