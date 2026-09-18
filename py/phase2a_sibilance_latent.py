"""Phase 2a -- sibilant over-drive: latent vs. vocoder.

Deconstructs the pipeline style_ttl -> text_encoder -> vector_estimator (8
steps) -> **latent** -> vocoder -> waveform, to ask which side of the vocoder
the sibilant over-drive artifact (docs/GLITCH_MITIGATION.md, bench 7) lives
on. Nobody has looked at the latent (`xt` in `helper.py`'s `TextToSpeech._infer`,
just before `self.vocoder_ort.run`) before this script.

`helper.py` is not modified. `infer_capture_latent()` below reproduces
`_infer`'s body line-for-line (same ORT calls, same seeding convention:
`np.random.seed(seed)` immediately before the noisy-latent draw, matching
every other script in this project that needs a reproducible render) and
additionally returns the final denoised latent `xt`.

Two renders sets:

1. **Cross-preset, fixed text/seed** (task's main ask): all 10 shipped
   presets render the library sentence ("The library closes early on
   Thursday...") at the same seed (20361268, F3's own flagged bench-7 seed,
   for continuity) and this fork's speed default (1.0). Latent and waveform
   are compared preset-to-preset with text and seed held constant, which
   bench 7's mixed-text pool never gave us.
2. **Bench-7 pool reproduction** (validation): the 16 "pool" clips of
   `results/listening_sets/phase2a_baseline` (3 flagged: F3 t4, F5 t1, F5 t5;
   2 ambiguous, excluded like the original script; 11 clean) are re-rendered
   bit-for-bit (same preset/text/seed/speed=1.05/step=8 as the original bench
   render, `benches/phase2a_baseline_bench.py:298-301`) so latent-side
   measures can be checked against real listener verdicts, not just eyeballed
   across an unlabelled 10-preset set. The 4 "check_*" clips are skipped --
   they were copied in from a different script's seed pool
   (`results/listening_sets/phase2a_seed_variance`) and reproducing them
   bit-identically here is not confirmed, unlike the pool clips whose
   render call is documented in the bench script itself.

Sibilant localization: faster-whisper tiny.en word timings (words containing
s/z) + the per-window top-30%-of-high-frequency-ratio criterion, verbatim
from `phase2a_sibilance.py` (imported, not reimplemented) -> a sample-level
sibilant mask on the waveform. That mask is pushed into latent time by
integer division by `chunk_size = base_chunk_size * chunk_compress_factor`
samples/latent-frame, read from `assets/onnx/tts.json` (512 * 6 = 3072
samples = ~69.7 ms/frame at 44.1 kHz), never assumed.

Discipline (CLAUDE.md): every measure here is checked against bench 7's real
flagged/clean pool-clip verdicts before being cited. If a measure does not
separate them, this script says so instead of presenting a table.

Usage (from py/):
    python3 phase2a_sibilance_latent.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper import chunk_text, load_text_to_speech, load_voice_style  # noqa: E402
from phase2a_sibilance import (  # noqa: E402
    HF_CUTOFF_HZ,
    HOP,
    NFFT,
    frame_to_sample_range,
    hf_ratio_per_frame,
    stft_mag2,
    transcribe_words,
)
from phase2b_generate import LANG, ONNX_DIR, TEXTS, VOICE_STYLE_DIR  # noqa: E402

OUT_DIR = "results/phase2a"
AUDIO_DIR = os.path.join(OUT_DIR, "sibilance_latent_audio")
OUT_JSON = os.path.join(OUT_DIR, "sibilance_latent.json")

LIBRARY_TEXT_IDX = 4
LIBRARY_TEXT = TEXTS[LIBRARY_TEXT_IDX]
assert LIBRARY_TEXT.startswith("The library closes early")

MAIN_SEED = 20361268           # F3's own bench-7 flagged seed, reused for continuity
MAIN_SPEED = 1.0                # this fork's default (see CLAUDE.md)
MAIN_TOTAL_STEP = 8
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

FLAGGED_PRESETS = ["F3", "F5"]   # per the task brief
CLEAN_PRESETS = ["M3", "M4", "F2"]

BAND_LO_HZ = 4000.0
BAND_HI_HZ = 11000.0

BENCH7_MANIFEST = "results/listening_sets/phase2a_baseline/manifest.json"
BENCH7_SPEED = 1.05              # original bench-7 render speed (see benches/phase2a_baseline_bench.py)
BENCH7_TOTAL_STEP = 8


# ---------------------------------------------------------------------------
# Inference path reproduced from helper.TextToSpeech._infer, with the latent
# captured. helper.py itself is untouched.
# ---------------------------------------------------------------------------
def infer_capture_latent(tts, text, lang, style, total_step, speed, seed):
    max_len = 120 if lang in ("ko", "ja") else 300
    text_list = chunk_text(text, max_len=max_len)
    assert len(text_list) == 1, "library/pool sentences must fit in one chunk"
    text = text_list[0]

    np.random.seed(seed)  # same convention as every render script in this project
    text_ids, text_mask = tts.text_processor([text], [lang])
    dur_onnx, *_ = tts.dp_ort.run(
        None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    )
    dur_onnx = dur_onnx / speed
    text_emb_onnx, *_ = tts.text_enc_ort.run(
        None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
    )
    xt, latent_mask = tts.sample_noisy_latent(dur_onnx)
    total_step_np = np.array([total_step], dtype=np.float32)
    for step in range(total_step):
        current_step = np.array([step], dtype=np.float32)
        xt, *_ = tts.vector_est_ort.run(
            None,
            {
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            },
        )
    latent_final = xt.copy()  # [1, ldim*chunk_compress_factor, latent_len] -- this IS `xt` from _infer
    wav, *_ = tts.vocoder_ort.run(None, {"latent": xt})
    dur_s = float(dur_onnx[0].item())
    trimmed = wav[0, : int(tts.sample_rate * dur_s)].astype(np.float32)
    return dict(
        wav=trimmed, sr=tts.sample_rate, dur_s=dur_s,
        latent=latent_final[0],  # [C, T_latent]
        latent_mask=latent_mask[0, 0].astype(bool),  # [T_latent]
    )


# ---------------------------------------------------------------------------
# Sibilant localization (waveform-sample level), reusing phase2a_sibilance's
# exact word-timing + HF-ratio-percentile criterion.
# ---------------------------------------------------------------------------
def locate_sibilant_samples(whisper_model, wav_path, y, sr):
    import re
    f, t, mag2 = stft_mag2(y, sr)
    hf_ratio = hf_ratio_per_frame(f, mag2)
    n_frames = mag2.shape[1]
    n_samples = len(y)

    words = transcribe_words(whisper_model, wav_path)
    sib_words = [w for w in words if re.search(r"[sSzZ]", w["word"])]

    PAD = 0.03
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
        thresh = np.percentile(window_ratio, 70)
        sib_frame_mask[f_lo:f_hi] |= window_ratio >= thresh

    sib_sample_mask = np.zeros(n_samples, dtype=bool)
    for fi in np.nonzero(sib_frame_mask)[0]:
        lo, hi = frame_to_sample_range(fi, HOP, n_samples)
        sib_sample_mask[lo:hi] = True

    return dict(
        words=[w["word"] for w in words], sibilant_words=[w["word"] for w in sib_words],
        sib_sample_mask=sib_sample_mask, sib_frame_mask=sib_frame_mask,
        candidate_frame_mask=candidate_frame_mask, hf_ratio=hf_ratio, f=f, mag2=mag2,
        n_stft_frames=n_frames,
    )


def band_ratio_per_frame(f, mag2, lo_hz, hi_hz):
    band_mask = (f >= lo_hz) & (f <= hi_hz)
    total = mag2.sum(axis=0).clip(min=1e-12)
    band = mag2[band_mask].sum(axis=0)
    return band / total


def contiguous_runs(mask):
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, len(mask)))
    return runs


def waveform_measures(sr, sib_info):
    """Task step 4: band energy, fricative duration, onset sharpness -- all
    ratio/rate quantities, so no separate level-matching is needed (CLAUDE.md's
    level-matching rule is about magnitude-sensitive comparisons; these are
    scale-invariant by construction, like the existing hf_ratio measure)."""
    f, mag2 = sib_info["f"], sib_info["mag2"]
    hf_ratio = sib_info["hf_ratio"]
    band_ratio = band_ratio_per_frame(f, mag2, BAND_LO_HZ, BAND_HI_HZ)
    sib_mask = sib_info["sib_frame_mask"]
    non_sib_mask = ~sib_info["candidate_frame_mask"]

    runs = contiguous_runs(sib_mask)
    frame_dur_s = HOP / sr
    fricative_durations_s = [(b - a) * frame_dur_s for a, b in runs]

    onset_sharpness = []
    for a, b in runs:
        lookback = 2
        if a - lookback >= 0:
            onset_sharpness.append(float(hf_ratio[min(b - 1, a + 1)] - hf_ratio[a - lookback]))
    return dict(
        n_sibilant_runs=len(runs),
        fricative_duration_s_mean=float(np.mean(fricative_durations_s)) if fricative_durations_s else None,
        fricative_duration_s_total=float(np.sum(fricative_durations_s)) if fricative_durations_s else 0.0,
        onset_sharpness_mean=float(np.mean(onset_sharpness)) if onset_sharpness else None,
        band_4_11k_ratio_sibilant_mean=float(band_ratio[sib_mask].mean()) if sib_mask.any() else None,
        band_4_11k_ratio_nonsibilant_mean=float(band_ratio[non_sib_mask].mean()) if non_sib_mask.any() else None,
        hf_ratio_sibilant_mean=float(hf_ratio[sib_mask].mean()) if sib_mask.any() else None,
        hf_ratio_nonsibilant_mean=float(hf_ratio[non_sib_mask].mean()) if non_sib_mask.any() else None,
    )


def sample_mask_to_latent_mask(sib_sample_mask, n_samples, chunk_size, n_latent_frames, frac_thresh=0.5):
    """A latent frame li covers waveform samples [li*chunk_size, (li+1)*chunk_size).
    Marked sibilant if >= frac_thresh of its samples are sibilant."""
    lat_mask = np.zeros(n_latent_frames, dtype=bool)
    lat_frac = np.zeros(n_latent_frames, dtype=float)
    for li in range(n_latent_frames):
        lo = li * chunk_size
        hi = min(n_samples, (li + 1) * chunk_size)
        if hi <= lo:
            continue
        frac = sib_sample_mask[lo:hi].mean()
        lat_frac[li] = frac
        lat_mask[li] = frac >= frac_thresh
    return lat_mask, lat_frac


def latent_measures(latent, latent_valid_mask, lat_sib_mask):
    """latent: [C, T]. Per-frame L2 norm, per-frame variance across channels,
    and frame-to-frame delta norm (temporal derivative)."""
    C, T = latent.shape
    norm = np.linalg.norm(latent, axis=0)               # [T]
    var = latent.var(axis=0)                              # [T]
    delta = np.diff(latent, axis=1)                        # [C, T-1]
    delta_norm = np.linalg.norm(delta, axis=0)             # [T-1]
    delta_norm_padded = np.concatenate([[np.nan], delta_norm])  # align to frame i (diff from i-1)

    valid = latent_valid_mask.copy()
    sib = lat_sib_mask & valid
    non_sib = valid & ~lat_sib_mask

    def safe_mean(arr, mask):
        v = arr[mask]
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else None

    def safe_std(arr, mask):
        v = arr[mask]
        v = v[~np.isnan(v)]
        return float(v.std()) if v.size else None

    norm_sib, norm_non = safe_mean(norm, sib), safe_mean(norm, non_sib)
    var_sib, var_non = safe_mean(var, sib), safe_mean(var, non_sib)
    dn_sib, dn_non = safe_mean(delta_norm_padded, sib), safe_mean(delta_norm_padded, non_sib)
    dn_std_non = safe_std(delta_norm_padded, non_sib)

    return dict(
        n_latent_frames=int(T), n_sibilant_latent_frames=int(sib.sum()),
        n_nonsibilant_latent_frames=int(non_sib.sum()),
        norm_sibilant_mean=norm_sib, norm_nonsibilant_mean=norm_non,
        norm_ratio=(norm_sib / norm_non) if (norm_sib is not None and norm_non) else None,
        var_sibilant_mean=var_sib, var_nonsibilant_mean=var_non,
        var_ratio=(var_sib / var_non) if (var_sib is not None and var_non) else None,
        delta_norm_sibilant_mean=dn_sib, delta_norm_nonsibilant_mean=dn_non,
        delta_norm_ratio=(dn_sib / dn_non) if (dn_sib is not None and dn_non) else None,
        delta_norm_zscore=((dn_sib - dn_non) / dn_std_non) if (dn_sib is not None and dn_non is not None and dn_std_non) else None,
    )


def render_and_analyze(tts, whisper_model, preset, text, seed, speed, total_step, style_path_dir, stem, save_audio=True):
    style = load_voice_style([os.path.join(style_path_dir, f"{preset}.json")])
    r = infer_capture_latent(tts, text, LANG, style, total_step, speed, seed)
    wav_path = os.path.join(AUDIO_DIR, f"{stem}.wav")
    if save_audio:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        sf.write(wav_path, r["wav"], r["sr"], subtype="PCM_16")
    else:
        # whisper still needs a file; write to a scratch path without keeping it
        os.makedirs(AUDIO_DIR, exist_ok=True)
        sf.write(wav_path, r["wav"], r["sr"], subtype="PCM_16")

    y = r["wav"].astype(np.float64)
    sib_info = locate_sibilant_samples(whisper_model, wav_path, y, r["sr"])
    wave_meas = waveform_measures(r["sr"], sib_info)

    n_samples = len(y)
    n_latent_frames = r["latent"].shape[1]
    chunk_size = tts.base_chunk_size * tts.chunk_compress_factor
    lat_sib_mask, lat_frac = sample_mask_to_latent_mask(
        sib_info["sib_sample_mask"], n_samples, chunk_size, n_latent_frames
    )
    lat_meas = latent_measures(r["latent"], r["latent_mask"], lat_sib_mask)

    return dict(
        stem=stem, preset=preset, seed=seed, speed=speed, total_step=total_step,
        text=text, wav_path=wav_path, dur_s=r["dur_s"], n_samples=n_samples,
        chunk_size_samples=chunk_size, n_latent_frames=n_latent_frames,
        n_sibilant_latent_frames=int(lat_sib_mask.sum()),
        words=sib_info["words"], sibilant_words=sib_info["sibilant_words"],
        n_sibilant_stft_frames=int(sib_info["sib_frame_mask"].sum()),
        waveform=wave_meas, latent=lat_meas,
    )


def summarize_group(records, presets):
    sub = [r for r in records if r["preset"] in presets]
    out = {}
    for key_path, label in [
        (("latent", "delta_norm_ratio"), "latent_delta_norm_ratio"),
        (("latent", "delta_norm_zscore"), "latent_delta_norm_zscore"),
        (("latent", "norm_ratio"), "latent_norm_ratio"),
        (("latent", "var_ratio"), "latent_var_ratio"),
        (("waveform", "band_4_11k_ratio_sibilant_mean"), "wave_band_4_11k_sibilant"),
        (("waveform", "hf_ratio_sibilant_mean"), "wave_hf_ratio_sibilant"),
        (("waveform", "onset_sharpness_mean"), "wave_onset_sharpness"),
    ]:
        vals = []
        for r in sub:
            v = r
            for k in key_path:
                v = v[k] if v is not None else None
            if v is not None:
                vals.append(v)
        out[label] = dict(n=len(vals), mean=float(np.mean(vals)) if vals else None,
                           values=[round(v, 5) for v in vals])
    return out


def reference_sample_check(whisper_model):
    """Task step 5: compare upstream's own audio_samples/, if sibilants are
    present, against our generated clips' sibilant spectral profile. These are
    different speakers, different recordings/mics and (almost certainly)
    different sentences from our corpus, so this is at best a directional
    sanity check, not a matched comparison -- flagged explicitly below."""
    samples_dir = os.path.join("..", "assets", "audio_samples")
    samples_dir = os.path.normpath(samples_dir)
    results = []
    if not os.path.isdir(samples_dir):
        return dict(available=False, note="assets/audio_samples not found"), False
    import glob
    paths = sorted(glob.glob(os.path.join(samples_dir, "*_supertonic3.wav"))) + \
        sorted(glob.glob(os.path.join(samples_dir, "*_reference.wav")))
    for p in paths:
        y, sr = sf.read(p, dtype="float64")
        if y.ndim > 1:
            y = y.mean(axis=1)
        try:
            sib_info = locate_sibilant_samples(whisper_model, p, y, sr)
        except Exception as e:
            results.append(dict(path=p, error=str(e)))
            continue
        wave_meas = waveform_measures(sr, sib_info)
        results.append(dict(
            path=p, sr=sr, dur_s=len(y) / sr,
            transcript=" ".join(sib_info["words"]),
            sibilant_words=sib_info["sibilant_words"],
            n_sibilant_stft_frames=int(sib_info["sib_frame_mask"].sum()),
            waveform=wave_meas,
        ))
    return dict(available=True, samples_dir=samples_dir, results=results), True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    from faster_whisper import WhisperModel

    print("Loading faster-whisper tiny.en ...", flush=True)
    whisper_model = WhisperModel(
        "Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8", local_files_only=True,
    )

    print("Loading TTS engine ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    print(f"base_chunk_size={tts.base_chunk_size} chunk_compress_factor={tts.chunk_compress_factor} "
          f"-> {tts.base_chunk_size * tts.chunk_compress_factor} samples/latent-frame "
          f"({tts.base_chunk_size * tts.chunk_compress_factor / tts.sample_rate * 1000:.2f} ms) "
          f"at sr={tts.sample_rate}")

    # ---- Part A: cross-preset, fixed text + seed -------------------------
    print("\n=== Part A: 10 presets, library sentence, fixed seed, speed=1.0 ===")
    main_records = []
    for preset in PRESETS:
        stem = f"main_{preset}_library_seed{MAIN_SEED}"
        r = render_and_analyze(
            tts, whisper_model, preset, LIBRARY_TEXT, MAIN_SEED, MAIN_SPEED, MAIN_TOTAL_STEP,
            VOICE_STYLE_DIR, stem,
        )
        main_records.append(r)
        print(f"  {preset}: n_lat={r['n_latent_frames']:3d} n_sib_lat={r['n_sibilant_latent_frames']:2d} "
              f"delta_norm_ratio={r['latent']['delta_norm_ratio']} "
              f"wave_band_sib={r['waveform']['band_4_11k_ratio_sibilant_mean']}")

    flagged_summary = summarize_group(main_records, FLAGGED_PRESETS)
    clean_summary = summarize_group(main_records, CLEAN_PRESETS)
    all_summary = summarize_group(main_records, PRESETS)

    # ---- Part B: bench-7 pool reproduction, for validation ----------------
    print("\n=== Part B: bench-7 pool clips, re-rendered with latent capture ===")
    with open(BENCH7_MANIFEST) as fh:
        manifest = json.load(fh)
    from phase2a_sibilance import VERDICTS

    pool_clips = [c for c in manifest["clips"] if c["kind"] == "pool"]
    pool_records = []
    bit_diffs = []
    for c in pool_clips:
        stem = c["stem"]
        r = render_and_analyze(
            tts, whisper_model, c["preset"], c["text"], c["seed"], BENCH7_SPEED, BENCH7_TOTAL_STEP,
            VOICE_STYLE_DIR, f"pool_repro_{stem}",
        )
        v = VERDICTS.get(stem, dict(verdict="clean", family="n/a"))
        r["verdict"] = v["verdict"]
        pool_records.append(r)

        # sanity: compare against the ORIGINAL bench-7 WAV on disk (bit-for-bit
        # reproduction check of our own infer_capture_latent)
        orig_path = os.path.join("results/listening_sets/phase2a_baseline", f"{stem}.wav")
        if os.path.exists(orig_path):
            y_orig, _ = sf.read(orig_path, dtype="float32")
            y_new, _ = sf.read(r["wav_path"], dtype="float32")
            n = min(len(y_orig), len(y_new))
            diff = float(np.abs(y_orig[:n] - y_new[:n]).max()) if n else None
            bit_diffs.append(diff)
        print(f"  {stem}: verdict={r['verdict']:10s} delta_norm_ratio={r['latent']['delta_norm_ratio']}")

    reproduction_check = dict(
        n_compared=len(bit_diffs),
        max_abs_diff=float(np.max(bit_diffs)) if bit_diffs else None,
        mean_abs_diff=float(np.mean(bit_diffs)) if bit_diffs else None,
        note="max abs diff between our infer_capture_latent reproduction and the original bench-7 "
             "WAV (PCM_16 on disk, so ~1/32768 quantization noise is expected even for a bit-exact "
             "ORT replay); large values would mean our reproduction of _infer is wrong.",
    )

    strict = [r for r in pool_records if r["verdict"] in ("flagged", "clean")]
    flagged = [r for r in strict if r["verdict"] == "flagged"]
    clean = [r for r in strict if r["verdict"] == "clean"]

    def arr(records, key_path):
        vals = []
        for r in records:
            v = r
            for k in key_path:
                v = v[k] if v is not None else None
            if v is not None:
                vals.append(v)
        return np.array(vals)

    validation = {}
    for name, key_path in [
        ("latent_delta_norm_ratio", ("latent", "delta_norm_ratio")),
        ("latent_delta_norm_zscore", ("latent", "delta_norm_zscore")),
        ("latent_norm_ratio", ("latent", "norm_ratio")),
        ("latent_var_ratio", ("latent", "var_ratio")),
        ("wave_band_4_11k_sibilant", ("waveform", "band_4_11k_ratio_sibilant_mean")),
        ("wave_hf_ratio_sibilant", ("waveform", "hf_ratio_sibilant_mean")),
        ("wave_onset_sharpness", ("waveform", "onset_sharpness_mean")),
    ]:
        fv = arr(flagged, key_path)
        cv = arr(clean, key_path)
        if fv.size == 0 or cv.size == 0:
            validation[name] = dict(separates=None, note="insufficient data")
            continue
        overlap = bool(fv.min() <= cv.max() and cv.min() <= fv.max())
        validation[name] = dict(
            n_flagged=int(fv.size), n_clean=int(cv.size),
            flagged_mean=float(fv.mean()), flagged_min=float(fv.min()), flagged_max=float(fv.max()),
            clean_mean=float(cv.mean()), clean_min=float(cv.min()), clean_max=float(cv.max()),
            overlap=overlap, separates=not overlap,
            flagged_gt_clean_mean=bool(fv.mean() > cv.mean()),
        )

    # ---- Part C: upstream reference-sample check --------------------------
    print("\n=== Part C: upstream audio_samples/ sibilance check ===")
    ref_check, ref_available = reference_sample_check(whisper_model)

    out = dict(
        experiment="phase2a_sibilance_latent",
        chunk_size_samples=tts.base_chunk_size * tts.chunk_compress_factor,
        base_chunk_size=tts.base_chunk_size, chunk_compress_factor=tts.chunk_compress_factor,
        sample_rate=tts.sample_rate,
        main_comparison=dict(
            text=LIBRARY_TEXT, seed=MAIN_SEED, speed=MAIN_SPEED, total_step=MAIN_TOTAL_STEP,
            records=main_records,
            flagged_presets=FLAGGED_PRESETS, clean_presets=CLEAN_PRESETS,
            flagged_summary=flagged_summary, clean_summary=clean_summary, all_preset_summary=all_summary,
        ),
        bench7_pool_reproduction=dict(
            records=pool_records, reproduction_check=reproduction_check,
            flagged_vs_clean_validation=validation,
            n_flagged=len(flagged), n_clean=len(clean),
        ),
        upstream_reference_check=ref_check,
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")

    print("\n=== Validation summary (bench-7 pool, flagged vs clean) ===")
    for name, res in validation.items():
        print(f"  {name}: {res}")

    print("\n=== Reproduction sanity (our infer_capture_latent vs original bench-7 WAVs) ===")
    print(f"  {reproduction_check}")


if __name__ == "__main__":
    main()
