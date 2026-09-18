"""Phase 2a -- how many latent frames does one sibilant actually span, and
what happens when only those frames are attenuated.

Task 1: frame-span geometry, on 4 clips (2 flagged, 2 clean, same sentences):
  F3 library seed20361268 (flagged), M4 library seed20361077 (clean)
  F5 seashells seed20361347 (flagged), F2 seashells seed20361205 (clean)

Task 2: direct manipulation -- capture the latent for the flagged clips,
attenuate ONLY the latent frames covering their sibilants by a factor, swept
over {1.0, 0.9, 0.8, 0.7, 0.6}, re-decode with the vocoder alone (no
re-running the denoising loop), and report peaks in/out of the sibilant
region plus whether the rest of the clip and the words are intact.

Uses phase2a_sibilance_latent.py's infer_capture_latent (re-imported, not
reimplemented) so the render path is identical/validated (bit-exact vs the
original bench-7 WAVs, per that script's own reproduction_check).

Usage (from py/):
    python3 phase2a_latent_frame_span.py
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper import load_text_to_speech, load_voice_style  # noqa: E402
from phase2a_sibilance import HOP, frame_to_sample_range, hf_ratio_per_frame, stft_mag2, transcribe_words  # noqa: E402
from phase2a_sibilance_latent import infer_capture_latent, contiguous_runs  # noqa: E402
from phase2b_generate import LANG, ONNX_DIR, VOICE_STYLE_DIR  # noqa: E402

OUT_DIR = "results/phase2a"
AUDIO_DIR = os.path.join(OUT_DIR, "frame_span_audio")
OUT_JSON = os.path.join(OUT_DIR, "latent_frame_span.json")
BENCH7_SPEED = 1.05
BENCH7_TOTAL_STEP = 8

CLIPS = [
    dict(name="F3_library_flagged", preset="F3", seed=20361268,
         text="The library closes early on Thursday, so bring your books back."),
    dict(name="M4_library_clean", preset="M4", seed=20361077,
         text="The library closes early on Thursday, so bring your books back."),
    dict(name="F5_seashells_flagged", preset="F5", seed=20361347,
         text="She sells seashells by the sea shore every summer morning."),
    dict(name="F2_seashells_clean", preset="F2", seed=20361205,
         text="She sells seashells by the sea shore every summer morning."),
]

ATTEN_FACTORS = [1.0, 0.9, 0.8, 0.7, 0.6]


def locate_sibilant_runs(whisper_model, wav_path, y, sr):
    """Per-sibilant-word runs on the STFT-frame grid (not merged across words),
    using the exact HF-ratio-percentile criterion from phase2a_sibilance."""
    f, t, mag2 = stft_mag2(y, sr)
    hf_ratio = hf_ratio_per_frame(f, mag2)
    n_frames = mag2.shape[1]
    n_samples = len(y)

    words = transcribe_words(whisper_model, wav_path)
    sib_words = [w for w in words if re.search(r"[sSzZ]", w["word"])]

    PAD = 0.03
    per_word_runs = []
    for w in sib_words:
        f_lo = max(0, int(round((w["start"] - PAD) * sr / HOP)))
        f_hi = min(n_frames, int(round((w["end"] + PAD) * sr / HOP)) + 1)
        if f_hi <= f_lo:
            continue
        window_ratio = hf_ratio[f_lo:f_hi]
        if window_ratio.size == 0:
            continue
        thresh = np.percentile(window_ratio, 70)
        local_mask = window_ratio >= thresh
        # contiguous runs of the fricative-core mask *within this word's window*
        for a, b in contiguous_runs(local_mask):
            run_lo, run_hi = f_lo + a, f_lo + b
            s_lo, _ = frame_to_sample_range(run_lo, HOP, n_samples)
            _, s_hi = frame_to_sample_range(run_hi - 1, HOP, n_samples)
            per_word_runs.append(dict(
                word=w["word"], word_start_s=w["start"], word_end_s=w["end"],
                stft_frame_lo=run_lo, stft_frame_hi=run_hi,
                sample_lo=s_lo, sample_hi=s_hi,
                duration_s=(s_hi - s_lo) / sr,
            ))
    return per_word_runs, dict(f=f, mag2=mag2, hf_ratio=hf_ratio, n_samples=n_samples, words=words)


def frame_span_of_run(sample_lo, sample_hi, chunk_size, n_latent_frames):
    li_lo = sample_lo // chunk_size
    li_hi = (sample_hi - 1) // chunk_size
    li_hi = min(li_hi, n_latent_frames - 1)
    n_frames_spanned = li_hi - li_lo + 1
    occupancies = []
    for li in range(li_lo, li_hi + 1):
        frame_lo, frame_hi = li * chunk_size, (li + 1) * chunk_size
        overlap_lo = max(frame_lo, sample_lo)
        overlap_hi = min(frame_hi, sample_hi)
        frac_of_frame = max(0, overlap_hi - overlap_lo) / chunk_size
        occupancies.append(dict(latent_frame=int(li), frac_of_frame_occupied_by_sibilant=round(float(frac_of_frame), 4)))
    return int(li_lo), int(li_hi), int(n_frames_spanned), occupancies


def attenuate_and_decode(tts, latent, latent_frames, factor):
    """Zero in on ONLY the given latent frame indices, scale by `factor`,
    leave every other frame untouched, and run the vocoder alone (no
    re-running text_enc/vector_estimator -- this is a post-hoc decode of an
    already-denoised latent, exactly what Task 2 asks for)."""
    lat = latent.copy()  # [C, T]
    lat[:, latent_frames] *= factor
    wav, *_ = tts.vocoder_ort.run(None, {"latent": lat[None, ...]})
    return wav[0]


def peak_in_mask(y, sample_mask):
    seg = y[sample_mask]
    return float(np.abs(seg).max()) if seg.size else None


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
    chunk_size = tts.base_chunk_size * tts.chunk_compress_factor
    print(f"chunk_size_samples={chunk_size} sr={tts.sample_rate} "
          f"ms_per_frame={chunk_size / tts.sample_rate * 1000:.3f}")

    task1_results = []
    manipulation_targets = {}  # name -> dict(latent, sib_frames, sr, wav_path, sample_mask)

    for c in CLIPS:
        style = load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{c['preset']}.json")])
        r = infer_capture_latent(tts, c["text"], LANG, style, BENCH7_TOTAL_STEP, BENCH7_SPEED, c["seed"])
        wav_path = os.path.join(AUDIO_DIR, f"{c['name']}.wav")
        sf.write(wav_path, r["wav"], r["sr"], subtype="PCM_16")

        y = r["wav"].astype(np.float64)
        n_latent_frames = r["latent"].shape[1]
        runs, ctx = locate_sibilant_runs(whisper_model, wav_path, y, r["sr"])

        run_records = []
        all_sib_frames = set()
        for run in runs:
            li_lo, li_hi, n_spanned, occ = frame_span_of_run(
                run["sample_lo"], run["sample_hi"], chunk_size, n_latent_frames
            )
            for li in range(li_lo, li_hi + 1):
                all_sib_frames.add(li)
            run_records.append(dict(**run, latent_frame_lo=li_lo, latent_frame_hi=li_hi,
                                     n_latent_frames_spanned=n_spanned, per_frame_occupancy=occ))

        n_spans = [rr["n_latent_frames_spanned"] for rr in run_records]
        task1_results.append(dict(
            name=c["name"], preset=c["preset"], seed=c["seed"], text=c["text"],
            n_latent_frames_total=n_latent_frames, dur_s=r["dur_s"],
            n_sibilant_runs=len(run_records),
            frame_spans=n_spans,
            mean_frames_spanned=float(np.mean(n_spans)) if n_spans else None,
            max_frames_spanned=int(np.max(n_spans)) if n_spans else None,
            runs=run_records,
        ))
        print(f"{c['name']}: {len(run_records)} sibilant runs, spans={n_spans}")

        manipulation_targets[c["name"]] = dict(
            latent=r["latent"], sib_frames=sorted(all_sib_frames), sr=r["sr"],
            n_latent_frames=n_latent_frames, chunk_size=chunk_size,
        )

    # ---- Task 2: direct manipulation on the two FLAGGED clips --------------
    task2_results = {}
    for name in ("F3_library_flagged", "F5_seashells_flagged"):
        tgt = manipulation_targets[name]
        latent, sib_frames, sr = tgt["latent"], tgt["sib_frames"], tgt["sr"]
        # rebuild sample-level sibilant mask from the recorded runs for this clip
        rec = next(r for r in task1_results if r["name"] == name)
        n_samples = None
        sweep = []
        base_wav = None
        for factor in ATTEN_FACTORS:
            wav = attenuate_and_decode(tts, latent, sib_frames, factor)
            dur_s = rec["dur_s"]
            trimmed = wav[: int(sr * dur_s)].astype(np.float32)
            if factor == 1.0:
                base_wav = trimmed.copy()
            n_samples = len(trimmed)
            sib_mask = np.zeros(n_samples, dtype=bool)
            for run in rec["runs"]:
                lo, hi = run["sample_lo"], min(run["sample_hi"], n_samples)
                if hi > lo:
                    sib_mask[lo:hi] = True
            non_sib_mask = ~sib_mask

            out_path = os.path.join(AUDIO_DIR, f"{name}_atten{factor}.wav")
            sf.write(out_path, trimmed, sr, subtype="PCM_16")

            words = transcribe_words(whisper_model, out_path)
            transcript = " ".join(w["word"] for w in words)

            diff_outside = None
            if factor != 1.0 and base_wav is not None and len(base_wav) == n_samples:
                diff_outside = float(np.abs(trimmed[non_sib_mask] - base_wav[non_sib_mask]).max()) if non_sib_mask.any() else 0.0

            sweep.append(dict(
                factor=factor, wav_path=out_path,
                sibilant_peak=peak_in_mask(trimmed, sib_mask),
                nonsibilant_peak=peak_in_mask(trimmed, non_sib_mask),
                whole_peak=float(np.abs(trimmed).max()),
                max_abs_diff_outside_sibilant_region_vs_factor1=diff_outside,
                transcript=transcript,
                n_words=len(words),
            ))
            print(f"  {name} factor={factor}: sib_peak={sweep[-1]['sibilant_peak']:.4f} "
                  f"nonsib_peak={sweep[-1]['nonsibilant_peak']:.4f} "
                  f"diff_outside={diff_outside} words={len(words)}")

        task2_results[name] = dict(
            n_sibilant_latent_frames_attenuated=len(sib_frames),
            sibilant_latent_frames=sib_frames,
            n_latent_frames_total=tgt["n_latent_frames"],
            sweep=sweep,
            reference_transcript=" ".join(w for w in rec.get("_words", []) or []),
        )

    out = dict(
        experiment="phase2a_latent_frame_span",
        chunk_size_samples=chunk_size, sample_rate=tts.sample_rate,
        ms_per_latent_frame=chunk_size / tts.sample_rate * 1000,
        atten_factors=ATTEN_FACTORS,
        task1_frame_span=task1_results,
        task2_manipulation=task2_results,
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
