"""Phase 2a -- is the flagged-word hiccup a time-compression artifact of
`speed`, or something else?

Two listener reports on bench 6 (K4_typical control, seed 20261069,
woodchuck; K16_worst control, seed 20261449, seashells -- both unperturbed
M1, see results/listening_sets/phase2a_capacity/manifest.json) describe a
COMPRESSION symptom, not a noise symptom: "'morning' is pronounced too
quickly so it sounds like a hiccup", "the 'chuck' after woodchuck sounds
condensed/hiccuped", "a spike at the end of 'jump'", "the final 'wood' is
distorted". Two of the three flagged words are utterance-final.

`helper.py`'s default `speed=1.05` divides the duration predictor's own
estimate by 1.05 (`dur_onnx = dur_onnx / speed`, TextToSpeech._infer), which
shrinks the ENTIRE time budget (`sample_noisy_latent` derives
`wav_lengths`/`latent_len` from that shortened duration, and every render
script in this repo then trims the raw vocoder output to
`int(sr * dur[0])`, matching that same shortened duration -- see
`phase2a_seed_variance.py`'s `render`, `benches/phase2a_capacity_bench.py`'s
`render`, etc.). Two effects follow from one cause: (1) the acoustic model
must fit the whole utterance into ~5% fewer latent frames than its own
predictor asked for, which should hurt most where there is least slack
(dense stressed syllables); (2) the vocoder output is then cut short at that
same reduced duration, which should hurt utterance-final content
specifically.

This script:
  1. Instruments one render per flagged sentence at its flagged seed,
     default settings, printing every stage of the time budget: raw
     duration-predictor output, duration after /speed, wav_lengths,
     latent_len, latent_mask valid length, raw vocoder samples, trimmed
     samples, and whether the discarded tail carries signal above the noise
     floor.
  2. Experiment A -- speed sweep {1.05, 1.00, 0.95, 0.90} x 2 sentences x 3
     seeds (flagged + 2 extra), TOTAL_STEP held at the repo default (8).
     Saves both the full (untrimmed) and trimmed vocoder output for every
     condition, which doubles as Experiment B (trim test) at speed=1.05
     with no extra renders.
  3. Experiment C -- TOTAL_STEP sweep {8, 16, 32} at speed=1.05 (the 8-step
     condition is already covered by Experiment A), same 2 sentences x 3
     seeds.
  4. Per-word duration extraction (faster-whisper tiny.en, word_timestamps)
     on every Experiment-A trimmed clip, tracking the flagged words
     ('chuck' -- the FIRST occurrence, right after "woodchuck"; 'wood' --
     the utterance-final occurrence; 'morning' -- utterance-final) across
     the speed sweep.

Usage (from py/):
    python3 phase2a_speed_rootcause.py
"""

import json
import os
import time

import numpy as np
import soundfile as sf

from helper import Style, get_latent_mask, load_text_to_speech, load_voice_style
from phase2b_generate import LANG, ONNX_DIR, SEED_BASE, TEXTS, VOICE_STYLE_DIR

OUT_DIR = "results/phase2a"
LISTEN_DIR = "results/listening_sets/phase2a_speed_rootcause"
REPORT_PATH = os.path.join(OUT_DIR, "speed_rootcause.json")

BASE_PRESET = "M1"
VOICE_STYLE_PATH = os.path.join(VOICE_STYLE_DIR, f"{BASE_PRESET}.json")

WOODCHUCK_TEXT_IDX = 3
SEASHELLS_TEXT_IDX = 1
assert TEXTS[WOODCHUCK_TEXT_IDX].startswith("How much wood")
assert TEXTS[SEASHELLS_TEXT_IDX].startswith("She sells seashells")

DEFAULT_SPEED = 1.05
DEFAULT_TOTAL_STEP = 8
SPEEDS = [1.05, 1.00, 0.95, 0.90]
STEPS_LIST = [8, 16, 32]

SENTENCES = {
    "woodchuck": dict(
        text_idx=WOODCHUCK_TEXT_IDX,
        text=TEXTS[WOODCHUCK_TEXT_IDX],
        flagged_seed=20261069,
        extra_seeds=[SEED_BASE + 30001, SEED_BASE + 30002],
        flagged_word="chuck",
        flagged_word_occurrence=0,  # first "chuck" -- right after "woodchuck"
        final_word="wood",
        note="'chuck' after woodchuck sounds condensed/hiccuped; final 'wood' distorted",
    ),
    "seashells": dict(
        text_idx=SEASHELLS_TEXT_IDX,
        text=TEXTS[SEASHELLS_TEXT_IDX],
        flagged_seed=20261449,
        extra_seeds=[SEED_BASE + 30003, SEED_BASE + 30004],
        flagged_word="morning",
        flagged_word_occurrence=0,  # only occurrence, also utterance-final
        final_word="morning",
        note="final word 'morning' pronounced too quickly, sounds like a hiccup",
    ),
}


def timer(name):
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *a):
            print(f"[{name}] {time.time() - self.t0:.1f}s")

    return _T()


def infer_instrumented(tts, text, lang, style, total_step, speed, seed, verbose=False):
    """Reimplements TextToSpeech._infer with every intermediate stage
    exposed, so we can report the time-budget pipeline without touching
    helper.py. Mirrors helper.py's TextToSpeech._infer/sample_noisy_latent
    line for line."""
    np.random.seed(seed)
    text_ids, text_mask = tts.text_processor([text], [lang])
    dur_raw, *_ = tts.dp_ort.run(
        None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    )
    dur_onnx = dur_raw / speed
    text_emb_onnx, *_ = tts.text_enc_ort.run(
        None,
        {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
    )

    bsz = len(dur_onnx)
    wav_len_max = dur_onnx.max() * tts.sample_rate
    wav_lengths = (dur_onnx * tts.sample_rate).astype(np.int64)
    chunk_size = tts.base_chunk_size * tts.chunk_compress_factor
    latent_len = int(((wav_len_max + chunk_size - 1) / chunk_size))
    latent_dim = tts.ldim * tts.chunk_compress_factor
    xt = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
    latent_mask = get_latent_mask(wav_lengths, tts.base_chunk_size, tts.chunk_compress_factor)
    xt = xt * latent_mask

    total_step_np = np.array([total_step] * bsz, dtype=np.float32)
    for step in range(total_step):
        current_step = np.array([step] * bsz, dtype=np.float32)
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
    wav, *_ = tts.vocoder_ort.run(None, {"latent": xt})

    trimmed_len = int(tts.sample_rate * dur_onnx[0].item())
    raw_len = wav.shape[1]
    info = dict(
        raw_dur_sec=float(dur_raw[0]),
        dur_after_speed_sec=float(dur_onnx[0]),
        wav_lengths_samples=int(wav_lengths[0]),
        latent_len=int(latent_len),
        latent_mask_valid_frames=int(latent_mask[0].sum()),
        raw_vocoder_samples=int(raw_len),
        trimmed_samples=int(trimmed_len),
        discarded_samples=int(raw_len - trimmed_len),
        discarded_ms=float((raw_len - trimmed_len) / tts.sample_rate * 1000),
    )
    return wav, dur_onnx, info


def dbfs(x):
    r = float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0
    return 20 * np.log10(max(r, 1e-12))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)

    print("Loading TTS engine ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    sr = tts.sample_rate
    style = load_voice_style([VOICE_STYLE_PATH])

    report = {"instrumentation": {}, "speed_sweep": [], "trim_test": [], "steps_sweep": []}

    # ------------------------------------------------------------------
    # 1. Instrumentation: one render per sentence, flagged seed, defaults.
    # ------------------------------------------------------------------
    print("\n=== INSTRUMENTATION (flagged seed, speed=1.05, TOTAL_STEP=8) ===")
    for key, spec in SENTENCES.items():
        wav, dur, info = infer_instrumented(
            tts, spec["text"], LANG, style, DEFAULT_TOTAL_STEP, DEFAULT_SPEED, spec["flagged_seed"]
        )
        trimmed = wav[0, : info["trimmed_samples"]]
        discarded = wav[0, info["trimmed_samples"]:]
        info["discarded_tail_dbfs"] = dbfs(discarded)
        info["trimmed_clip_dbfs"] = dbfs(trimmed)
        info["discarded_tail_peak_dbfs"] = dbfs(np.array([np.abs(discarded).max()])) if discarded.size else -120.0
        info["discarded_above_noise_floor"] = bool(
            discarded.size and info["discarded_tail_dbfs"] > info["trimmed_clip_dbfs"] - 20
        )
        report["instrumentation"][key] = info
        print(f"\n-- {key} (seed {spec['flagged_seed']}) --")
        for k, v in info.items():
            print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 2/3. Experiment A (speed sweep) + B (trim test, piggybacked on A).
    # ------------------------------------------------------------------
    print("\n=== EXPERIMENT A: speed sweep + B: trim test ===")
    a_records = []
    with timer("speed sweep"):
        for key, spec in SENTENCES.items():
            seeds = [spec["flagged_seed"]] + spec["extra_seeds"]
            for seed in seeds:
                for speed in SPEEDS:
                    wav, dur, info = infer_instrumented(
                        tts, spec["text"], LANG, style, DEFAULT_TOTAL_STEP, speed, seed
                    )
                    trimmed = wav[0, : info["trimmed_samples"]].astype(np.float32)
                    full = wav[0].astype(np.float32)
                    stem = f"{key}_seed{seed}_speed{speed:.2f}"
                    sf.write(os.path.join(LISTEN_DIR, f"{stem}_trimmed.wav"), trimmed, sr, subtype="PCM_16")
                    sf.write(os.path.join(LISTEN_DIR, f"{stem}_full.wav"), full, sr, subtype="PCM_16")
                    rec = dict(
                        sentence=key, seed=seed, speed=speed, total_step=DEFAULT_TOTAL_STEP,
                        is_flagged_seed=(seed == spec["flagged_seed"]),
                        trimmed_file=f"{stem}_trimmed.wav", full_file=f"{stem}_full.wav",
                        **info,
                    )
                    a_records.append(rec)
    report["speed_sweep"] = a_records

    # ------------------------------------------------------------------
    # 4. Experiment C: TOTAL_STEP sweep at speed=1.05 (steps=8 already in A).
    # ------------------------------------------------------------------
    print("\n=== EXPERIMENT C: TOTAL_STEP sweep ===")
    c_records = [
        {**r, "total_step": DEFAULT_TOTAL_STEP} for r in a_records if r["speed"] == DEFAULT_SPEED
    ]
    with timer("steps sweep"):
        for key, spec in SENTENCES.items():
            seeds = [spec["flagged_seed"]] + spec["extra_seeds"]
            for seed in seeds:
                for total_step in STEPS_LIST:
                    if total_step == DEFAULT_TOTAL_STEP:
                        continue  # already rendered in A
                    wav, dur, info = infer_instrumented(
                        tts, spec["text"], LANG, style, total_step, DEFAULT_SPEED, seed
                    )
                    trimmed = wav[0, : info["trimmed_samples"]].astype(np.float32)
                    stem = f"{key}_seed{seed}_steps{total_step}"
                    sf.write(os.path.join(LISTEN_DIR, f"{stem}_trimmed.wav"), trimmed, sr, subtype="PCM_16")
                    rec = dict(
                        sentence=key, seed=seed, speed=DEFAULT_SPEED, total_step=total_step,
                        is_flagged_seed=(seed == spec["flagged_seed"]),
                        trimmed_file=f"{stem}_trimmed.wav",
                        **info,
                    )
                    c_records.append(rec)
    report["steps_sweep"] = c_records

    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote WAVs to {LISTEN_DIR}")


if __name__ == "__main__":
    main()
