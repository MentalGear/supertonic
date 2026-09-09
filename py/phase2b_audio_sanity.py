"""Phase 2b, stage 2b: is the perturbed audio still voice-like?

The probe result at a given perturbation magnitude only means something if the
audio at that magnitude is still a plausible voice. This is the on-manifold /
off-manifold half of the design: perturbations big enough to be genuinely
high-dimensional may stop rendering as speech, and perturbations small enough to
stay realistic may be too low-dimensional for the probe result to be
interesting. This script measures both ends of that trade-off rather than
picking one silently.

Per condition, on a random subsample of clips:

  * voiced fraction, median F0 and F0 spread (librosa pyin) -- speech that has
    stopped being voice-like loses voicing first;
  * mean spectral flatness -- noise-likeness;
  * 2-8 Hz envelope modulation ratio -- the syllabic rhythm band; speech has
    energy there, texture does not;
  * word error rate against the known prompt text (faster-whisper tiny.en) --
    a direct intelligibility number, if faster-whisper is installed.

A reference band is computed by rendering the ten unperturbed presets over the
same eight sentences, so every number can be read against "what this engine
does normally". Three listening exemplars per condition are re-rendered at the
full 44.1 kHz synthesis rate.

Usage (from py/):
    python3 phase2b_audio_sanity.py [--per-condition 64]
"""

import argparse
import json
import os
import re

import numpy as np
import soundfile as sf

from helper import Style, load_text_to_speech, load_voice_style

OUT_DIR = "results/phase2b"
AUDIO_DIR = os.path.join(OUT_DIR, "audio16k")
LISTEN_DIR = "results/listening_sets/phase2b_perturbation"
ONNX_DIR = "assets/onnx"
VOICE_STYLE_DIR = "assets/voice_styles"
SR = 16000
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]


def acoustic_metrics(wav):
    import librosa
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=400, sr=SR,
                                 frame_length=1024, hop_length=256)
    vf = float(np.nanmean(voiced.astype(float)))
    f0v = f0[np.isfinite(f0)]
    flat = float(librosa.feature.spectral_flatness(y=wav, n_fft=1024, hop_length=256).mean())
    env = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)).sum(0)
    env = env - env.mean()
    fr = np.fft.rfftfreq(len(env), d=256 / SR)
    p = np.abs(np.fft.rfft(env)) ** 2
    band = (fr >= 2) & (fr <= 8)
    tot = fr > 0
    return {
        "voiced_fraction": vf,
        "median_f0": float(np.median(f0v)) if f0v.size else float("nan"),
        "f0_iqr": float(np.subtract(*np.percentile(f0v, [75, 25]))) if f0v.size else float("nan"),
        "spectral_flatness": flat,
        "mod_2_8hz_ratio": float(p[band].sum() / p[tot].sum()) if p[tot].sum() > 0 else float("nan"),
    }


def norm_text(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def wer(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / max(1, len(r))


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90))} if v.size else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-condition", type=int, default=64)
    ap.add_argument("--exemplars", type=int, default=3)
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    recs = meta["records"]
    texts = meta["texts"]
    os.makedirs(LISTEN_DIR, exist_ok=True)

    asr = None
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        print("faster-whisper tiny.en loaded; WER will be reported")
    except Exception as e:
        print(f"faster-whisper unavailable ({e}); WER omitted")

    def transcribe(path):
        segs, _ = asr.transcribe(path, language="en", beam_size=1)
        return " ".join(s.text for s in segs)

    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    styles = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]) for p in PRESETS}
    dp_ref = styles["M1"].dp

    # ---- reference band: the unperturbed presets on the same eight sentences
    ref_rows = []
    os.makedirs(os.path.join(OUT_DIR, "reference16k"), exist_ok=True)
    from scipy.signal import resample_poly
    for p in PRESETS:
        for ti, t in enumerate(texts):
            np.random.seed(hash((p, ti)) % (2 ** 31))
            wav, dur = tts(t, meta["lang"], Style(styles[p].ttl, dp_ref.copy()),
                           meta["total_step"], meta["speed"])
            w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
            w16 = resample_poly(w, SR, tts.sample_rate).astype(np.float32)
            fp = os.path.join(OUT_DIR, "reference16k", f"{p}_t{ti}.wav")
            sf.write(fp, w16, SR, subtype="PCM_16")
            # read back from the written PCM_16 file, exactly as the condition
            # clips are read, so the two sides of the comparison share the same
            # quantization noise floor (measuring the float array here instead
            # made reference spectral flatness look 3x lower than it is)
            w16, _ = sf.read(fp, dtype="float32")
            row = acoustic_metrics(w16)
            row["preset"], row["text_idx"] = p, ti
            if asr:
                row["wer"] = wer(t, transcribe(fp))
            ref_rows.append(row)
        print(f"  reference {p} done", flush=True)

    # ---- per-condition subsample
    rng = np.random.default_rng(5)
    by_cond, per_clip = {}, []
    for c in dict.fromkeys(r["condition"] for r in recs):
        pool = [r for r in recs if r["condition"] == c]
        pick = rng.choice(len(pool), size=min(args.per_condition, len(pool)), replace=False)
        rows = []
        for i in pick:
            r = pool[int(i)]
            fp = os.path.join(AUDIO_DIR, r["file"])
            w, _ = sf.read(fp, dtype="float32")
            row = acoustic_metrics(w)
            row.update({"peak": r["peak"], "rms": r["rms"]})
            if asr:
                row["wer"] = wer(texts[r["text_idx"]], transcribe(fp))
            row["idx"], row["condition"], row["base"] = r["idx"], c, r["base"]
            rows.append(row)
            per_clip.append(row)
        by_cond[c] = summarize(rows)
        print(f"  condition {c}: " + ", ".join(
            f"{k}={v['mean']:.3f}" for k, v in by_cond[c].items() if v), flush=True)

    # ---- listening exemplars, re-rendered at the full synthesis rate
    S = np.load(os.path.join(OUT_DIR, "styles.npz"))
    ttl_all = S["ttl"]
    exemplars = []
    for c in dict.fromkeys(r["condition"] for r in recs):
        pool = [r for r in recs if r["condition"] == c and r["split"] == "test"]
        for r in pool[: args.exemplars]:
            np.random.seed(r["seed"])
            st = Style(ttl_all[r["idx"]][None].astype(np.float32), dp_ref.copy())
            wav, dur = tts(texts[r["text_idx"]], meta["lang"], st,
                           meta["total_step"], meta["speed"])
            w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
            fn = f"{c}_{r['base'].replace(':', '-')}_idx{r['idx']:05d}.wav"
            sf.write(os.path.join(LISTEN_DIR, fn), w, tts.sample_rate)
            exemplars.append({"file": fn, "condition": c, "eps": r["eps"], "base": r["base"],
                              "text": texts[r["text_idx"]], "idx": r["idx"], "seed": r["seed"]})
    # unperturbed references for comparison
    for p in ("M1", "F1"):
        np.random.seed(0)
        wav, dur = tts(texts[0], meta["lang"], Style(styles[p].ttl, dp_ref.copy()),
                       meta["total_step"], meta["speed"])
        w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
        sf.write(os.path.join(LISTEN_DIR, f"reference_{p}.wav"), w, tts.sample_rate)
        exemplars.append({"file": f"reference_{p}.wav", "condition": "unperturbed_preset",
                          "eps": 0.0, "base": p, "text": texts[0], "idx": None, "seed": 0})

    out = {
        "experiment": "phase2b_audio_sanity",
        "reference_band_unperturbed_presets": summarize(ref_rows),
        "by_condition": by_cond,
        "asr": "faster-whisper tiny.en, greedy" if asr else None,
        "listening_exemplars_dir": LISTEN_DIR,
        "listening_exemplars": exemplars,
        "per_clip": per_clip,
    }
    with open(os.path.join(OUT_DIR, "audio_sanity.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT_DIR}/audio_sanity.json and {len(exemplars)} exemplars in {LISTEN_DIR}")


if __name__ == "__main__":
    main()
