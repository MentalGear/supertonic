"""Phase 2b: a listenable perturbation ray.

The sanity exemplars vary text and base voice, which makes them hard to compare
by ear. This renders a single controlled ray instead: one base preset, one
sentence, one fixed random direction per active row, scaled through a sweep of
magnitudes. Clip k differs from clip k-1 only by how far along that one
direction the style sits, so the listener hears the perturbation magnitude and
nothing else. eps beyond the sampled range is included so the failure point can
be located rather than assumed.

Per-row angular displacement is about atan(eps): eps=0.2 ~ 11 deg, eps=0.8 ~ 39
deg, eps=1.6 ~ 58 deg. For scale, per-row angles between presets M1 and F1 run
18 deg mean / 37 deg max.

Usage (from py/):  python3 phase2b_perturbation_sweep.py
"""

import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from helper import Style, load_text_to_speech, load_voice_style
from phase2b_audio_sanity import acoustic_metrics, wer
from phase2b_generate import ACTIVE_ROWS, TEXTS, unit_rows

SAVE_DIR = "results/listening_sets/phase2b_perturbation_ray"
BASES = ["M1", "F1"]
TEXT = TEXTS[0]
EPS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]
SEED = 4242


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp

    asr = None
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"faster-whisper unavailable ({e})")

    rng = np.random.default_rng(SEED)
    direction = unit_rows(rng.standard_normal((len(ACTIVE_ROWS), 256)).astype(np.float32))

    out = []
    for b in BASES:
        base = load_voice_style([f"assets/voice_styles/{b}.json"]).ttl.astype(np.float32)
        for eps in EPS:
            ttl = base.copy()
            if eps > 0:
                ttl[0, ACTIVE_ROWS, :] = unit_rows(ttl[0, ACTIVE_ROWS, :] + eps * direction)
            dev = float(np.abs(np.linalg.norm(ttl, axis=-1) - 1.0).max())
            cos = float((ttl[0, ACTIVE_ROWS] * base[0, ACTIVE_ROWS]).sum(-1).mean())
            np.random.seed(SEED)
            wav, dur = tts(TEXT, "en", Style(ttl, dp_ref.copy()), 8, 1.05)
            w = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
            fn = f"{b}_eps{eps:.2f}.wav"
            sf.write(os.path.join(SAVE_DIR, fn), w, tts.sample_rate)
            w16 = resample_poly(w, 16000, tts.sample_rate).astype(np.float32)
            tmp = os.path.join(SAVE_DIR, "_tmp16k.wav")
            sf.write(tmp, w16, 16000, subtype="PCM_16")
            m = acoustic_metrics(sf.read(tmp, dtype="float32")[0])
            if asr:
                segs, _ = asr.transcribe(tmp, language="en", beam_size=1)
                hyp = " ".join(s.text for s in segs)
                m["wer"] = wer(TEXT, hyp)
                m["transcript"] = hyp.strip()
            m.update({"file": fn, "base": b, "eps": eps,
                      "mean_active_row_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
                      "row_norm_max_dev": dev, "peak": float(np.abs(w).max()),
                      "duration_sec": float(dur[0].item())})
            out.append(m)
            print(f"{fn}: angle {m['mean_active_row_angle_deg']:5.1f} deg  "
                  f"voiced {m['voiced_fraction']:.2f}  F0 {m['median_f0']:6.1f}  "
                  f"flat {m['spectral_flatness']:.3f}  mod2-8 {m['mod_2_8hz_ratio']:.2f}  "
                  f"peak {m['peak']:.2f}  WER {m.get('wer', float('nan')):.2f}", flush=True)
    os.remove(os.path.join(SAVE_DIR, "_tmp16k.wav"))

    with open(os.path.join(SAVE_DIR, "manifest.json"), "w") as f:
        json.dump({"experiment": "phase2b_perturbation_ray",
                   "text": TEXT, "lang": "en", "total_step": 8, "speed": 1.05,
                   "style_dp": "held fixed at M1's", "rng_seed": SEED,
                   "direction": "one fixed unit vector per active row, shared by every clip",
                   "active_rows": ACTIVE_ROWS, "outputs": out}, f, indent=2)
    print(f"\nWrote {SAVE_DIR}")


if __name__ == "__main__":
    main()
