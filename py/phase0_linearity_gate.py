"""Phase 0 linearity gate (see ../new-plan.md).

Interpolates between two preset voice styles' `style_ttl` tensors at a sweep
of weights and synthesizes identical text at each point, to check whether the
style manifold is locally linear enough between speakers for the downstream
parametric-voice-space plan to be worth pursuing.

Reuses the existing inference path from helper.py (load_text_to_speech,
load_voice_style) rather than duplicating it -- this is not a new synthesis
pipeline, just a sweep driver over Style.with_deltas().

Usage (from py/):
    python3 phase0_linearity_gate.py
"""

import datetime
import json
import os

import numpy as np
import soundfile as sf

from helper import Style, load_text_to_speech, load_voice_style, timer

ONNX_DIR = "assets/onnx"
VOICE_STYLE_DIR = "assets/voice_styles"
VOICE_A_NAME = "M1"
VOICE_B_NAME = "F1"
TEXT = "The quick brown fox jumps over the lazy dog."
LANG = "en"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
TOTAL_STEP = 8
SPEED = 1.05
INCLUDE_DURATION = False
SAVE_DIR = "results/listening_sets/phase0_linearity"


def weight_tag(w: float) -> str:
    # 0.0 -> "0.00", 0.25 -> "0.25", 1.0 -> "1.00"
    return f"{w:.2f}"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    voice_a_path = os.path.join(VOICE_STYLE_DIR, f"{VOICE_A_NAME}.json")
    voice_b_path = os.path.join(VOICE_STYLE_DIR, f"{VOICE_B_NAME}.json")

    style_a = load_voice_style([voice_a_path], verbose=True)
    style_b = load_voice_style([voice_b_path], verbose=True)

    # Sanity: confirm the row-unit-norm assumption new-plan.md Phase 1a flags,
    # since it's what makes w=1.0 recover B exactly under with_deltas().
    norms_a = np.linalg.norm(style_a.ttl, axis=-1)
    norms_b = np.linalg.norm(style_b.ttl, axis=-1)
    print(f"{VOICE_A_NAME} TTL row-norm range: [{norms_a.min():.4f}, {norms_a.max():.4f}]")
    print(f"{VOICE_B_NAME} TTL row-norm range: [{norms_b.min():.4f}, {norms_b.max():.4f}]")

    # delta = B - A, expressed as a Style so with_deltas() can consume it.
    # dp difference is computed for completeness but not used since
    # include_duration=False holds dp fixed at A's.
    delta = Style(style_b.ttl - style_a.ttl, style_b.dp - style_a.dp)

    text_to_speech = load_text_to_speech(ONNX_DIR, use_gpu=False)

    manifest = {
        "experiment": "phase0_linearity_gate",
        "date": datetime.date.today().isoformat(),
        "voice_a": VOICE_A_NAME,
        "voice_b": VOICE_B_NAME,
        "text": TEXT,
        "lang": LANG,
        "weights": WEIGHTS,
        "total_step": TOTAL_STEP,
        "speed": SPEED,
        "include_duration": INCLUDE_DURATION,
        "dp_source": f"held fixed at {VOICE_A_NAME}'s style_dp (include_duration=False)",
        "row_norms": {
            VOICE_A_NAME: {"min": float(norms_a.min()), "max": float(norms_a.max())},
            VOICE_B_NAME: {"min": float(norms_b.min()), "max": float(norms_b.max())},
        },
        "outputs": [],
    }

    for w in WEIGHTS:
        interpolated = style_a.with_deltas([(delta, w)], include_duration=INCLUDE_DURATION)

        fname = f"interp_w{weight_tag(w)}.wav"
        fpath = os.path.join(SAVE_DIR, fname)

        with timer(f"Synthesizing w={w}"):
            wav, duration = text_to_speech(TEXT, LANG, interpolated, TOTAL_STEP, SPEED)

        w_trim = wav[0, : int(text_to_speech.sample_rate * duration[0].item())]
        sf.write(fpath, w_trim, text_to_speech.sample_rate)
        print(f"Saved: {fpath}")

        manifest["outputs"].append(
            {
                "weight": w,
                "file": fname,
                "duration_sec": float(duration[0].item()),
            }
        )

    # Numeric endpoint check: with_deltas at w=0.0 / w=1.0 should reproduce
    # A / B's own tensors exactly (up to float32 associativity).
    style_at_0 = style_a.with_deltas([(delta, 0.0)], include_duration=INCLUDE_DURATION)
    style_at_1 = style_a.with_deltas([(delta, 1.0)], include_duration=INCLUDE_DURATION)

    max_diff_0_vs_a = float(np.abs(style_at_0.ttl - style_a.ttl).max())
    max_diff_1_vs_b = float(np.abs(style_at_1.ttl - style_b.ttl).max())
    print(f"max|w=0 interpolated TTL - {VOICE_A_NAME} TTL| = {max_diff_0_vs_a:.3e}")
    print(f"max|w=1 interpolated TTL - {VOICE_B_NAME} TTL| = {max_diff_1_vs_b:.3e}")

    manifest["endpoint_check"] = {
        "max_abs_diff_w0_vs_A_ttl": max_diff_0_vs_a,
        "max_abs_diff_w1_vs_B_ttl": max_diff_1_vs_b,
    }

    manifest_path = os.path.join(SAVE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
