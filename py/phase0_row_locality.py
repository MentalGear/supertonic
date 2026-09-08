"""Phase 0 companion probe: is `style_ttl`'s 50x256 grid row-structured?

See ../new-plan.md, "Companion experiment: is the 50x256 grid structured?".
The doc's procedure is followed literally: take voices A and B and synthesize
50 hybrids where hybrid `i` uses A's TTL with row `i` replaced by B's, then
measure and listen. Two things are added on top of it, both cheap:

  * a purely numeric per-row profile (no synthesis needed) of how far A and B
    -- and all ten released presets -- actually differ row by row, and
  * contiguous band swaps plus data-driven "active rows only" / "inactive rows
    only" swaps, which are what is actually listenable: 50 single-row clips is
    not a listening set, it is a measurement.

Everything reuses the inference path from helper.py and mirrors the settings of
phase0_linearity_gate.py (same text, base voice, total_step, speed, lang), so
this set is directly comparable to the linearity gate's.

`style_dp` is held fixed at A's throughout (include_duration=False), so every
clip has the identical predicted duration and identical latent length.

The vocoder's initial latent is sampled with an unseeded np.random.randn, which
makes two renders of the *same* tensor differ as waveforms. numpy is therefore
re-seeded to SEED immediately before every synthesis call, so any audio
difference between clips is attributable to the style tensor alone. That is
what makes the spectral distances below mean anything.

File naming (all under results/listening_sets/phase0_row_locality/):
    endpoint_M1.wav                 -- unmodified voice A
    endpoint_F1.wav                 -- unmodified voice B
    single_row/row07_from_F1.wav    -- A's TTL, row 7 replaced by B's row 7
    band_rows00-24_from_F1.wav      -- A's TTL, rows 0..24 replaced by B's
    set_active_from_F1.wav          -- A's TTL, the high-variance rows from B
    set_inactive_from_F1.wav        -- A's TTL, the low-variance rows from B

Usage (from py/):
    python3 phase0_row_locality.py
"""

import datetime
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import stft

from helper import Style, load_text_to_speech, load_voice_style, timer

ONNX_DIR = "assets/onnx"
VOICE_STYLE_DIR = "assets/voice_styles"
VOICE_A_NAME = "M1"
VOICE_B_NAME = "F1"
# All ten released presets, used only for the numeric per-row activity profile.
ALL_PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# Identical to phase0_linearity_gate.py so the two sets are comparable.
TEXT = "The quick brown fox jumps over the lazy dog."
LANG = "en"
TOTAL_STEP = 8
SPEED = 1.05
INCLUDE_DURATION = False

SEED = 0
N_ROWS = 50
# Contiguous bands: the two halves, then fifths.
BANDS = [(0, 25), (25, 50), (0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
# A row counts as "active" if the ten presets spread further than this from
# their own per-row centroid. The gap in the sorted profile is wide enough
# that the exact cut does not matter (see the printed profile).
ACTIVE_SPREAD_THRESHOLD = 0.10

SAVE_DIR = "results/listening_sets/phase0_row_locality"
SINGLE_ROW_SUBDIR = "single_row"


def swap_rows(ttl_a: np.ndarray, ttl_b: np.ndarray, rows) -> np.ndarray:
    """A's TTL with the given row indices taken from B's."""
    out = ttl_a.copy()
    out[:, list(rows), :] = ttl_b[:, list(rows), :]
    return out


def row_norm_report(ttl: np.ndarray) -> dict:
    norms = np.linalg.norm(ttl, axis=-1)
    return {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "max_abs_dev_from_unit": float(np.abs(norms - 1.0).max()),
    }


def log_spectrum(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log magnitude STFT, used as a cheap perceptual-ish distance space."""
    _, _, z = stft(wav, fs=sample_rate, nperseg=1024, noverlap=768)
    return np.log(np.abs(z) + 1e-6)


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[-1], b.shape[-1])
    return float(np.abs(a[..., :n] - b[..., :n]).mean())


def main():
    single_row_dir = os.path.join(SAVE_DIR, SINGLE_ROW_SUBDIR)
    os.makedirs(single_row_dir, exist_ok=True)

    style_a = load_voice_style(
        [os.path.join(VOICE_STYLE_DIR, f"{VOICE_A_NAME}.json")], verbose=True
    )
    style_b = load_voice_style(
        [os.path.join(VOICE_STYLE_DIR, f"{VOICE_B_NAME}.json")], verbose=True
    )
    ttl_a, ttl_b = style_a.ttl, style_b.ttl

    # ---------------------------------------------------------------- numeric
    # Per-row A-vs-B profile.
    a0, b0 = ttl_a[0], ttl_b[0]
    norms_a = np.linalg.norm(a0, axis=-1)
    norms_b = np.linalg.norm(b0, axis=-1)
    cos_ab = (a0 * b0).sum(-1) / (norms_a * norms_b)
    delta_norm = np.linalg.norm(b0 - a0, axis=-1)
    delta_share = delta_norm**2 / (delta_norm**2).sum()

    # Per-row activity across all ten presets: mean distance from the per-row
    # centroid. A row that never moves between released voices cannot be
    # carrying speaker identity.
    all_ttl = np.stack(
        [
            load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{n}.json")]).ttl[0]
            for n in ALL_PRESETS
        ]
    )
    spread = np.linalg.norm(all_ttl - all_ttl.mean(0), axis=-1).mean(0)
    active_rows = [int(i) for i in range(N_ROWS) if spread[i] >= ACTIVE_SPREAD_THRESHOLD]
    inactive_rows = [i for i in range(N_ROWS) if i not in active_rows]

    print(f"{VOICE_A_NAME} TTL row-norm range: [{norms_a.min():.4f}, {norms_a.max():.4f}]")
    print(f"{VOICE_B_NAME} TTL row-norm range: [{norms_b.min():.4f}, {norms_b.max():.4f}]")
    print(
        f"per-row cos({VOICE_A_NAME},{VOICE_B_NAME}): "
        f"min {cos_ab.min():.4f} mean {cos_ab.mean():.4f} max {cos_ab.max():.4f}"
    )
    top = np.argsort(-delta_share)[:10]
    print(
        "top-10 rows by share of total squared delta: "
        + ", ".join(f"{int(i)} ({delta_share[i]*100:.1f}%)" for i in top)
        + f"  [sum {delta_share[top].sum()*100:.1f}%, uniform would be 20%]"
    )
    print(f"active rows (spread >= {ACTIVE_SPREAD_THRESHOLD}): {len(active_rows)} of {N_ROWS} -> {active_rows}")

    text_to_speech = load_text_to_speech(ONNX_DIR, use_gpu=False)

    manifest = {
        "experiment": "phase0_row_locality",
        "date": datetime.date.today().isoformat(),
        "question": "is style_ttl diffuse across all 50 rows, or localized to specific rows?",
        "voice_a": VOICE_A_NAME,
        "voice_b": VOICE_B_NAME,
        "hybrid_rule": "voice_a's style_ttl with the listed rows replaced verbatim by voice_b's",
        "text": TEXT,
        "lang": LANG,
        "total_step": TOTAL_STEP,
        "speed": SPEED,
        "include_duration": INCLUDE_DURATION,
        "dp_source": f"held fixed at {VOICE_A_NAME}'s style_dp (include_duration=False)",
        "rng_seed": SEED,
        "rng_note": (
            "np.random.seed(SEED) is called immediately before every synthesis, so the "
            "vocoder's initial noisy latent is identical for every clip and all audio "
            "differences are caused by the style tensor"
        ),
        "distance_metric": (
            "mean absolute difference of log |STFT| (nperseg=1024, noverlap=768); "
            "travel = d_to_A / (d_to_A + d_to_B), 0.0 = sounds like A, 1.0 = sounds like B"
        ),
        "row_norms": {
            VOICE_A_NAME: row_norm_report(ttl_a),
            VOICE_B_NAME: row_norm_report(ttl_b),
        },
        "per_row_profile": {
            "cos_a_vs_b": [round(float(c), 4) for c in cos_ab],
            "delta_norm": [round(float(d), 4) for d in delta_norm],
            "delta_share_pct": [round(float(s) * 100, 3) for s in delta_share],
            "spread_across_10_presets": [round(float(s), 4) for s in spread],
            "active_rows": active_rows,
            "inactive_rows": inactive_rows,
            "active_spread_threshold": ACTIVE_SPREAD_THRESHOLD,
        },
        "outputs": [],
    }

    # ---------------------------------------------------------------- synth
    def render(ttl: np.ndarray, fname: str, label: str, rows) -> dict:
        style = Style(ttl, style_a.dp.copy())
        norms = row_norm_report(ttl)
        if norms["max_abs_dev_from_unit"] > 1e-4:
            print(f"  !! WARNING off-unit rows in {fname}: {norms}")
        fpath = os.path.join(SAVE_DIR, fname)
        np.random.seed(SEED)  # identical vocoder noise for every clip
        with timer(f"Synthesizing {label}"):
            wav, duration = text_to_speech(TEXT, LANG, style, TOTAL_STEP, SPEED)
        trimmed = wav[0, : int(text_to_speech.sample_rate * duration[0].item())]
        sf.write(fpath, trimmed, text_to_speech.sample_rate)
        return {
            "file": fname,
            "label": label,
            "rows_from_b": list(rows),
            "n_rows_from_b": len(list(rows)),
            "duration_sec": float(duration[0].item()),
            "row_norms": norms,
            "_wav": trimmed,
        }

    entries = []
    entries.append(render(ttl_a.copy(), f"endpoint_{VOICE_A_NAME}.wav", VOICE_A_NAME, []))
    entries.append(
        render(ttl_b.copy(), f"endpoint_{VOICE_B_NAME}.wav", VOICE_B_NAME, range(N_ROWS))
    )

    for lo, hi in BANDS:
        rows = range(lo, hi)
        entries.append(
            render(
                swap_rows(ttl_a, ttl_b, rows),
                f"band_rows{lo:02d}-{hi - 1:02d}_from_{VOICE_B_NAME}.wav",
                f"rows {lo}-{hi - 1} from {VOICE_B_NAME}",
                rows,
            )
        )

    entries.append(
        render(
            swap_rows(ttl_a, ttl_b, active_rows),
            f"set_active_from_{VOICE_B_NAME}.wav",
            f"{len(active_rows)} active rows from {VOICE_B_NAME}",
            active_rows,
        )
    )
    entries.append(
        render(
            swap_rows(ttl_a, ttl_b, inactive_rows),
            f"set_inactive_from_{VOICE_B_NAME}.wav",
            f"{len(inactive_rows)} inactive rows from {VOICE_B_NAME}",
            inactive_rows,
        )
    )

    # The doc's literal procedure: 50 single-row hybrids.
    for i in range(N_ROWS):
        entries.append(
            render(
                swap_rows(ttl_a, ttl_b, [i]),
                os.path.join(SINGLE_ROW_SUBDIR, f"row{i:02d}_from_{VOICE_B_NAME}.wav"),
                f"row {i} from {VOICE_B_NAME}",
                [i],
            )
        )

    # ---------------------------------------------------------------- measure
    sr = text_to_speech.sample_rate
    spec = {e["file"]: log_spectrum(e["_wav"], sr) for e in entries}
    spec_a = spec[f"endpoint_{VOICE_A_NAME}.wav"]
    spec_b = spec[f"endpoint_{VOICE_B_NAME}.wav"]
    d_ab = spectral_distance(spec_a, spec_b)

    for e in entries:
        s = spec[e["file"]]
        d_a = spectral_distance(s, spec_a)
        d_b = spectral_distance(s, spec_b)
        e["dist_to_a"] = round(d_a, 4)
        e["dist_to_b"] = round(d_b, 4)
        e["travel"] = round(d_a / (d_a + d_b), 4) if (d_a + d_b) > 0 else 0.0
        e["delta_share_pct"] = round(float(delta_share[e["rows_from_b"]].sum() * 100), 3)
        del e["_wav"]

    manifest["endpoint_spectral_distance_a_vs_b"] = round(d_ab, 4)
    manifest["outputs"] = entries

    singles = [e for e in entries if e["n_rows_from_b"] == 1]
    singles_sorted = sorted(singles, key=lambda e: -e["dist_to_a"])
    print("\nSingle-row swaps ranked by audio movement away from A:")
    for e in singles_sorted[:12]:
        print(
            f"  {e['label']:>22}: dist_to_A {e['dist_to_a']:.4f}  "
            f"travel {e['travel']:.3f}  delta share {e['delta_share_pct']:.1f}%"
        )
    print("  ... quietest:")
    for e in singles_sorted[-5:]:
        print(
            f"  {e['label']:>22}: dist_to_A {e['dist_to_a']:.4f}  "
            f"travel {e['travel']:.3f}  delta share {e['delta_share_pct']:.1f}%"
        )
    print(f"\nA vs B spectral distance (full swap): {d_ab:.4f}")
    print("\nMulti-row sets:")
    for e in entries:
        if e["n_rows_from_b"] not in (1,):
            print(
                f"  {e['file']:>34}: travel {e['travel']:.3f}  "
                f"dist_to_A {e['dist_to_a']:.4f}  dist_to_B {e['dist_to_b']:.4f}  "
                f"delta share {e['delta_share_pct']:.1f}%"
            )

    worst = max(entries, key=lambda e: e["row_norms"]["max_abs_dev_from_unit"])
    print(
        f"\nWorst row-norm deviation over all {len(entries)} tensors: "
        f"{worst['row_norms']['max_abs_dev_from_unit']:.3e} ({worst['file']})"
    )
    manifest["row_norm_check"] = {
        "worst_file": worst["file"],
        "worst_max_abs_dev_from_unit": worst["row_norms"]["max_abs_dev_from_unit"],
        "note": "row swaps between two unit-norm presets preserve unit rows exactly",
    }

    manifest_path = os.path.join(SAVE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
