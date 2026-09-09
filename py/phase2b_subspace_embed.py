"""Phase 2b subspace, stage 2: extract WavLM-large features from the ladder corpus.

Mirrors `phase2b_wavlm_embed.py` (see its docstring for why WavLM-large over
ECAPA). Reuses `load_model()` and `pooled_features()` from that module
verbatim -- this script does not re-run the model or the layer pooling logic
itself, it only decides what waveform gets fed in and what gets cached.

Two differences from the 2b version:

  * Every clip is RMS level-matched to a common target RMS before feature
    extraction, so a loudness difference between conditions cannot pose as a
    feature difference. A clip whose RMS is at or below `SILENCE_EPS` (i.e.
    effectively silent) is skipped rather than blown up by division -- it
    carries no speech signal for WavLM to encode anyway. The pre-normalization
    RMS of every clip that IS kept is recorded in the output (`rms_pre`), so a
    downstream probe can check whether the loudness channel leaks into
    predictions, per CLAUDE.md's "calibrate a distance/control before citing
    it as evidence".
  * All 24 layers' (mean, std) are cached, exactly as phase2b_wavlm_embed does,
    so a layer sweep is possible without re-running WavLM.

Level-matching is done by rescaling the waveform in memory and handing the
result to `pooled_features()` through an in-memory WAV buffer (soundfile
accepts file-like objects), rather than reimplementing the forward pass /
pooling with a waveform-array signature. `pooled_features` itself still does
its own layer-norm-for-the-model-input step (`normalize`), which is a
different, per-utterance operation required by the WAVLM_LARGE bundle and is
unrelated to the RMS level match applied here.

Outputs (default, under results/phase2b_subspace/):
  * wavlm_feats.npz -- mean (N, 24, 1024) and std (N, 24, 1024) float32, plus
    idx (N,) int64 and K (N,) int64 (the manifest's `idx`/`K` fields for each
    kept clip, so downstream code can join back to the manifest and to
    subspace.npz's per-K `c_realized_K{K}` arrays), rms_pre (N,) float32 (the
    RMS of each clip before level-matching), target_rms (scalar), and n (the
    count actually written, which can be less than the manifest's record
    count if any clips were skipped as silent).

Usage (from py/):
    python3 phase2b_subspace_embed.py [--limit N] [--k K]
    python3 phase2b_subspace_embed.py --manifest OTHER/manifest.json \
        --audio-dir OTHER/audio16k --out OTHER/wavlm_feats.npz  # smoke test
"""

import argparse
import io
import json
import os
import time

import numpy as np
import soundfile as sf

from phase2b_wavlm_embed import DIM, N_LAYERS, load_model, pooled_features

OUT_DIR = "results/phase2b_subspace"
SR = 16000
TARGET_RMS = 0.05  # close to a typical clip's natural level in this corpus
SILENCE_EPS = 1e-6  # RMS at or below this is treated as silence and skipped


def level_match(wav: np.ndarray, target_rms: float = TARGET_RMS, silence_eps: float = SILENCE_EPS):
    """Rescale `wav` to `target_rms`. Returns (leveled, rms_pre) or (None, rms_pre)
    if the clip is at/below the silence threshold (leveled is None -> skip)."""
    rms_pre = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms_pre <= silence_eps:
        return None, rms_pre
    leveled = (wav.astype(np.float64) * (target_rms / rms_pre)).astype(np.float32)
    peak = float(np.abs(leveled).max())
    if peak > 1.0:
        # Guard against clipping distortion feeding the model; this makes the
        # realized RMS of this one clip slightly below target_rms, which is
        # fine -- rms_pre (recorded) is what a downstream loudness-control
        # probe actually needs, not exact post-leveling RMS equality.
        leveled = leveled / peak
    return leveled, rms_pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR, help="root dir for default manifest/audio/out paths")
    ap.add_argument("--manifest", default=None, help="default: <out-dir>/manifest.json")
    ap.add_argument("--audio-dir", default=None, help="default: <out-dir>/audio16k")
    ap.add_argument("--out", default=None, help="default: <out-dir>/wavlm_feats.npz")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest records (after --k filtering)")
    ap.add_argument("--k", type=int, default=None, help="only embed records whose manifest 'K' field equals this")
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.json")
    audio_dir = args.audio_dir or os.path.join(args.out_dir, "audio16k")
    out_path = args.out or os.path.join(args.out_dir, "wavlm_feats.npz")

    with open(manifest_path) as f:
        meta = json.load(f)
    records = meta["records"]
    if args.k is not None:
        records = [r for r in records if r.get("K") == args.k]
    if args.limit:
        records = records[: args.limit]
    n_requested = len(records)

    bundle, model = load_model()
    norm = bool(getattr(bundle, "_normalize_waveform", True))
    print(f"WavLM-large loaded; normalize_waveform={norm}; target_rms={TARGET_RMS}; "
          f"{n_requested} clips requested, {N_LAYERS} layers x {DIM} dims", flush=True)

    mean_buf = np.zeros((n_requested, N_LAYERS, DIM), dtype=np.float32)
    std_buf = np.zeros((n_requested, N_LAYERS, DIM), dtype=np.float32)
    rms_pre_buf = np.zeros(n_requested, dtype=np.float32)
    idx_buf = np.zeros(n_requested, dtype=np.int64)
    k_buf = np.full(n_requested, -1, dtype=np.int64)

    n_kept = 0
    n_skipped_silent = 0
    t0 = time.time()
    for i, r in enumerate(records):
        path = os.path.join(audio_dir, r["file"])
        wav, sr = sf.read(path, dtype="float32")
        assert sr == SR, (path, sr)

        leveled, rms_pre = level_match(wav)
        if leveled is None:
            n_skipped_silent += 1
            print(f"  WARNING: skipping near-silent clip {r['file']} (rms={rms_pre:.2e})", flush=True)
            continue

        buf = io.BytesIO()
        sf.write(buf, leveled, SR, format="WAV", subtype="FLOAT")
        buf.seek(0)
        mean, std = pooled_features(model, norm, buf)

        mean_buf[n_kept] = mean
        std_buf[n_kept] = std
        rms_pre_buf[n_kept] = rms_pre
        idx_buf[n_kept] = int(r["idx"])
        k_buf[n_kept] = int(r["K"]) if "K" in r else -1
        n_kept += 1

        if n_kept % 50 == 0 or (i + 1) == n_requested:
            el = time.time() - t0
            rate = el / n_kept if n_kept else 0.0
            print(f"  {n_kept}/{n_requested} embedded  {rate:.2f}s/clip", flush=True)

    mean_buf = mean_buf[:n_kept]
    std_buf = std_buf[:n_kept]
    rms_pre_buf = rms_pre_buf[:n_kept]
    idx_buf = idx_buf[:n_kept]
    k_buf = k_buf[:n_kept]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        mean=mean_buf, std=std_buf,
        idx=idx_buf, K=k_buf, rms_pre=rms_pre_buf,
        target_rms=np.array([TARGET_RMS], dtype=np.float32),
        n=np.array([n_kept]),
    )
    print(f"\nWrote {out_path}: {n_kept}/{n_requested} clips "
          f"({n_skipped_silent} skipped as near-silent).")


if __name__ == "__main__":
    main()
