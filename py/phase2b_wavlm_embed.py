"""Phase 2b, stage 2b: extract WavLM-large frame features from the generated clips.

new-plan.md Phase 2b names "ECAPA/WavLM embedding -> style_ttl". Only ECAPA was
tested, and ECAPA is a 192-dim utterance-level speaker vector: a linear map out
of it can span at most 192 of the 6,120 target directions, so the negative
result was partly a statement about probe capacity. WavLM-large is 1024-dim per
frame per transformer layer, so mean+std pooling gives 2048 dims per layer and
concatenating layers gives more -- a strictly fairer test of the same question.

This script only EXTRACTS and CACHES. It does the full 24-layer forward once per
clip (all layers come out of one forward pass, so keeping all of them is free)
and stores, per layer, the mean and the std over frames. The probe
(`phase2b_wavlm.py`) then sweeps layers and poolings off the cache without ever
re-running the model.

Waveforms are layer-normalised before the model, as torchaudio's WAVLM_LARGE
bundle requires (`bundle._normalize_waveform is True`).

Outputs (under results/phase2b/):
  * wavlm_feats.npz       -- mean (N, 24, 1024) and std (N, 24, 1024), float32,
                             in generation_manifest record order.
  * wavlm_reference_feats.npz -- the same for the 10 presets x 8 sentences
                             reference renders, used for the 1-NN speaker-ID
                             sanity check that mirrors the ECAPA one.

Usage (from py/):
    python3 phase2b_wavlm_embed.py [--limit N]
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

OUT_DIR = "results/phase2b"
AUDIO_DIR = os.path.join(OUT_DIR, "audio16k")
REF_DIR = os.path.join(OUT_DIR, "reference16k")
SR = 16000
PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
N_LAYERS = 24
DIM = 1024


def load_model():
    torch.set_num_threads(os.cpu_count() or 4)
    bundle = torchaudio.pipelines.WAVLM_LARGE
    assert bundle.sample_rate == SR, bundle.sample_rate
    model = bundle.get_model()
    model.eval()
    return bundle, model


def pooled_features(model, normalize, path):
    """One forward pass -> per-layer (mean, std) over frames."""
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    x = torch.from_numpy(wav)[None, :]
    if normalize:
        x = torch.nn.functional.layer_norm(x, x.shape)
    with torch.no_grad():
        feats, _ = model.extract_features(x)
    assert len(feats) == N_LAYERS, len(feats)
    mean = np.empty((N_LAYERS, DIM), dtype=np.float32)
    std = np.empty((N_LAYERS, DIM), dtype=np.float32)
    for li, f in enumerate(feats):
        f = f[0]  # (T, 1024)
        mean[li] = f.mean(0).numpy()
        std[li] = f.std(0).numpy()
    return mean, std


def extract(model, normalize, paths, tag):
    M = np.zeros((len(paths), N_LAYERS, DIM), dtype=np.float32)
    S = np.zeros((len(paths), N_LAYERS, DIM), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        M[i], S[i] = pooled_features(model, normalize, p)
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"  [{tag}] {i + 1}/{len(paths)}  {rate:.2f}s/clip  "
                  f"eta {(len(paths) - i - 1) * rate / 60:.1f} min", flush=True)
    return M, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N generated clips (smoke test)")
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    records = meta["records"]
    if args.limit:
        records = records[: args.limit]

    bundle, model = load_model()
    norm = bool(getattr(bundle, "_normalize_waveform", True))
    print(f"WavLM-large loaded; normalize_waveform={norm}; "
          f"{len(records)} clips, {N_LAYERS} layers x {DIM} dims", flush=True)

    ref_paths, ref_labels, ref_text = [], [], []
    for p in PRESETS:
        for ti in range(8):
            ref_paths.append(os.path.join(REF_DIR, f"{p}_t{ti}.wav"))
            ref_labels.append(p)
            ref_text.append(ti)
    Mr, Sr = extract(model, norm, ref_paths, "reference")
    np.savez_compressed(os.path.join(OUT_DIR, "wavlm_reference_feats.npz"),
                        mean=Mr, std=Sr, labels=np.array(ref_labels),
                        text_idx=np.array(ref_text))
    print(f"Wrote {OUT_DIR}/wavlm_reference_feats.npz", flush=True)

    paths = [os.path.join(AUDIO_DIR, r["file"]) for r in records]
    M, S = extract(model, norm, paths, "generated")
    np.savez(os.path.join(OUT_DIR, "wavlm_feats.npz"), mean=M, std=S,
             n=np.array([len(records)]))
    print(f"Wrote {OUT_DIR}/wavlm_feats.npz for {len(records)} clips.")


if __name__ == "__main__":
    main()
