"""Phase 2b, stage 2: extract speaker embeddings from the generated clips.

Two feature sets are extracted for every clip:

  * `ecapa`  -- 192-dim ECAPA-TDNN speaker embedding (speechbrain
    `spkrec-ecapa-voxceleb`, the standard off-the-shelf speaker encoder named in
    new-plan.md Phase 2b). This is the probe input the phase is actually about.
  * `spectral` -- a deliberately dumb 120-dim baseline: mean and std of 40 MFCCs
    plus mean/std of log-mel band energies. It exists so the ECAPA R^2 can be
    read against something trivial; if a bag of MFCC moments does as well, the
    result is about audio statistics, not about speaker-verification space.

Usage (from py/):
    python3 phase2b_embed.py
"""

import json
import os

import numpy as np
import soundfile as sf

OUT_DIR = "results/phase2b"
AUDIO_DIR = os.path.join(OUT_DIR, "audio16k")
SR = 16000
ECAPA_CACHE = "/home/user/.cache/ecapa"


def spectral_features(wav: np.ndarray) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=40, n_mels=64)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=SR, n_mels=20))
    return np.concatenate([mfcc.mean(1), mfcc.std(1), mel.mean(1), mel.std(1)]).astype(np.float32)


def main():
    with open(os.path.join(OUT_DIR, "generation_manifest.json")) as f:
        meta = json.load(f)
    records = meta["records"]

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_CACHE,
        run_opts={"device": "cpu"},
    )
    enc.eval()

    ecapa = np.zeros((len(records), 192), dtype=np.float32)
    spec = np.zeros((len(records), 120), dtype=np.float32)

    for i, r in enumerate(records):
        wav, sr = sf.read(os.path.join(AUDIO_DIR, r["file"]), dtype="float32")
        assert sr == SR
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wav)[None, :])
        ecapa[i] = e.squeeze().numpy()
        spec[i] = spectral_features(wav)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)} embedded", flush=True)

    np.savez_compressed(os.path.join(OUT_DIR, "embeddings.npz"), ecapa=ecapa, spectral=spec)
    print(f"Saved embeddings for {len(records)} clips.")
    print(f"ECAPA norm range: [{np.linalg.norm(ecapa, axis=1).min():.2f}, "
          f"{np.linalg.norm(ecapa, axis=1).max():.2f}]")


if __name__ == "__main__":
    main()
