"""
Phase 2a word-localized spectral difference.

Turns qualitative listener reports ("the spike at the end of 'jump'", "a
hesitation on 'woodchuck'") into a per-word, per-triple measurement.

For each of the 9 bench-6 triples in results/listening_sets/phase2a_capacity/:
  - transcribe the CONTROL (base) clip with faster-whisper tiny.en,
    word_timestamps=True, to get word boundaries
  - sanity-check by also transcribing the TRUE clip and comparing boundaries
  - compute log-mel spectrograms for base/pred/true (librosa)
  - for each word interval and each pairing (true-vs-control,
    pred-vs-control, true-vs-pred): mean |log-mel diff| and peak |log-mel
    diff| over that word's frames
  - rank words within each triple by difference

Does not render any audio. Read-only analysis of existing WAVs plus small
whisper transcription runs.
"""
import json
import os
import sys
import glob
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
from faster_whisper import WhisperModel

DATA_DIR = "results/listening_sets/phase2a_capacity"
OUT_JSON = "results/phase2a/word_diff.json"
OUT_FIG_DIR = "results/phase2a/word_diff"

N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN = 0
FMAX = None  # -> sr/2

ANNOTATED_TRIPLES = ["K4_typical_idx00161", "K16_best_idx00067", "K16_typical_idx00277"]


def discover_triples(data_dir):
    triples = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(data_dir, "*.wav"))):
        base = os.path.basename(f)[:-4]
        key, kind = base.rsplit("_", 1)
        triples[key][kind] = f
    return dict(triples)


def load_manifest_rows(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    rows_by_key = {}
    for row in manifest["rows"]:
        key = f"K{row['K']}_{row['rank_label']}_idx{row['idx']:05d}"
        rows_by_key[key] = row
    return manifest, rows_by_key


def transcribe_words(model, wav_path):
    segments, _info = model.transcribe(wav_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def log_mel(wav_path, sr_target=None):
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX or sr / 2, power=2.0,
    )
    log_S = np.log(np.maximum(S, 1e-10))
    return log_S, sr


def frame_index(t_sec, sr, hop):
    return int(round(t_sec * sr / hop))


def word_stats(diff, frame_lo, frame_hi):
    frame_hi = max(frame_hi, frame_lo + 1)
    frame_hi = min(frame_hi, diff.shape[1])
    frame_lo = max(0, min(frame_lo, diff.shape[1] - 1))
    seg = diff[:, frame_lo:frame_hi]
    if seg.size == 0:
        return 0.0, 0.0
    return float(seg.mean()), float(seg.max())


def boundary_agreement(words_a, words_b):
    """Compare word lists positionally; return list of per-word start/end
    deltas in seconds where the transcripts have the same number of words
    and same word text (case/punct-insensitive), else a note."""
    norm_a = [w["word"].lower().strip(".,!?") for w in words_a]
    norm_b = [w["word"].lower().strip(".,!?") for w in words_b]
    if norm_a != norm_b:
        return None, f"word sequences differ: {norm_a} vs {norm_b}"
    deltas = []
    for wa, wb in zip(words_a, words_b):
        deltas.append({
            "word": wa["word"],
            "start_delta": wb["start"] - wa["start"],
            "end_delta": wb["end"] - wa["end"],
        })
    return deltas, None


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    manifest, rows_by_key = load_manifest_rows(DATA_DIR)
    triples = discover_triples(DATA_DIR)

    print("Loading faster-whisper tiny.en (local_files_only, int8, cpu)...")
    model = WhisperModel(
        "Systran/faster-whisper-tiny.en",
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )

    results = {}
    alignment_report = {}

    for key in sorted(triples):
        paths = triples[key]
        assert set(paths) == {"base", "pred", "true"}, f"{key}: missing clip(s): {paths}"
        row = rows_by_key.get(key)

        # --- sample-count alignment check ---
        lens = {}
        srs = {}
        for kind, p in paths.items():
            y, sr = sf.read(p)
            lens[kind] = len(y)
            srs[kind] = sr
        aligned = len(set(lens.values())) == 1 and len(set(srs.values())) == 1
        alignment_report[key] = {"lengths": lens, "srs": srs, "aligned": aligned}
        if not aligned:
            print(f"WARNING: {key} NOT aligned: {lens} srs={srs}")

        # --- word timings from control (base) ---
        control_words = transcribe_words(model, paths["base"])
        true_words = transcribe_words(model, paths["true"])
        deltas, mismatch_note = boundary_agreement(control_words, true_words)

        # --- spectrograms ---
        log_mel_base, sr = log_mel(paths["base"])
        log_mel_pred, sr_p = log_mel(paths["pred"])
        log_mel_true, sr_t = log_mel(paths["true"])
        assert sr == sr_p == sr_t
        n_frames = min(log_mel_base.shape[1], log_mel_pred.shape[1], log_mel_true.shape[1])
        log_mel_base = log_mel_base[:, :n_frames]
        log_mel_pred = log_mel_pred[:, :n_frames]
        log_mel_true = log_mel_true[:, :n_frames]

        diff_true_vs_control = np.abs(log_mel_true - log_mel_base)
        diff_pred_vs_control = np.abs(log_mel_pred - log_mel_base)
        diff_true_vs_pred = np.abs(log_mel_true - log_mel_pred)

        # --- per-word stats ---
        per_word = []
        for w in control_words:
            f_lo = frame_index(w["start"], sr, HOP)
            f_hi = frame_index(w["end"], sr, HOP)
            mean_tc, peak_tc = word_stats(diff_true_vs_control, f_lo, f_hi)
            mean_pc, peak_pc = word_stats(diff_pred_vs_control, f_lo, f_hi)
            mean_tp, peak_tp = word_stats(diff_true_vs_pred, f_lo, f_hi)
            per_word.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "true_vs_control": {"mean": mean_tc, "peak": peak_tc},
                "pred_vs_control": {"mean": mean_pc, "peak": peak_pc},
                "true_vs_pred": {"mean": mean_tp, "peak": peak_tp},
            })

        rank_true_vs_control_mean = sorted(per_word, key=lambda w: -w["true_vs_control"]["mean"])
        rank_true_vs_control_peak = sorted(per_word, key=lambda w: -w["true_vs_control"]["peak"])
        rank_pred_vs_control_mean = sorted(per_word, key=lambda w: -w["pred_vs_control"]["mean"])
        rank_pred_vs_control_peak = sorted(per_word, key=lambda w: -w["pred_vs_control"]["peak"])

        results[key] = {
            "row": row,
            "n_frames": n_frames,
            "sr": sr,
            "hop": HOP,
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "control_words": control_words,
            "true_words": true_words,
            "boundary_deltas_true_vs_control": deltas,
            "boundary_mismatch_note": mismatch_note,
            "per_word": per_word,
            "rank_true_vs_control_mean": [w["word"] for w in rank_true_vs_control_mean],
            "rank_true_vs_control_peak": [w["word"] for w in rank_true_vs_control_peak],
            "rank_pred_vs_control_mean": [w["word"] for w in rank_pred_vs_control_mean],
            "rank_pred_vs_control_peak": [w["word"] for w in rank_pred_vs_control_peak],
        }

        print(f"{key}: words={[w['word'] for w in control_words]}")
        print(f"  top3 true-vs-control (peak): {[w['word'] for w in rank_true_vs_control_peak[:3]]}")
        print(f"  top3 pred-vs-control (peak): {[w['word'] for w in rank_pred_vs_control_peak[:3]]}")
        if mismatch_note:
            print(f"  BOUNDARY MISMATCH: {mismatch_note}")

        if key in ANNOTATED_TRIPLES:
            make_figure(key, log_mel_true, log_mel_pred, log_mel_base,
                        control_words, sr, HOP, row)

    with open(OUT_JSON, "w") as fh:
        json.dump({"alignment": alignment_report, "triples": results}, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")


def make_figure(key, log_mel_true, log_mel_pred, log_mel_base, words, sr, hop, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff_tc = np.abs(log_mel_true - log_mel_base)
    diff_pc = np.abs(log_mel_pred - log_mel_base)

    n_frames = diff_tc.shape[1]
    t_axis = np.arange(n_frames) * hop / sr

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, diff, title in [
        (axes[0], diff_tc, "true vs control  |log-mel diff|"),
        (axes[1], diff_pc, "prediction vs control  |log-mel diff|"),
    ]:
        im = ax.imshow(
            diff, origin="lower", aspect="auto",
            extent=[t_axis[0], t_axis[-1], 0, diff.shape[0]],
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_ylabel("mel bin")
        for w in words:
            ax.axvline(w["start"], color="cyan", linewidth=0.6, alpha=0.8)
            ax.text(w["start"] + 0.01, diff.shape[0] * 0.92, w["word"],
                    color="cyan", fontsize=8, rotation=0, va="top")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("time (s)")

    text = row["text"] if row else ""
    cap = (f"{key}  |  text: \"{text}\"  |  n_fft={N_FFT} hop={HOP} n_mels={N_MELS}  |  "
           f"RMS level-matched, shared vocoder seed, style_dp fixed (durations pinned), "
           f"word boundaries from faster-whisper tiny.en on the control clip")
    fig.suptitle(cap, fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUT_FIG_DIR, f"{key}_worddiff.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
