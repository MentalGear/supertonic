"""
Cross-seed disagreement analysis (Phase 2a, glitch-instability hypothesis).

`sample_noisy_latent()` in helper.py draws its noisy latent unseeded, so the
same style tensor renders differently every time -- and a listener has
already heard that difference matter: bench 6's control clips (unperturbed
style_ttl/style_dp, a shipped preset) were called "hiccupy" at one vocoder
seed and "normal and the best" at another, same preset, same sentence. This
script asks whether an unstable *word* leaves a signature that shows up
without needing a listener for every render.

Question: does instability show up as disagreement ACROSS seeds at a
well-localized word, even though a single render carries no absolute
signature that separates a hiccup from an ordinary consonant transient?
(Two single-render detectors were tried first and both failed this
distinction -- see phase2a_timing_drift.py's spectral-flux check and the
per-word log-mel check; both are documented as failing in
docs/GLITCH_MITIGATION.md.)

Analysis-only: reads existing WAVs in
results/listening_sets/phase2a_seed_variance/ (12 vocoder seeds per
condition, style and text held fixed within a condition), transcribes each
with faster-whisper (word timestamps), computes log-mel spectrograms, and
for each of the 6 (text, style) conditions measures, across the 12 seeds:
  - per-word duration spread (std, ms)
  - per-word start-boundary spread (std, ms)
  - per-word mel-content spread (mean pairwise L2 distance between
    per-word-averaged, RMS-normalized log-mel vectors)
  - whole-utterance medoid / outlier (frame-aligned, since duration is
    pinned exactly across seeds within a condition)

Finding: cross-seed disagreement WORKS as a word-level diagnostic. Ranking
words by combined z-scored spread (duration + start + mel) puts the
listener-flagged words at the top in every condition it was checked against:
'wood' and 'chuck' rank 1-2 of 13 in the woodchuck sentence under all three
styles tested, the final 'wood?' rises to rank 3 of 13 specifically under
the true style that a listener separately flagged there, and 'morning.'
ranks 2 of 9 in the seashells sentence under all three styles. Function
words sit at the bottom of every ranking. This localizes WHICH WORDS are
unstable -- it does not rank WHICH RENDER is good: a medoid-of-N check built
on this same distance data (see MEDOID_PAIRS below) put the listener's
"clean" seed as the bigger outlier in 3 of 4 directional checks, so that use
is contradicted and should not be relied on as-is.

Does not render any audio.
"""
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
from faster_whisper import WhisperModel

SEED_DIR = "results/listening_sets/phase2a_seed_variance"
OUT_JSON = "results/phase2a/cross_seed_disagreement.json"

N_FFT = 1024
HOP = 256
N_MELS = 80

SEED_FNAME_RE = re.compile(r"^(.*)_seed(\d+)\.wav$")

FLAGGED = {
    ("woodchuck_control", "20261069"): "chuck (flagged hiccupy)",
    ("seashells_control", "20261449"): "morning (flagged glitch)",
}
FLAGGED_WORDS = {
    "woodchuck_control": {"chuck"},
    "seashells_control": {"morning"},
}
# "woodchuck"/final-"wood" region flagged under a *true* style at seed 20261295
TRUE_STYLE_FLAG_SEED = "20261295"
TRUE_STYLE_FLAG_WORDS = {"wood", "woodchuck", "chuck"}

MEDOID_PAIRS = {
    "woodchuck_control": {"hiccupy": "20261069", "clean_best": "20261295"},
    "seashells_control": {"hiccupy": "20261449", "clean_best": "20262075"},
}

_model = None
def get_model():
    global _model
    if _model is None:
        print("Loading faster-whisper tiny.en...")
        _model = WhisperModel("Systran/faster-whisper-tiny.en", device="cpu",
                               compute_type="int8", local_files_only=True)
    return _model

_tw_cache = {}
def transcribe_words(path):
    if path in _tw_cache:
        return _tw_cache[path]
    model = get_model()
    segments, _ = model.transcribe(path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    _tw_cache[path] = words
    return words

def norm_word(w):
    return w.lower().strip(".,!?")

def discover_conditions(seed_dir):
    conditions = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(seed_dir, "*.wav"))):
        base = os.path.basename(f)
        m = SEED_FNAME_RE.match(base)
        if not m:
            continue
        cond, seed = m.group(1), m.group(2)
        conditions[cond].append((seed, f))
    for cond in conditions:
        conditions[cond].sort()
    return dict(conditions)

def rms_normalize(y, target_rms=0.05):
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12
    return (y * (target_rms / rms)).astype(np.float32)

def log_mel(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = rms_normalize(y.astype(np.float32))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=0, fmax=sr / 2, power=2.0)
    return np.log(np.maximum(S, 1e-10)), sr

def frame_index(t, sr, hop):
    return int(round(t * sr / hop))

def analyze_condition(cond, seed_files):
    seeds = [s for s, _ in seed_files]
    paths = {s: p for s, p in seed_files}

    words_by_seed = {s: transcribe_words(p) for s, p in seed_files}
    norm_seqs = {s: tuple(norm_word(w["word"]) for w in words_by_seed[s]) for s in seeds}
    seq_counts = defaultdict(list)
    for s, seq in norm_seqs.items():
        seq_counts[seq].append(s)
    canonical_seq, canonical_seeds = max(seq_counts.items(), key=lambda kv: len(kv[1]))
    mismatched_seeds = [s for s in seeds if s not in canonical_seeds]

    mels = {}
    srs = {}
    for s, p in seed_files:
        mels[s], srs[s] = log_mel(p)
    sr = srs[seeds[0]]
    assert len(set(srs.values())) == 1

    # ---- whole-utterance frame-aligned medoid/outlier ----
    n_frames = min(m.shape[1] for m in mels.values())
    stacked = np.stack([mels[s][:, :n_frames] for s in seeds], axis=0)  # (12, mel, frames)
    flat = stacked.reshape(len(seeds), -1)
    dmat = np.zeros((len(seeds), len(seeds)))
    for i, j in itertools.combinations(range(len(seeds)), 2):
        d = float(np.mean(np.abs(flat[i] - flat[j])))
        dmat[i, j] = dmat[j, i] = d
    summed = dmat.sum(axis=1)
    order = np.argsort(summed)
    ranked_seeds = [(seeds[k], float(summed[k])) for k in order]
    medoid_seed = ranked_seeds[0][0]
    outlier_seed = ranked_seeds[-1][0]

    # ---- per-word stats (only over the canonical-sequence seeds) ----
    n_words = len(canonical_seq)
    per_word = []
    for wi in range(n_words):
        durations = []
        starts = []
        mel_vecs = []
        for s in canonical_seeds:
            w = words_by_seed[s][wi]
            durations.append(w["end"] - w["start"])
            starts.append(w["start"])
            f_lo = frame_index(w["start"], sr, HOP)
            f_hi = max(frame_index(w["end"], sr, HOP), f_lo + 1)
            f_hi = min(f_hi, mels[s].shape[1])
            f_lo = min(f_lo, mels[s].shape[1] - 1)
            seg = mels[s][:, f_lo:f_hi]
            mel_vecs.append(seg.mean(axis=1))
        durations = np.array(durations)
        starts = np.array(starts)
        mel_vecs = np.stack(mel_vecs, axis=0)  # (n_seeds_canon, 80)

        # mean pairwise L2 distance among mel vectors
        n = mel_vecs.shape[0]
        pair_d = []
        for i, j in itertools.combinations(range(n), 2):
            pair_d.append(float(np.linalg.norm(mel_vecs[i] - mel_vecs[j])))
        mel_spread = float(np.mean(pair_d)) if pair_d else 0.0

        per_word.append({
            "index": wi,
            "word": words_by_seed[canonical_seeds[0]][wi]["word"],
            "duration_std_ms": float(durations.std() * 1000),
            "duration_mean_ms": float(durations.mean() * 1000),
            "start_std_ms": float(starts.std() * 1000),
            "mel_spread_l2": mel_spread,
        })

    # combined z-scored rank
    def zvals(key):
        vals = np.array([w[key] for w in per_word])
        std = vals.std()
        return (vals - vals.mean()) / std if std > 0 else np.zeros_like(vals)

    z_dur = zvals("duration_std_ms")
    z_start = zvals("start_std_ms")
    z_mel = zvals("mel_spread_l2")
    for i, w in enumerate(per_word):
        w["combined_z"] = float(z_dur[i] + z_start[i] + z_mel[i])

    ranked_by_combined = sorted(per_word, key=lambda w: -w["combined_z"])
    ranked_by_mel = sorted(per_word, key=lambda w: -w["mel_spread_l2"])

    return {
        "n_seeds": len(seeds),
        "n_canonical_seeds": len(canonical_seeds),
        "mismatched_seeds": mismatched_seeds,
        "canonical_word_seq": list(canonical_seq),
        "per_word": per_word,
        "rank_by_combined": [w["word"] for w in ranked_by_combined],
        "rank_by_combined_full": [(w["word"], round(w["combined_z"], 3)) for w in ranked_by_combined],
        "rank_by_mel": [w["word"] for w in ranked_by_mel],
        "medoid_seed": medoid_seed,
        "outlier_seed": outlier_seed,
        "seed_distance_ranking": ranked_seeds,  # (seed, summed_distance), medoid first
    }

def main():
    conditions = discover_conditions(SEED_DIR)
    results = {}
    for cond, seed_files in sorted(conditions.items()):
        print(f"=== {cond} ({len(seed_files)} seeds) ===")
        results[cond] = analyze_condition(cond, seed_files)
        r = results[cond]
        print(f"  canonical_seeds={r['n_canonical_seeds']}/{r['n_seeds']} mismatched={r['mismatched_seeds']}")
        print(f"  rank_by_combined: {r['rank_by_combined']}")
        print(f"  medoid={r['medoid_seed']} outlier={r['outlier_seed']}")
        print(f"  seed_distance_ranking: {r['seed_distance_ranking']}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")

if __name__ == "__main__":
    main()
