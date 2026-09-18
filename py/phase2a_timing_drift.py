"""
Phase 2a timing-drift check.

Question: does style_ttl redistribute time BETWEEN words while total
utterance duration stays pinned (style_dp fixed)? phase2a_word_diff.py
turned up boundary drifts up to 0.24s on K16_best/woodchuck between the
true-style clip and the unperturbed control. This script asks whether that
drift is real (exceeds the measurement noise) or just whisper timestamp /
vocoder sampling jitter.

Three measurements, same units (seconds, per word start/end boundary):

1. Whisper determinism floor -- transcribe the same wav twice, see how much
   faster-whisper's own timestamps move with nothing else changing.
2. Vocoder-sampling floor -- using results/listening_sets/phase2a_seed_variance/
   (same style, 12 different vocoder seeds), how much do word boundaries move
   from sampling alone?
3. Style-induced drift -- for the 9 bench-6 triples in
   results/listening_sets/phase2a_capacity/ (shared seed within a triple),
   true-vs-control and pred-vs-control boundary deltas.

Read-only analysis of existing WAVs. Does not render any audio.
"""
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

CAPACITY_DIR = "results/listening_sets/phase2a_capacity"
SEED_DIR = "results/listening_sets/phase2a_seed_variance"
OUT_JSON = "results/phase2a/timing_drift.json"

AUDIBLE_KEYS = {"K4_typical", "K16_best", "K16_typical", "K16_worst"}
INAUDIBLE_KEYS = {"K4_best", "K4_worst", "K64_best", "K64_typical", "K64_worst"}

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading faster-whisper tiny.en (local_files_only, int8, cpu)...")
        _model = WhisperModel(
            "Systran/faster-whisper-tiny.en",
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )
    return _model


_transcribe_cache = {}


def transcribe_words(wav_path):
    if wav_path in _transcribe_cache:
        return _transcribe_cache[wav_path]
    model = get_model()
    segments, _info = model.transcribe(wav_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    _transcribe_cache[wav_path] = words
    return words


def transcribe_words_fresh(wav_path):
    """Bypass the cache -- used for the determinism check, which needs two
    independent transcription runs of the same file."""
    model = get_model()
    segments, _info = model.transcribe(wav_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    return words


def norm_word(w):
    return w["word"].lower().strip(".,!?")


def boundary_deltas(words_a, words_b):
    """Positional comparison; None + note if word sequences disagree."""
    norm_a = [norm_word(w) for w in words_a]
    norm_b = [norm_word(w) for w in words_b]
    if norm_a != norm_b:
        return None, f"word sequences differ: {norm_a} vs {norm_b}"
    deltas = []
    for i, (wa, wb) in enumerate(zip(words_a, words_b)):
        deltas.append({
            "index": i,
            "word": wa["word"],
            "start_delta": wb["start"] - wa["start"],
            "end_delta": wb["end"] - wa["end"],
        })
    return deltas, None


def stats(values):
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "p95": float(np.percentile(arr, 95)),
    }


def abs_stats_from_deltas(delta_lists, field):
    """delta_lists: list of per-comparison delta-lists (each a list of dicts
    with 'start_delta'/'end_delta'). Pools all words across all comparisons."""
    vals = []
    for deltas in delta_lists:
        for d in deltas:
            vals.append(abs(d[field]))
    return stats(vals)


def signed_stats_from_deltas(delta_lists, field):
    vals = []
    for deltas in delta_lists:
        for d in deltas:
            vals.append(d[field])
    return stats(vals)


# ---------------------------------------------------------------------------
# 1. Whisper determinism floor
# ---------------------------------------------------------------------------

def determinism_floor():
    # A handful of files spanning both directories used below.
    files = [
        os.path.join(CAPACITY_DIR, "K16_best_idx00067_true.wav"),
        os.path.join(CAPACITY_DIR, "K16_best_idx00067_base.wav"),
        os.path.join(CAPACITY_DIR, "K4_typical_idx00161_true.wav"),
        os.path.join(CAPACITY_DIR, "K64_worst_idx00393_base.wav"),
        os.path.join(SEED_DIR, "woodchuck_control_seed20261069.wav"),
        os.path.join(SEED_DIR, "seashells_true_K16_worst_seed20270908.wav"),
    ]
    files = [f for f in files if os.path.exists(f)]

    per_file = {}
    all_deltas = []
    mismatches = []
    for f in files:
        run1 = transcribe_words_fresh(f)
        run2 = transcribe_words_fresh(f)
        deltas, note = boundary_deltas(run1, run2)
        per_file[os.path.basename(f)] = {
            "n_words": len(run1),
            "mismatch_note": note,
            "deltas": deltas,
        }
        if deltas is not None:
            all_deltas.append(deltas)
        else:
            mismatches.append(os.path.basename(f))

    return {
        "files_tested": [os.path.basename(f) for f in files],
        "n_files": len(files),
        "n_files_mismatched": len(mismatches),
        "mismatched_files": mismatches,
        "abs_start_delta": abs_stats_from_deltas(all_deltas, "start_delta"),
        "abs_end_delta": abs_stats_from_deltas(all_deltas, "end_delta"),
        "per_file": per_file,
    }


# ---------------------------------------------------------------------------
# 2. Vocoder-sampling floor (style fixed, seed varies)
# ---------------------------------------------------------------------------

SEED_FNAME_RE = re.compile(r"^(.*)_seed(\d+)\.wav$")


def discover_seed_conditions(seed_dir):
    """Group seed_variance files by condition (everything but the seed)."""
    conditions = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(seed_dir, "*.wav"))):
        base = os.path.basename(f)
        m = SEED_FNAME_RE.match(base)
        if not m:
            continue
        cond, seed = m.group(1), m.group(2)
        conditions[cond].append((seed, f))
    return dict(conditions)


def sampling_floor():
    conditions = discover_seed_conditions(SEED_DIR)
    per_condition = {}
    all_deltas = []
    all_lengths_pinned = True

    for cond, seed_files in sorted(conditions.items()):
        seed_files.sort()
        lengths = {}
        srs = {}
        for seed, f in seed_files:
            info = sf.info(f)
            lengths[seed] = info.frames
            srs[seed] = info.samplerate
        length_set = set(lengths.values())
        sr_set = set(srs.values())
        duration_pinned = len(length_set) == 1 and len(sr_set) == 1
        if not duration_pinned:
            all_lengths_pinned = False

        words_by_seed = {seed: transcribe_words(f) for seed, f in seed_files}

        cond_deltas = []
        n_mismatched = 0
        mismatch_examples = []
        for (seed_a, _), (seed_b, _) in itertools.combinations(seed_files, 2):
            deltas, note = boundary_deltas(words_by_seed[seed_a], words_by_seed[seed_b])
            if deltas is None:
                n_mismatched += 1
                if len(mismatch_examples) < 3:
                    mismatch_examples.append({"pair": [seed_a, seed_b], "note": note})
                continue
            cond_deltas.append(deltas)

        all_deltas.extend(cond_deltas)
        per_condition[cond] = {
            "n_seeds": len(seed_files),
            "n_pairs_total": len(seed_files) * (len(seed_files) - 1) // 2,
            "n_pairs_usable": len(cond_deltas),
            "n_pairs_mismatched": n_mismatched,
            "mismatch_examples": mismatch_examples,
            "sample_lengths_by_seed": lengths,
            "duration_pinned_across_seeds": duration_pinned,
            "abs_start_delta": abs_stats_from_deltas(cond_deltas, "start_delta"),
            "abs_end_delta": abs_stats_from_deltas(cond_deltas, "end_delta"),
            "signed_start_delta": signed_stats_from_deltas(cond_deltas, "start_delta"),
        }

    pooled = {
        "n_conditions": len(conditions),
        "n_pairs_usable": len(all_deltas),
        "abs_start_delta": abs_stats_from_deltas(all_deltas, "start_delta"),
        "abs_end_delta": abs_stats_from_deltas(all_deltas, "end_delta"),
        "duration_pinned_across_all_seeds": all_lengths_pinned,
    }

    return {"conditions": per_condition, "pooled": pooled}


# ---------------------------------------------------------------------------
# 3. Style-induced drift (bench-6 triples, seed shared within a triple)
# ---------------------------------------------------------------------------

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


def condition_label(key):
    # key like "K16_best_idx00067" -> "K16_best"
    return key.rsplit("_idx", 1)[0]


def interior_extreme(deltas, field):
    """Max abs delta among interior boundaries (excludes the very first
    word's start and the very last word's end, which are pinned near the
    utterance edges almost by construction)."""
    if not deltas:
        return None
    interior = deltas
    if field == "start_delta":
        interior = deltas[1:]  # drop first word's start
    elif field == "end_delta":
        interior = deltas[:-1]  # drop last word's end
    if not interior:
        return None
    return max(abs(d[field]) for d in interior)


def style_induced_drift():
    manifest, rows_by_key = load_manifest_rows(CAPACITY_DIR)
    triples = discover_triples(CAPACITY_DIR)

    per_triple = {}
    tvc_deltas_all = []
    pvc_deltas_all = []

    for key in sorted(triples):
        paths = triples[key]
        assert set(paths) == {"base", "pred", "true"}, f"{key}: missing clip(s)"
        row = rows_by_key.get(key)
        cond = condition_label(key)

        lengths = {kind: sf.info(p).frames for kind, p in paths.items()}
        srs = {kind: sf.info(p).samplerate for kind, p in paths.items()}
        duration_pinned = len(set(lengths.values())) == 1 and len(set(srs.values())) == 1

        base_words = transcribe_words(paths["base"])
        true_words = transcribe_words(paths["true"])
        pred_words = transcribe_words(paths["pred"])

        tvc, tvc_note = boundary_deltas(base_words, true_words)
        pvc, pvc_note = boundary_deltas(base_words, pred_words)

        if tvc is not None:
            tvc_deltas_all.append(tvc)
        if pvc is not None:
            pvc_deltas_all.append(pvc)

        entry = {
            "K": row["K"] if row else None,
            "rank_label": row["rank_label"] if row else None,
            "sample_r2": row["sample_r2"] if row else None,
            "audible": cond in AUDIBLE_KEYS,
            "text": row["text"] if row else None,
            "sample_lengths": lengths,
            "srs": srs,
            "duration_pinned_exact_samples": duration_pinned,
            "true_vs_control": {
                "mismatch_note": tvc_note,
                "deltas": tvc,
                "max_abs_start_delta": max((abs(d["start_delta"]) for d in tvc), default=None) if tvc else None,
                "max_abs_end_delta": max((abs(d["end_delta"]) for d in tvc), default=None) if tvc else None,
                "max_abs_interior_start_delta": interior_extreme(tvc, "start_delta") if tvc else None,
                "max_abs_interior_end_delta": interior_extreme(tvc, "end_delta") if tvc else None,
                "first_word_start_delta": tvc[0]["start_delta"] if tvc else None,
                "last_word_end_delta": tvc[-1]["end_delta"] if tvc else None,
            },
            "pred_vs_control": {
                "mismatch_note": pvc_note,
                "deltas": pvc,
                "max_abs_start_delta": max((abs(d["start_delta"]) for d in pvc), default=None) if pvc else None,
                "max_abs_end_delta": max((abs(d["end_delta"]) for d in pvc), default=None) if pvc else None,
                "max_abs_interior_start_delta": interior_extreme(pvc, "start_delta") if pvc else None,
                "max_abs_interior_end_delta": interior_extreme(pvc, "end_delta") if pvc else None,
            },
        }
        per_triple[key] = entry
        print(f"{key}: audible={entry['audible']} K={entry['K']} r2={entry['sample_r2']:.3f} "
              f"dur_pinned={duration_pinned} "
              f"tvc_max_interior_start={entry['true_vs_control']['max_abs_interior_start_delta']}")

    pooled = {
        "n_triples": len(triples),
        "true_vs_control_abs_start_delta": abs_stats_from_deltas(tvc_deltas_all, "start_delta"),
        "true_vs_control_abs_end_delta": abs_stats_from_deltas(tvc_deltas_all, "end_delta"),
        "pred_vs_control_abs_start_delta": abs_stats_from_deltas(pvc_deltas_all, "start_delta"),
        "pred_vs_control_abs_end_delta": abs_stats_from_deltas(pvc_deltas_all, "end_delta"),
    }

    return {"per_triple": per_triple, "pooled": pooled}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    print("=== 1. Whisper determinism floor ===")
    det = determinism_floor()
    print(json.dumps({"abs_start_delta": det["abs_start_delta"], "abs_end_delta": det["abs_end_delta"]}, indent=2))

    print("\n=== 2. Vocoder-sampling floor ===")
    samp = sampling_floor()
    print(json.dumps(samp["pooled"], indent=2))

    print("\n=== 3. Style-induced drift ===")
    style = style_induced_drift()
    print(json.dumps(style["pooled"], indent=2))

    # --- 4/5/6: comparison, scaling, duration-pinning summary ---
    floor_mean = samp["pooled"]["abs_start_delta"]["mean"]
    floor_p95 = samp["pooled"]["abs_start_delta"]["p95"]
    floor_max = samp["pooled"]["abs_start_delta"]["max"]

    per_triple_summary = []
    for key, e in sorted(style["per_triple"].items()):
        max_interior = e["true_vs_control"]["max_abs_interior_start_delta"]
        per_triple_summary.append({
            "key": key,
            "K": e["K"],
            "rank_label": e["rank_label"],
            "sample_r2": e["sample_r2"],
            "audible": e["audible"],
            "duration_pinned_exact_samples": e["duration_pinned_exact_samples"],
            "max_abs_interior_start_delta_true_vs_control": max_interior,
            "exceeds_sampling_floor_p95": (max_interior is not None and max_interior > floor_p95),
            "exceeds_sampling_floor_max": (max_interior is not None and max_interior > floor_max),
        })

    n_exceed_p95 = sum(1 for r in per_triple_summary if r["exceeds_sampling_floor_p95"])
    audible_exceed = sum(1 for r in per_triple_summary if r["audible"] and r["exceeds_sampling_floor_p95"])
    inaudible_exceed = sum(1 for r in per_triple_summary if not r["audible"] and r["exceeds_sampling_floor_p95"])
    n_audible = sum(1 for r in per_triple_summary if r["audible"])
    n_inaudible = sum(1 for r in per_triple_summary if not r["audible"])

    all_duration_pinned = all(r["duration_pinned_exact_samples"] for r in per_triple_summary)

    comparison = {
        "sampling_floor_abs_start_delta_mean": floor_mean,
        "sampling_floor_abs_start_delta_p95": floor_p95,
        "sampling_floor_abs_start_delta_max": floor_max,
        "style_drift_abs_start_delta_mean_true_vs_control": style["pooled"]["true_vs_control_abs_start_delta"]["mean"],
        "style_drift_abs_start_delta_p95_true_vs_control": style["pooled"]["true_vs_control_abs_start_delta"].get("p95"),
        "n_triples": len(per_triple_summary),
        "n_triples_exceeding_sampling_floor_p95": n_exceed_p95,
        "n_audible_triples": n_audible,
        "n_audible_triples_exceeding_floor": audible_exceed,
        "n_inaudible_triples": n_inaudible,
        "n_inaudible_triples_exceeding_floor": inaudible_exceed,
        "all_triples_duration_pinned_exact_samples": all_duration_pinned,
        "per_triple_summary": per_triple_summary,
        "verdict_style_induced_drift_exceeds_sampling_floor": (
            style["pooled"]["true_vs_control_abs_start_delta"]["mean"] > floor_p95
        ),
    }

    print("\n=== Comparison ===")
    print(json.dumps(comparison, indent=2))

    out = {
        "determinism_floor": det,
        "sampling_floor": samp,
        "style_induced_drift": style,
        "comparison": comparison,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
