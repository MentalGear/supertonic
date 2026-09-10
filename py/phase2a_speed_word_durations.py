"""Per-word duration extraction (faster-whisper tiny.en) for the Experiment-A
speed sweep produced by phase2a_speed_rootcause.py. Tracks the flagged words
('chuck' -- first occurrence, right after "woodchuck"; 'wood' -- the
utterance-final occurrence; 'morning' -- utterance-final, only occurrence)
across speed in {1.05, 1.00, 0.95, 0.90}.

Usage (from py/), after phase2a_speed_rootcause.py has run:
    python3 phase2a_speed_word_durations.py
"""

import json
import os

from faster_whisper import WhisperModel

OUT_DIR = "results/phase2a"
REPORT_PATH = os.path.join(OUT_DIR, "speed_rootcause.json")
OUT_PATH = os.path.join(OUT_DIR, "speed_word_durations.json")
LISTEN_DIR = "results/listening_sets/phase2a_speed_rootcause"


def transcribe_words(model, wav_path):
    segments, _info = model.transcribe(wav_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def norm(w):
    return w.lower().strip(".,!?")


def main():
    report = json.load(open(REPORT_PATH))
    print("Loading faster-whisper tiny.en (local_files_only, int8, cpu)...")
    model = WhisperModel(
        "Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8", local_files_only=True
    )

    out = []
    for rec in report["speed_sweep"]:
        path = os.path.join(LISTEN_DIR, rec["trimmed_file"])
        words = transcribe_words(model, path)
        norm_words = [norm(w["word"]) for w in words]

        flagged = None
        final = None
        if rec["sentence"] == "woodchuck":
            chuck_idx = [i for i, w in enumerate(norm_words) if w == "chuck"]
            if chuck_idx:
                fw = words[chuck_idx[0]]
                flagged = dict(word="chuck", occurrence="first", start=fw["start"], end=fw["end"],
                               duration=fw["end"] - fw["start"])
        else:
            morning_idx = [i for i, w in enumerate(norm_words) if w == "morning"]
            if morning_idx:
                fw = words[morning_idx[0]]
                flagged = dict(word="morning", occurrence="only/final", start=fw["start"], end=fw["end"],
                                duration=fw["end"] - fw["start"])
        if words:
            lw = words[-1]
            final = dict(word=lw["word"], start=lw["start"], end=lw["end"], duration=lw["end"] - lw["start"])

        row = dict(
            sentence=rec["sentence"], seed=rec["seed"], speed=rec["speed"],
            is_flagged_seed=rec["is_flagged_seed"],
            clip_duration_sec=rec["dur_after_speed_sec"],
            transcript=[norm_words],
            flagged_word=flagged, final_word=final,
        )
        out.append(row)
        fw_dur = flagged["duration"] if flagged else None
        lw_dur = final["duration"] if final else None
        print(f"{rec['sentence']:10s} seed={rec['seed']} speed={rec['speed']:.2f} "
              f"clip_dur={rec['dur_after_speed_sec']:.3f}s flagged_word_dur={fw_dur} final_word_dur={lw_dur}")

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
