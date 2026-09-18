"""Phase 2a -- attacking sibilant DURATION instead of (or alongside) energy.

Bench 12 (`phase2a_deess_fix_bench.py`, see docs/LISTENING_BENCHES.md) failed:
a listener judged the untreated originals, the latent-attenuated clips, and
the 4-11 kHz band-split de-esser at factor=0.6 ALL "yes"/"maybe" for an
over-driven 's', and only the unmodified clean-preset controls came back
clean. The de-esser was independently confirmed to have done what it
claimed (measured -4.29 dB in-band against -4.44 dB intended, over 11% of
the clip -- see `results/phase2a/deesser_baseline.json`), so a correct
energy cut did not relieve the perception. The property that separates
flagged from clean ON THE SAME TEXT is fricative DURATION (0.057-0.064 s
flagged vs. 0.031-0.048 s clean, softer onsets -- docs/GLITCH_MITIGATION.md,
`bench7_pool_reproduction` in `results/phase2a/sibilance_latent.json`).

FIRST ATTEMPT AND ITS BUG (corrected here). The first build of this script
"shortened" each sibilant run by naive resampling (linear interpolation onto
fewer samples). That is a playback-speed change, not a pure duration change:
it pitch-shifts the segment up and applies an interpolation low-pass, and
direct measurement showed the resulting in-band spectral change was often
LARGER than the deliberate attenuation conditions and inconsistent in sign
between the two clips (library: -11..-22 dB "for free"; seashells: +0.4..+0.8
dB "for free"). A confound of that size cannot test the duration hypothesis,
so that attempt was not published and is corrected below. Factor 0.5 was
also dropped entirely: it broke real words on the seashells sentence
("sells seashells" transcribed as "tells details" by faster-whisper) --
useless for judging harshness if the clip is not intelligible.

THE FIX: pitch-preserving time-scale modification (`librosa.effects.
time_stretch`, a phase vocoder) instead of resampling, applied only to each
isolated sibilant segment, spliced back with the same short crossfade as
before. Every condition is then VERIFIED, not assumed: this script reports,
per sibilant run, the spectral centroid before/after the stretch and the
in-band (4-11 kHz) dB change measured directly on the isolated segment
(before splicing), with an explicit `spectrally_neutral` flag
(|centroid shift| <= 150 Hz AND |dB change| <= 1 dB where no deliberate
attenuation was applied). If a run fails that check the JSON says so
plainly rather than treating the stretch as clean.

This script builds and measures three interventions on the same two flagged
clips and the SAME sibilant-window detection reused verbatim from
`phase2a_deesser_baseline.py` / `phase2a_sibilance.py` (band_split,
envelope_from_mask, sib_mask_from_runs, hf_ratio machinery), then builds the
follow-up bench:

  1. shorten_only      -- pitch-preserving time-stretch of each detected
                           sibilant run, factor in {0.8, 0.7} (0.5 dropped,
                           see above). The clip gets shorter; nothing padded.
  2. shorten_and_atten -- the same time-stretch at factor 0.7 ONLY, PLUS the
                           existing -4.4 dB (factor=0.6) band cut applied to
                           the (now-shorter, re-positioned) sibilant regions.
  3. atten_harder       -- the band-split de-esser alone, no shortening, at
                           factor=0.3 (~-10.5 dB), to rule out "4.4 dB was
                           just too gentle" before concluding energy is the
                           wrong target. Unchanged from the first build --
                           this condition never depended on the resampling
                           bug and its numbers (measured -9.97 / -8.64 dB
                           against -10.46 dB intended) already stood.

Usage (from py/):
    python3 benches/phase2a_sibilance_duration_bench.py

Writes:
  results/phase2a/sibilance_duration.json         -- numeric findings
  results/listening_sets/phase2a_sibilance_duration/*.wav + manifest.json
  results/benches/phase2a_sibilance_duration.html -- NOT published; local only
"""
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase2a_deesser_baseline import band_split, envelope_from_mask, BAND_LO_HZ, BAND_HI_HZ, CROSSFADE_S  # noqa: E402
from phase2a_sibilance import transcribe_words  # noqa: E402
from phase2b_speaker_similarity import level_match  # noqa: E402
from bench_common import PLAYER_PAUSE_SCRIPT, b64  # noqa: E402

FRAME_SPAN_JSON = "results/phase2a/latent_frame_span.json"
CLEAN_DIR = "results/phase2a/frame_span_audio"
AUDIO_DIR = "results/phase2a/sibilance_duration_audio"
LISTEN_DIR = "results/listening_sets/phase2a_sibilance_duration"
OUT_HTML = "results/benches/phase2a_sibilance_duration.html"
FINDINGS_JSON = "results/phase2a/sibilance_duration.json"

SHORTEN_FACTORS = [0.8, 0.7]      # factor 0.5 dropped -- broke real words (see docstring)
SHORTEN_ATTEN_FACTOR = 0.7        # the single factor carried into shorten+atten
ATTEN_FACTOR_MATCHED = 0.6        # -4.4 dB, the same factor bench 12 used and that failed
ATTEN_FACTOR_HARD = 0.3           # ~-10.5 dB, real-de-esser territory; unchanged from first build
CROSSFADE_HERE_S = 0.005          # 5 ms, same as the existing de-esser

CENTROID_NEUTRAL_HZ = 150.0       # verification threshold on the stretch itself
DB_NEUTRAL = 1.0                  # dB, verification threshold where no deliberate atten is applied

SHUFFLE_SEED = 20270918 + 13

FAMILIES = [
    dict(name="library", flagged="F3_library_flagged", clean="M4_library_clean",
         text="The library closes early on Thursday, so bring your books back."),
    dict(name="seashells", flagged="F5_seashells_flagged", clean="F2_seashells_clean",
         text="She sells seashells by the sea shore every summer morning."),
]

CLOSING_READ = (
    "A negative on attenuate-harder (factor 0.3, ~-10.5 dB) would strengthen the duration "
    "hypothesis rather than fix anything -- it changes nothing about fricative duration, the "
    "property that actually separates flagged from clean on fixed text, so if it still reads "
    "'yes'/'maybe' that rules out 'the energy cut was simply too gentle' and leaves duration as "
    "the remaining candidate. Time-stretched fricatives may introduce their own artifact (a "
    "warble or smeared onset from the phase vocoder) even when the spectral-neutrality check "
    "below passes on the isolated segment -- neutrality of centroid and in-band energy does not "
    "guarantee the transient/onset character survives the stretch, and onset sharpness is part "
    "of what distinguished flagged from clean in the first place. Shorten-and-attenuate inherits "
    "both risks at once. None of these three should be assumed to be a clean fix; the bench "
    "exists because the previous, competently-executed energy cut was not."
)


def load_wav(path, dtype="float64"):
    y, sr = sf.read(path, dtype=dtype)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def norm_text(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).split()


def segment_centroid_and_band_rms(seg, sr, lo=BAND_LO_HZ, hi=BAND_HI_HZ):
    """Single-window spectral centroid and 4-11 kHz-band RMS of a short
    segment, computed directly (Hann-windowed FFT), independent of any
    frame-grid alignment issues that a full-clip STFT would have once the
    clip has been spliced to a different length."""
    seg = np.asarray(seg, dtype=np.float64)
    n = len(seg)
    if n < 8:
        return None, None
    window = np.hanning(n)
    X = np.fft.rfft(seg * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(X)
    total = mag.sum()
    centroid = float((mag * freqs).sum() / total) if total > 0 else None
    band_mask = (freqs >= lo) & (freqs <= hi)
    Xband = np.where(band_mask, X, 0.0)
    band_time = np.fft.irfft(Xband, n)
    band_rms = float(np.sqrt((band_time ** 2).mean()))
    return centroid, band_rms


def time_stretch_segment(core, factor, sr):
    """Pitch-preserving time compression of a short segment via librosa's
    phase-vocoder time_stretch. `factor` < 1 shortens the segment (factor=0.7
    -> 70% of the original duration), so the vocoder `rate` (speed-up) is
    1/factor. n_fft/hop are shrunk to fit segments as short as ~6 ms."""
    rate = 1.0 / factor
    n = len(core)
    n_fft = 512
    while n_fft > n and n_fft > 32:
        n_fft //= 2
    n_fft = max(32, n_fft)
    hop_length = max(1, n_fft // 4)
    y32 = core.astype(np.float32)
    try:
        stretched = librosa.effects.time_stretch(y32, rate=rate, n_fft=n_fft, hop_length=hop_length)
        degraded = False
    except Exception as e:
        n_new = max(1, int(round(n * factor)))
        stretched = np.interp(np.linspace(0.0, 1.0, n_new), np.linspace(0.0, 1.0, n), core).astype(np.float32)
        degraded = f"time_stretch failed ({e}); fell back to naive resample for this run only"
    return stretched.astype(np.float64), degraded


# ---------------------------------------------------------------------------
# Time compression of each detected sibilant run via pitch-preserving
# time-stretch, spliced back with a short equal-length overlap-add crossfade
# against the untouched audio on both sides.
# ---------------------------------------------------------------------------
def compress_clip(y, sr, runs, factor, crossfade_s=CROSSFADE_HERE_S):
    fade_n_target = max(1, int(round(crossfade_s * sr)))
    runs_sorted = sorted(runs, key=lambda r: r["sample_lo"])
    n = len(y)
    chunks = []
    cursor = 0
    out_len = 0
    region_reports = []
    new_region_spans = []  # (out_lo, out_hi) in the OUTPUT array, per run

    for r in runs_sorted:
        lo, hi = r["sample_lo"], min(r["sample_hi"], n)
        if hi <= lo or lo < cursor:
            continue
        pre = y[cursor:lo]
        core = y[lo:hi]
        n_orig = hi - lo
        compressed, degraded = time_stretch_segment(core, factor, sr)
        n_new = len(compressed)

        cen_before, rms_before = segment_centroid_and_band_rms(core, sr)
        cen_after, rms_after = segment_centroid_and_band_rms(compressed, sr)
        centroid_shift = (cen_after - cen_before) if (cen_before is not None and cen_after is not None) else None
        db_change_region = None
        if rms_before is not None and rms_before > 0 and rms_after is not None:
            db_change_region = 20.0 * np.log10(max(rms_after, 1e-12) / rms_before)
        spectrally_neutral = None
        if centroid_shift is not None and db_change_region is not None:
            spectrally_neutral = (abs(centroid_shift) <= CENTROID_NEUTRAL_HZ) and (abs(db_change_region) <= DB_NEUTRAL)

        fN = min(fade_n_target, len(pre), n_new // 3 if n_new >= 3 else 0)
        gN = min(fade_n_target, max(0, n - hi), (n_new - fN) // 2 if n_new - fN >= 2 else 0)

        region_out_lo = out_len + max(0, len(pre) - fN)

        if fN > 0:
            t = np.linspace(0.0, 1.0, fN)
            head = pre[-fN:] * (1.0 - t) + compressed[:fN] * t
            chunks.append(pre[:-fN]); out_len += len(pre) - fN
            chunks.append(head); out_len += fN
        else:
            chunks.append(pre); out_len += len(pre)

        mid_hi = n_new - gN
        mid = compressed[fN:mid_hi] if mid_hi > fN else compressed[fN:fN]
        chunks.append(mid); out_len += len(mid)

        if gN > 0:
            t2 = np.linspace(0.0, 1.0, gN)
            tail = compressed[mid_hi:] * (1.0 - t2) + y[hi:hi + gN] * t2
            chunks.append(tail); out_len += gN
            cursor = hi + gN
        else:
            cursor = hi

        region_out_hi = out_len
        new_region_spans.append((region_out_lo, region_out_hi))
        region_reports.append(dict(
            word=r["word"], orig_duration_s=n_orig / sr, new_duration_s=n_new / sr,
            centroid_before_hz=cen_before, centroid_after_hz=cen_after,
            centroid_shift_hz=centroid_shift,
            band_db_change_region=db_change_region,
            spectrally_neutral=spectrally_neutral,
            degraded_to_resample=degraded,
        ))

    chunks.append(y[cursor:]); out_len += len(y) - cursor
    out = np.concatenate(chunks)
    assert len(out) == out_len
    return out, region_reports, new_region_spans


def band_rms_db_change(y_before, y_after_full, mask_before, mask_after, sr):
    """Whole-clip RMS of the 4-11 kHz band component inside the touched
    region, before vs. after, in dB -- used for the deliberate-attenuation
    conditions (matches phase2a_deesser_baseline.py's own measurement)."""
    band_before, _ = band_split(y_before, sr, BAND_LO_HZ, BAND_HI_HZ)
    band_after, _ = band_split(y_after_full, sr, BAND_LO_HZ, BAND_HI_HZ)
    seg_before = band_before[mask_before]
    seg_after = band_after[mask_after]
    if seg_before.size == 0 or seg_after.size == 0:
        return None
    rms_before = float(np.sqrt((seg_before ** 2).mean()))
    rms_after = float(np.sqrt((seg_after ** 2).mean()))
    if rms_before <= 0:
        return None
    return 20.0 * np.log10(max(rms_after, 1e-12) / rms_before)


def mask_from_spans(spans, n):
    m = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        m[lo:min(hi, n)] = True
    return m


def mask_from_runs(runs, n):
    m = np.zeros(n, dtype=bool)
    for r in runs:
        lo, hi = r["sample_lo"], min(r["sample_hi"], n)
        if hi > lo:
            m[lo:hi] = True
    return m


def transcribe_and_check(model, wav_path, expected_words):
    words = transcribe_words(model, wav_path)
    got = [w["word"] for w in words]
    got_norm = norm_text(" ".join(got))
    exp_norm = norm_text(expected_words)
    return dict(transcript=" ".join(got), n_words=len(words),
                matches_expected=(got_norm == exp_norm), expected=" ".join(exp_norm))


def summarize_regions(region_reports):
    shifts = [r["centroid_shift_hz"] for r in region_reports if r["centroid_shift_hz"] is not None]
    dbs = [r["band_db_change_region"] for r in region_reports if r["band_db_change_region"] is not None]
    neutral_flags = [r["spectrally_neutral"] for r in region_reports if r["spectrally_neutral"] is not None]
    return dict(
        n_regions=len(region_reports),
        mean_centroid_shift_hz=float(np.mean(shifts)) if shifts else None,
        max_abs_centroid_shift_hz=float(np.max(np.abs(shifts))) if shifts else None,
        mean_band_db_change_region=float(np.mean(dbs)) if dbs else None,
        max_abs_band_db_change_region=float(np.max(np.abs(dbs))) if dbs else None,
        n_spectrally_neutral=int(sum(1 for f in neutral_flags if f)),
        n_regions_checked=len(neutral_flags),
        all_spectrally_neutral=bool(all(neutral_flags)) if neutral_flags else None,
        any_degraded_to_resample=any(r["degraded_to_resample"] for r in region_reports),
    )


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    with open(FRAME_SPAN_JSON) as fh:
        span = json.load(fh)
    runs_by_name = {rec["name"]: rec["runs"] for rec in span["task1_frame_span"]}

    print("Loading faster-whisper tiny.en (local, int8, cpu) ...", flush=True)
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        "Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8", local_files_only=True,
    )

    findings = {"experiment": "phase2a_sibilance_duration",
                "method_note": "corrected: pitch-preserving librosa.effects.time_stretch, "
                                "not naive resampling (see module docstring for the retracted "
                                "first attempt)",
                "shorten_factors": SHORTEN_FACTORS,
                "shorten_atten_factor": SHORTEN_ATTEN_FACTOR,
                "atten_factor_matched": ATTEN_FACTOR_MATCHED,
                "atten_factor_hard": ATTEN_FACTOR_HARD,
                "centroid_neutral_threshold_hz": CENTROID_NEUTRAL_HZ,
                "db_neutral_threshold": DB_NEUTRAL,
                "closing_read": CLOSING_READ,
                "families": []}
    blind = []

    for fam in FAMILIES:
        flagged_name = fam["flagged"]
        runs = runs_by_name[flagged_name]
        src_path = os.path.join(CLEAN_DIR, f"{flagged_name}.wav")
        y0, sr = load_wav(src_path)
        n0 = len(y0)
        mask0 = mask_from_runs(runs, n0)
        orig_fric_durations = [(r["sample_hi"] - r["sample_lo"]) / sr for r in runs]

        fam_findings = dict(name=fam["name"], text=fam["text"],
                             orig_dur_s=n0 / sr,
                             orig_fricative_durations_s=orig_fric_durations,
                             conditions=[])
        ref_rms = float(np.sqrt((y0 ** 2).mean()))

        def add_blind(kind, y_out, sr_out, extra):
            leveled, gain_db = level_match(y_out.astype(np.float32), ref_rms)
            stem = f"{fam['name']}_{kind}"
            wav_path = os.path.join(LISTEN_DIR, f"{stem}.wav")
            sf.write(wav_path, leveled, sr_out, subtype="PCM_16")
            rec = dict(family=fam["name"], kind=kind, wav_path=wav_path,
                       levelmatch_gain_db=round(gain_db, 2),
                       is_control=(kind == "clean_control"), **extra)
            blind.append(rec)
            return rec

        # --- untreated original ---
        sf.write(os.path.join(AUDIO_DIR, f"{fam['name']}_original.wav"),
                  y0.astype(np.float32), sr, subtype="PCM_16")
        orig_check = transcribe_and_check(whisper_model, src_path, fam["text"])
        add_blind("original_flagged", y0, sr, dict(
            note="untouched flagged render", dur_s=n0 / sr,
            transcript=orig_check["transcript"], matches_expected=orig_check["matches_expected"],
        ))

        # --- 1. shorten only (pitch-preserving), factors 0.8 and 0.7 ---
        shorten_results = {}
        for factor in SHORTEN_FACTORS:
            y1, region_reports, spans1 = compress_clip(y0, sr, runs, factor)
            shorten_results[factor] = (y1, region_reports, spans1)
            out_path = os.path.join(AUDIO_DIR, f"{fam['name']}_shorten{factor}.wav")
            sf.write(out_path, y1.astype(np.float32), sr, subtype="PCM_16")
            chk = transcribe_and_check(whisper_model, out_path, fam["text"])

            reg_summary = summarize_regions(region_reports)
            new_fric_durations = [rr["new_duration_s"] for rr in region_reports]
            cond = dict(
                kind=f"shorten_only_{factor}", factor=factor,
                new_dur_s=len(y1) / sr, dur_change_s=len(y1) / sr - n0 / sr,
                orig_fricative_durations_s=[rr["orig_duration_s"] for rr in region_reports],
                new_fricative_durations_s=new_fric_durations,
                mean_orig_fricative_s=float(np.mean([rr["orig_duration_s"] for rr in region_reports])),
                mean_new_fricative_s=float(np.mean(new_fric_durations)),
                region_diagnostics=region_reports,
                region_summary=reg_summary,
                transcript=chk["transcript"], matches_expected=chk["matches_expected"],
            )
            if reg_summary["all_spectrally_neutral"] is False:
                cond["warning"] = (
                    f"NOT spectrally neutral: max |centroid shift|="
                    f"{reg_summary['max_abs_centroid_shift_hz']:.1f} Hz, "
                    f"max |in-band dB change|={reg_summary['max_abs_band_db_change_region']:.2f} dB "
                    f"on at least one sibilant run (thresholds {CENTROID_NEUTRAL_HZ} Hz / {DB_NEUTRAL} dB)."
                )
            fam_findings["conditions"].append(cond)
            add_blind(f"shorten_only_{factor}", y1, sr, dict(
                factor=factor, method="shorten_only",
                dur_s=len(y1) / sr, dur_change_s=round(len(y1) / sr - n0 / sr, 4),
                spectrally_neutral=reg_summary["all_spectrally_neutral"],
            ))

        # --- 2. shorten + attenuate, factor 0.7 only, matched -4.4 dB band cut ---
        y2, region_reports2, spans2 = shorten_results[SHORTEN_ATTEN_FACTOR]
        mask2 = mask_from_spans(spans2, len(y2))
        band2, resid2 = band_split(y2, sr, BAND_LO_HZ, BAND_HI_HZ)
        env2 = envelope_from_mask(mask2, sr, ATTEN_FACTOR_MATCHED, CROSSFADE_S)
        y2_att = resid2 + band2 * env2
        out_path = os.path.join(AUDIO_DIR, f"{fam['name']}_shorten{SHORTEN_ATTEN_FACTOR}_atten{ATTEN_FACTOR_MATCHED}.wav")
        sf.write(out_path, y2_att.astype(np.float32), sr, subtype="PCM_16")
        chk2 = transcribe_and_check(whisper_model, out_path, fam["text"])
        db_change2 = band_rms_db_change(y0, y2_att, mask0, mask2, sr)

        new_fric_durations2 = [rr["new_duration_s"] for rr in region_reports2]
        fam_findings["conditions"].append(dict(
            kind=f"shorten_{SHORTEN_ATTEN_FACTOR}_atten_{ATTEN_FACTOR_MATCHED}",
            shorten_factor=SHORTEN_ATTEN_FACTOR, atten_factor=ATTEN_FACTOR_MATCHED,
            new_dur_s=len(y2_att) / sr, dur_change_s=len(y2_att) / sr - n0 / sr,
            mean_orig_fricative_s=float(np.mean([rr["orig_duration_s"] for rr in region_reports2])),
            mean_new_fricative_s=float(np.mean(new_fric_durations2)),
            band_db_change_in_touched_region_whole_clip=db_change2,
            intended_db=20 * np.log10(ATTEN_FACTOR_MATCHED),
            note="in-band dB here is measured on the WHOLE spliced+attenuated clip against the "
                 "original (region_summary of the underlying shorten_only_0.7 condition above "
                 "already isolates the stretch's own spectral effect)",
            transcript=chk2["transcript"], matches_expected=chk2["matches_expected"],
        ))
        add_blind(f"shorten_{SHORTEN_ATTEN_FACTOR}_atten_{ATTEN_FACTOR_MATCHED}", y2_att, sr, dict(
            shorten_factor=SHORTEN_ATTEN_FACTOR, atten_factor=ATTEN_FACTOR_MATCHED, method="shorten_and_atten",
            dur_s=len(y2_att) / sr, dur_change_s=round(len(y2_att) / sr - n0 / sr, 4),
        ))

        # --- 3. attenuate harder (no shortening), unchanged from first build ---
        band3, resid3 = band_split(y0, sr, BAND_LO_HZ, BAND_HI_HZ)
        env3 = envelope_from_mask(mask0, sr, ATTEN_FACTOR_HARD, CROSSFADE_S)
        y3 = resid3 + band3 * env3
        out_path = os.path.join(AUDIO_DIR, f"{fam['name']}_atten{ATTEN_FACTOR_HARD}.wav")
        sf.write(out_path, y3.astype(np.float32), sr, subtype="PCM_16")
        chk3 = transcribe_and_check(whisper_model, out_path, fam["text"])
        db_change3 = band_rms_db_change(y0, y3, mask0, mask0, sr)
        fam_findings["conditions"].append(dict(
            kind=f"atten_hard_{ATTEN_FACTOR_HARD}", atten_factor=ATTEN_FACTOR_HARD,
            new_dur_s=len(y3) / sr, dur_change_s=0.0,
            band_db_change_in_touched_region=db_change3,
            intended_db=20 * np.log10(ATTEN_FACTOR_HARD),
            transcript=chk3["transcript"], matches_expected=chk3["matches_expected"],
        ))
        add_blind(f"atten_hard_{ATTEN_FACTOR_HARD}", y3, sr, dict(
            atten_factor=ATTEN_FACTOR_HARD, method="atten_harder",
            dur_s=len(y3) / sr, dur_change_s=0.0,
        ))

        # --- hidden clean-preset control ---
        clean_path = os.path.join(CLEAN_DIR, f"{fam['clean']}.wav")
        y_clean, sr_c = load_wav(clean_path)
        add_blind("clean_control", y_clean, sr_c, dict(note=f"stock {fam['clean']}, same sentence, never flagged"))

        findings["families"].append(fam_findings)
        print(f"\n=== {fam['name']} ===")
        for c in fam_findings["conditions"]:
            printable = {k: v for k, v in c.items()
                         if k not in ("transcript", "region_diagnostics", "orig_fricative_durations_s", "new_fricative_durations_s")}
            print(f"  {c['kind']}: {json.dumps(printable, default=str)}")
            print(f"      transcript: {c.get('transcript')!r} matches_expected={c.get('matches_expected')}")
            if c.get("warning"):
                print(f"      WARNING: {c['warning']}")

    order = list(range(len(blind)))
    import random
    random.Random(SHUFFLE_SEED).shuffle(order)
    for display_i, src_i in enumerate(order, start=1):
        blind[src_i]["id"] = f"clip{display_i:02d}"
        blind[src_i]["label"] = f"Clip {display_i:02d}"
    blind.sort(key=lambda c: c["id"])

    manifest = {
        "experiment": "phase2a_sibilance_duration_bench",
        "shuffle_seed": SHUFFLE_SEED,
        "families": [f["name"] for f in FAMILIES],
        "clips": [{k: v for k, v in c.items() if k != "wav_path"} for c in blind],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(FINDINGS_JSON, "w") as fh:
        json.dump(findings, fh, indent=2, default=str)
    print(f"\nWrote {LISTEN_DIR}/manifest.json and {FINDINGS_JSON}")

    build_html(blind)


def build_html(blind):
    def clip_card(c):
        bits = [f"family <code>{c['family']}</code>", f"kind <code>{c['kind']}</code>",
                f"level-match gain {c['levelmatch_gain_db']:+.2f} dB"]
        if c.get("method"):
            bits.append(f"method <code>{c['method']}</code>")
        if c.get("factor") is not None:
            bits.append(f"factor <code>{c['factor']}</code>")
        if c.get("shorten_factor") is not None:
            bits.append(f"shorten factor <code>{c['shorten_factor']}</code>")
        if c.get("atten_factor") is not None:
            bits.append(f"atten factor <code>{c['atten_factor']}</code>")
        if c.get("dur_s") is not None:
            bits.append(f"clip duration <code>{c['dur_s']:.3f}s</code>")
        if c.get("dur_change_s") is not None:
            bits.append(f"duration change <code>{c['dur_change_s']:+.3f}s</code>")
        if c.get("spectrally_neutral") is not None:
            bits.append(f"stretch spectrally neutral (per-region check): <code>{c['spectrally_neutral']}</code>")
        if c.get("matches_expected") is not None:
            bits.append(f"transcript matches expected text: <code>{c['matches_expected']}</code>")
        if c.get("note"):
            bits.append(c["note"])
        reveal_body = "<p>" + " &middot; ".join(bits) + "</p>"
        return f"""      <article class="clip" data-clip="{c['id']}">
        <div class="clip-head"><span class="clip-label">{c['label']}</span></div>
        <audio controls preload="metadata" src="data:audio/wav;base64,{b64(c['wav_path'])}"></audio>
        <div class="verdict" data-clip="{c['id']}">
          <div class="q-block">
            <p class="q-label">Q1 &middot; Do you hear an over-driven or harsh 's' in this clip? <span class="tick" data-tick data-clip="{c['id']}" data-field="q1"></span></p>
            <div class="choices" data-field="q1">
              <label><input type="radio" name="{c['id']}_q1" value="yes_clear" disabled> Yes, clearly</label>
              <label><input type="radio" name="{c['id']}_q1" value="maybe" disabled> Maybe</label>
              <label><input type="radio" name="{c['id']}_q1" value="no" disabled> No</label>
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">Q2 &middot; Does anything else sound wrong or degraded? <span class="tick" data-tick data-clip="{c['id']}" data-field="q2"></span></p>
            <textarea data-field="q2" data-clip="{c['id']}" disabled rows="2"
              placeholder="free text -- clipped, lispy, rushed, warbly, garbled, wrong word, anything"></textarea>
          </div>
        </div>
        <details class="reveal">
          <summary>Reveal (after you answer)</summary>
          {reveal_body}
        </details>
      </article>"""

    clip_cards = "\n".join(clip_card(c) for c in blind)
    triples_for_js = [{"id": c["id"], "is_control": c["is_control"]} for c in blind]

    DB_SCRIPT = r"""(function () {
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  var verdictEls = $all('.verdict');
  var state = {}; var pending = {}; var timers = {}; var db = null;
  verdictEls.forEach(function (v) { var tid = v.getAttribute('data-clip'); state[tid] = {}; pending[tid] = {}; });

  function setTick(tid, field, mode) {
    var el = document.querySelector('[data-tick][data-clip="' + tid + '"][data-field="' + field + '"]');
    if (!el) return;
    el.className = 'tick' + (mode ? ' ' + mode : '');
    el.textContent = mode === 'saving' ? 'saving…' : mode === 'saved' ? 'saved ✓' : mode === 'err' ? 'not saved' : '';
  }
  function clipMeta(tid) { for (var i = 0; i < CLIPS.length; i++) { if (CLIPS[i].id === tid) return CLIPS[i]; } return null; }
  function save(tid) {
    if (!db) return;
    var fields = Object.keys(pending[tid]);
    if (!fields.length) return;
    pending[tid] = {};
    var meta = clipMeta(tid);
    var body = Object.assign({}, state[tid], { updated_at: new Date().toISOString() });
    if (meta) { body.is_control = meta.is_control; }
    db.doc('verdicts/' + tid).set(body).then(function () {
      fields.forEach(function (f) { setTick(tid, f, 'saved'); });
      updateProgress();
    }).catch(function () { fields.forEach(function (f) { setTick(tid, f, 'err'); }); });
  }
  function onFieldChange(tid, field, value) {
    state[tid][field] = value; pending[tid][field] = true; setTick(tid, field, 'saving');
    clearTimeout(timers[tid]); timers[tid] = setTimeout(function () { save(tid); }, 600);
  }
  function wireVerdict(v) {
    var tid = v.getAttribute('data-clip');
    $all('.choices', v).forEach(function (group) {
      var field = group.getAttribute('data-field');
      $all('input[type=radio]', group).forEach(function (input) {
        input.addEventListener('change', function () { if (input.checked) onFieldChange(tid, field, input.value); });
      });
    });
    $all('textarea', v).forEach(function (ta) {
      var field = ta.getAttribute('data-field');
      ta.addEventListener('input', function () { onFieldChange(tid, field, ta.value); });
    });
  }
  verdictEls.forEach(wireVerdict);
  function updateProgress() {
    var el = document.getElementById('progress-line');
    if (!el) return;
    if (!db) { el.textContent = 'Responses cannot be saved in this view.'; el.className = 'progress nodb'; return; }
    var n = CLIPS.filter(function (c) { return state[c.id] && state[c.id].q1 && state[c.id].q2; }).length;
    el.textContent = n + ' of ' + CLIPS.length + ' clips fully answered';
    el.className = 'progress ready';
  }
  function enableInputs() { $all('.verdict input, .verdict textarea').forEach(function (el) { el.disabled = false; }); }
  function hydrate() {
    var ids = CLIPS.map(function (c) { return c.id; }).concat(['_overall']);
    return Promise.all(ids.map(function (tid) {
      return db.doc('verdicts/' + tid).get().then(function (snap) {
        if (!snap.exists) return;
        var data = snap.data() || {};
        state[tid] = data;
        var v = document.querySelector('.verdict[data-clip="' + tid + '"]');
        if (!v) return;
        Object.keys(data).forEach(function (field) {
          if (field === 'updated_at' || field === 'is_control') return;
          var val = data[field];
          if (val === undefined || val === null || val === '') return;
          var radio = v.querySelector('.choices[data-field="' + field + '"] input[value="' + val + '"]');
          if (radio) { radio.checked = true; setTick(tid, field, 'saved'); return; }
          var ta = v.querySelector('textarea[data-field="' + field + '"]');
          if (ta) { ta.value = val; setTick(tid, field, 'saved'); }
        });
      }).catch(function () {});
    }));
  }
  (async function () {
    var claude = window.claude;
    db = (claude && typeof claude.use === 'function') ? await claude.use('db') : null;
    if (!db) { updateProgress(); return; }
    await hydrate(); enableInputs(); updateProgress();
  })();
})();"""

    HTML = """<title>Sibilance Duration Fix Bench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>
  :root {
    color-scheme: light;
    --ground:#F2F5F6; --surface:#FFFFFF; --sunk:#E4EAEC;
    --ink:#0D1417; --ink-2:#46565E; --ink-3:#7A8990; --rule:#D2DADD;
    --accent:#0E7C73; --accent-soft:#D5E9E6; --flag:#9A5B12;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
    --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
    --serif:"Newsreader",Georgia,serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --ground:#0B1013; --surface:#141B1F; --sunk:#090D10;
      --ink:#E5EDEF; --ink-2:#9CADB5; --ink-3:#6A7B84; --rule:#223037;
      --accent:#3EB9AD; --accent-soft:#0E312E; --flag:#D9A05B;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ground:#0B1013; --surface:#141B1F; --sunk:#090D10;
    --ink:#E5EDEF; --ink-2:#9CADB5; --ink-3:#6A7B84; --rule:#223037;
    --accent:#3EB9AD; --accent-soft:#0E312E; --flag:#D9A05B;
  }
  body { background:var(--ground); color:var(--ink); font-family:var(--serif); font-size:16px; line-height:1.6; padding:40px 22px 72px; }
  .wrap { max-width:860px; margin:0 auto; display:flex; flex-direction:column; gap:40px; }
  .eyebrow { font-family:var(--mono); font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); }
  h1 { font-family:var(--sans); font-weight:600; font-size:clamp(28px,4.4vw,40px); letter-spacing:-.02em; line-height:1.12; text-wrap:balance; margin:10px 0 14px; }
  .deck { color:var(--ink-2); max-width:64ch; margin:0; }
  h2 { font-family:var(--sans); font-weight:600; font-size:15px; margin:0 0 4px; }
  .sect-note { font-size:14.5px; color:var(--ink-3); margin:0 0 18px; max-width:64ch; }
  .task { background:var(--accent-soft); border:1px solid var(--accent); border-radius:3px; padding:18px 20px; }
  .task .tag { font-family:var(--mono); font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent); margin:0 0 9px; font-weight:500; }
  .task ol, .task ul { margin:0; padding-left:20px; color:var(--ink); }
  .task li { margin-bottom:7px; }
  .task li:last-child { margin-bottom:0; }
  .task .why { margin:12px 0 0; font-size:14px; color:var(--ink-2); }
  .warn { background:var(--sunk); border:1px solid var(--flag); border-radius:3px; padding:14px 18px; font-size:14px; color:var(--ink-2); }
  .warn strong { color:var(--flag); }
  .clips { display:flex; flex-direction:column; gap:14px; }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:15px 18px 17px; display:flex; flex-direction:column; gap:11px; }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:14px; }
  audio { width:100%; height:36px; display:block; }
  .reveal { font-size:12.5px; color:var(--ink-3); }
  .reveal summary { cursor:pointer; font-family:var(--mono); font-size:11px; letter-spacing:.04em; color:var(--ink-3); }
  .reveal summary:hover { color:var(--accent); }
  .reveal p { margin:6px 0 0; }
  .reveal code { font-family:var(--mono); font-size:.92em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:68ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .progress { font-family:var(--mono); font-size:12px; color:var(--ink-3); margin:0; }
  .progress.ready { color:var(--accent); }
  .progress.nodb { color:var(--flag); }
  .verdict { margin-top:0; padding-top:0; display:flex; flex-direction:column; gap:12px; }
  .q-block { display:flex; flex-direction:column; gap:6px; }
  .q-label { font-family:var(--sans); font-weight:600; font-size:12.5px; color:var(--ink); margin:0; display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .choices { display:flex; flex-direction:column; gap:5px; }
  .choices label { display:flex; align-items:flex-start; gap:7px; font-family:var(--sans); font-size:12.5px; color:var(--ink-2); cursor:pointer; }
  .choices input[type="radio"] { margin-top:2px; accent-color:var(--accent); flex-shrink:0; }
  textarea { width:100%; font-family:var(--sans); font-size:12.5px; color:var(--ink); background:var(--ground); border:1px solid var(--rule); border-radius:3px; padding:8px 10px; resize:vertical; box-sizing:border-box; }
  textarea::placeholder { color:var(--ink-3); }
  textarea:disabled, .choices input:disabled { opacity:.55; }
  .tick { font-family:var(--mono); font-size:10px; letter-spacing:.04em; color:var(--ink-3); font-weight:400; }
  .tick.saving { color:var(--ink-3); }
  .tick.saved { color:var(--accent); }
  .tick.err { color:var(--flag); }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2a &middot; sibilance, attacking duration (corrected)</div>
    <h1>Shorten the Fricative, Not Just Its Volume</h1>
    <p class="deck">The previous fix bench failed: untreated originals, the latent-domain
      attenuation, and a correctly-measured -4.4&nbsp;dB band-split de-esser all still read
      "yes"/"maybe" for over-driven 's'. Only unmodified clean-preset controls read clean. This
      set tests whether fricative DURATION -- the property that actually separates flagged from
      clean on fixed text -- is worth shortening, alone or combined with the same energy cut, and
      whether a much harder energy cut (real de-esser territory) does any better on its own. The
      shortening here uses a pitch-preserving time-stretch (not a resample); each stretch is
      verified for spectral neutrality per sibilant region, not assumed -- see the reveal panel
      on each shortened clip.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="warn">
      <strong>Heads up:</strong> the previous round of fixes did not work by ear. If several
      clips here also come back "yes" or "maybe" -- including a "no difference from the
      original" verdict -- that is a real, expected possible outcome, not a failure to notice.
      Please answer exactly what you hear, clip by clip.
    </div>
  </section>

  <section>
    <div class="task">
      <p class="tag">What we need from you, and what it decides</p>
      <ol>
        <li>For each clip: does it have an over-driven / harsh 's' (Q1)? Answer fresh, not by
          recalling which clip you think is which.</li>
        <li>Separately: does anything ELSE sound wrong or degraded (Q2, free text)? A
          pitch-preserving stretch can still sound warbly, smeared, or rushed at the onset --
          say so if you hear it, even if the harshness itself is gone.</li>
        <li>One clip per sentence family is a hidden, unmodified clean-preset control (different
          voice, never flagged) &mdash; if it gets flagged too, treat every "yes" on this page
          with more suspicion.</li>
      </ol>
      <p class="why">This decides whether duration, rather than band energy, is the property
        worth attacking -- and whether shortening buys anything a harder energy cut alone
        doesn't.</p>
    </div>
  </section>

  <section>
    <h2>The clips</h2>
    <p class="sect-note">2 sentence families &times; (untreated original, shorten-only at factor
      0.8 and 0.7, shorten+attenuate at factor 0.7, attenuate-harder, clean control) = 12 clips,
      blind order, level-matched per family so loudness cannot give away the condition.</p>
    <div class="clips">
__CLIP_CARDS__
    </div>
  </section>

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One free-text box across the whole set, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Which condition, if any, sounded like a real fix? Anything that surprised you"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. <code>results/phase2a/sibilance_duration.json</code> (this script's numeric
    findings, including per-region spectral-neutrality checks) and
    <code>results/phase2a/latent_frame_span.json</code> (sibilant-run detection, reused verbatim)
    supply the clips and windows. Clean controls are <code>M4_library_clean</code> and
    <code>F2_seashells_clean</code>, stock renders of the same sentences on never-flagged presets.
    Every clip is level-matched (RMS gain only) to its family's untreated-flagged-clip RMS before
    embedding; see <code>manifest.json</code> in
    <code>results/listening_sets/phase2a_sibilance_duration/</code> for the exact per-clip gain
    and the blind-id mapping.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
<script>
var CLIPS = __CLIPS_JSON__;
__DB_SCRIPT__
</script>
"""

    HTML = (HTML
            .replace("__CLIP_CARDS__", clip_cards)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(triples_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as fh:
        fh.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes (NOT published)")


if __name__ == "__main__":
    main()
