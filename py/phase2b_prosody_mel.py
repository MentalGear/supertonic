"""Phase 2b follow-up: time-frequency diff maps for the prosody-vs-timbre question.

The utterance-level battery that produced the recorded "most of style space is
nearly inaudible" collapsed both axes a prosodic change lives on. These clips
are frame-aligned by construction -- same text, same vocoder seed, `style_dp`
pinned, so every render is the same number of samples and the same phoneme
schedule -- which makes a direct signed log-mel difference against the
unperturbed base a valid instrument rather than a summary statistic.

The two hypotheses predict visibly different maps:

  * a prosodic / emphasis change is **localized in time** (a few words move) and
    **broadband in frequency** (the whole spectrum of those words moves together);
  * a timbre change is **spread in time** (every voiced frame moves) and
    **structured in frequency** (formant and tilt regions move, others do not).

So each diff is scored on exactly those two axes: a time-concentration index of
the diff energy, and the split of the diff into a broadband per-frame gain term
and a gain-free spectral-shape term. The full M1 -> F1 preset swap is rendered
into the same figures as the known-timbre reference.

Word boundaries come from faster-whisper word timestamps on the base clip,
snapped to the nearest local minimum of the smoothed energy envelope, then
reused unchanged for every clip sharing that base -- they are the same words at
the same times, so per-word energy is comparable clip to clip.

Usage (from py/):  python3 phase2b_prosody_mel.py
"""

import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from phase2b_prosody_analysis import (HOP, NFFT, OUT_DIR, PHASE0_DIR, PROSODY_DIR,
                                      RAY_DIR, SENTENCE, SR, Canvas, load16k, rms)

N_MELS, FMIN_MEL, FMAX_MEL = 80, 60.0, 7800.0
EPS_SHOW = [0.20, 0.80, 3.20]
WORDS = SENTENCE.rstrip(".").split()

SEQ = np.array([[12, 10, 30], [70, 25, 100], [160, 50, 90], [230, 120, 60], [252, 232, 150]], float)
DIV = np.array([[26, 68, 128], [120, 170, 210], [248, 248, 246], [232, 152, 108], [158, 34, 40]], float)


def cmap(x, anchors):
    """x in [0,1] -> RGB uint8 via piecewise-linear anchors."""
    a = np.asarray(anchors, float)
    pos = np.linspace(0, 1, len(a))
    x = np.clip(x, 0, 1)
    out = np.stack([np.interp(x, pos, a[:, c]) for c in range(3)], axis=-1)
    return out.astype(np.uint8)


def logmel(w):
    import librosa
    m = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=NFFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN_MEL, fmax=FMAX_MEL,
                                       center=True, power=2.0)
    return 10.0 * np.log10(m + 1e-10)


def envelope(M):
    return M.mean(0)


def env_lag_ms(a, b, max_ms=120.0):
    """Alignment in the envelope domain, with the lag-0 penalty alongside it.

    Raw-waveform cross-correlation is meaningless once F0 moves, but the energy
    contour is exactly what has to stay put. A shifted correlation peak at large
    eps can be genuine energy redistribution rather than a time shift, so the
    normalised correlation at lag 0 is reported next to the peak: if lag 0 is
    within a per cent or two of the best lag, the clips are aligned and the
    apparent lag is the effect being measured, not a misalignment.
    """
    n = min(a.size, b.size)
    x, y = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    k = int(max_ms * SR / 1000 / HOP)
    seg = x[k:n - k]
    xc = np.correlate(y, seg, mode="valid")
    win = np.lib.stride_tricks.sliding_window_view(y, seg.size)
    nrm = np.linalg.norm(win, axis=1) * np.linalg.norm(seg)
    r = xc / np.maximum(nrm, 1e-12)
    i = int(np.argmax(r))
    return float((i - k) * HOP * 1000.0 / SR), float(r[k]), float(r[i])


def word_bounds(path, base_env):
    """faster-whisper word timestamps, repaired and snapped to envelope minima.

    tiny.en places the boundaries of this sentence well except that it collapses
    "lazy" onto a near-zero-width split. Any word shorter than 60 ms is
    therefore pooled with its immediate neighbours and that span re-split in
    proportion to letter count. The word windows and the base level inside each
    are printed so the segmentation can be checked by hand.
    """
    from faster_whisper import WhisperModel
    m = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(path, language="en", beam_size=1, word_timestamps=True)
    words = [(w.word.strip(" .,"), float(w.start), float(w.end)) for s in segs for w in s.words]
    e = np.convolve(base_env, np.ones(3) / 3, mode="same")
    speech = np.flatnonzero(e > e.max() - 35.0)
    t0, t1 = speech[0] * HOP / SR, (speech[-1] + 1) * HOP / SR
    edges = np.array([t0] + [w[2] for w in words[:-1]] + [t1], float)
    edges = np.maximum.accumulate(np.clip(edges, t0, t1))

    # repair degenerate words by re-splitting their run on letter count
    minw = 0.060
    bad = [i for i in range(len(WORDS)) if edges[i + 1] - edges[i] < minw]
    for i in bad:
        lo_i, hi_i = max(i - 1, 0), min(i + 1, len(WORDS) - 1)
        lo, hi = edges[lo_i], edges[hi_i + 1]
        n = np.array([len(w) for w in WORDS[lo_i:hi_i + 1]], float)
        edges[lo_i:hi_i + 2] = lo + np.concatenate([[0.0], np.cumsum(n / n.sum() * (hi - lo))])

    # No envelope snapping: on this sentence it drags boundaries into stop
    # closures, leaving a word window sitting on the silence inside it. The
    # per-word base levels are printed so a degenerate window is visible.
    return [(WORDS[i], float(edges[i]), float(edges[i + 1])) for i in range(len(WORDS))]


def heat(cv, arr, x0, y0, x1, y1, vmin, vmax, anchors, div=False):
    """Draw a (bins, frames) array as an image block, low frequency at bottom."""
    h, w = int(y1 - y0), int(x1 - x0)
    a = np.asarray(arr, float)
    if div:
        z = 0.5 + 0.5 * np.clip(a / max(vmax, 1e-9), -1, 1)
    else:
        z = (a - vmin) / max(vmax - vmin, 1e-9)
    ri = np.clip((np.arange(h) / h * a.shape[0]).astype(int), 0, a.shape[0] - 1)[::-1]
    ci = np.clip((np.arange(w) / w * a.shape[1]).astype(int), 0, a.shape[1] - 1)
    cv.px[int(y0):int(y0) + h, int(x0):int(x0) + w] = cmap(z[np.ix_(ri, ci)], anchors)


def panel_fig(path, title, panels, bounds, dur_s, sym, note_lines, W=1180):
    """panels: list of (label, array, is_diff). One row each, shared time axis."""
    PH, GAP, T, L, R, B = 138, 36, 84, 96, 22, 96
    H = T + len(panels) * (PH + GAP) + B
    cv = Canvas(W, H)
    x0, x1 = L, W - R
    for i, (lab, arr, is_diff) in enumerate(panels):
        y0 = T + i * (PH + GAP)
        if is_diff:
            heat(cv, arr, x0, y0, x1, y0 + PH, -sym, sym, DIV, div=True)
        else:
            heat(cv, arr, x0, y0, x1, y0 + PH, np.percentile(arr, 5), arr.max(), SEQ)
        cv.text(L, y0 - 15, lab[:70], (40, 40, 44), 2)
        cv.text(4, y0 + 4, "7.8k", (120, 120, 120), 2)
        cv.text(4, y0 + PH - 12, "60", (120, 120, 120), 2)
        for _, s, e in bounds:
            xs = x0 + s / dur_s * (x1 - x0)
            cv.rect(xs, y0, xs + 1, y0 + PH, (255, 255, 255))
    yb = T + len(panels) * (PH + GAP) - GAP + 6
    for wlab, s, e in bounds:
        xs = x0 + s / dur_s * (x1 - x0)
        xe = x0 + e / dur_s * (x1 - x0)
        cv.rect(xs, yb, xe - 2, yb + 3, (90, 90, 95))
        cv.text(xs + 2, yb + 8, wlab[:8], (60, 60, 64), 2)
    cv.text(L, 18, title[:88], (25, 25, 30), 2)
    cv.text(L, 40, "blue = quieter than base   red = louder than base   "
                   f"scale +/- {sym:.0f} dB", (120, 120, 125), 2)
    for j, ln in enumerate(note_lines):
        cv.text(L, yb + 30 + j * 18, ln[:100], (70, 70, 76), 2)
    cv.save(path)


def concentration(x):
    """Participation ratio of a non-negative profile, as a fraction of its length.

    1.0 = perfectly spread over the utterance, small = concentrated on a few
    frames. Scale-free, so it compares diffs of very different magnitude.
    """
    p = np.asarray(x, float) ** 2
    p = p / max(p.sum(), 1e-12)
    return float(1.0 / np.sum(p ** 2) / p.size)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"n_mels": N_MELS, "hop": HOP, "sr": SR, "sentence": SENTENCE}

    mel, env = {}, {}
    def add(key, path):
        mel[key] = logmel(load16k(path))
        env[key] = envelope(mel[key])

    for b in ("M1", "F1"):
        add(f"{b}_0.00", os.path.join(RAY_DIR, f"{b}_eps0.00.wav"))
        for e in EPS_SHOW:
            add(f"{b}_{e:.2f}", os.path.join(RAY_DIR, f"{b}_eps{e:.2f}.wav"))
    for e in EPS_SHOW:
        add(f"M1_towardF1_{e:.2f}", os.path.join(PROSODY_DIR, f"M1_towardF1_eps{e:.2f}.wav"))

    nf = min(v.shape[1] for v in mel.values())
    dur = nf * HOP / SR

    # ---- alignment, in the domain that matters -----------------------------
    lag = {}
    for b in ("M1", "F1"):
        for e in EPS_SHOW:
            lag[f"{b}_eps{e:.2f}"] = env_lag_ms(env[f"{b}_0.00"], env[f"{b}_{e:.2f}"])
    lag["M1_vs_F1_swap"] = env_lag_ms(env["M1_0.00"], env["F1_0.00"])
    for e in EPS_SHOW:
        lag[f"M1_towardF1_eps{e:.2f}"] = env_lag_ms(env["M1_0.00"], env[f"M1_towardF1_{e:.2f}"])
    out["envelope_lag_ms_r0_rpeak"] = {k: list(v) for k, v in lag.items()}
    out["envelope_max_abs_lag_ms"] = float(max(abs(v[0]) for v in lag.values()))
    out["envelope_min_r0_over_rpeak"] = float(min(v[1] / v[2] for v in lag.values()))
    print(f"[align] envelope-domain lag: max |lag| = {out['envelope_max_abs_lag_ms']:.0f} ms "
          f"(one frame = {HOP * 1000 / SR:.0f} ms); worst r(lag 0)/r(peak) = "
          f"{out['envelope_min_r0_over_rpeak']:.4f}")
    for k, (l, r0, rp) in lag.items():
        print(f"        {k:<26} lag {l:+6.0f} ms   r0 {r0:.4f}  rpeak {rp:.4f}  "
              f"ratio {r0 / rp:.4f}")

    # ---- word boundaries ---------------------------------------------------
    bounds = {b: word_bounds(os.path.join(RAY_DIR, f"{b}_eps0.00.wav"), env[f"{b}_0.00"])
              for b in ("M1", "F1")}
    out["word_bounds_s"] = {b: [[w, round(s, 3), round(e, 3)] for w, s, e in v]
                            for b, v in bounds.items()}
    for b in ("M1", "F1"):
        lin = 10.0 ** (mel[f"{b}_0.00"] / 10.0)
        lv = [10 * np.log10(lin[:, int(s * SR / HOP):int(e * SR / HOP)].mean() + 1e-12)
              for _, s, e in bounds[b]]
        print(f"[words] {b}: " + " ".join(f"{w}[{s:.2f}-{e:.2f}]{v:+.0f}dB"
                                          for (w, s, e), v in zip(bounds[b], lv)))
        out.setdefault("word_base_level_db", {})[b] = [float(v) for v in lv]

    def fr(t):
        return int(np.clip(round(t * SR / HOP), 0, nf))

    def per_word(key, base):
        """Per-word energy in dB: mean LINEAR power over the word, then dB.

        Averaging dB instead would let a stop gap inside a word dominate the
        word's number, which turns a small absolute change in a near-silent
        frame into tens of dB.
        """
        lin = 10.0 ** (mel[key] / 10.0)
        return np.array([10.0 * np.log10(lin[:, fr(s):fr(e)].mean() + 1e-12)
                         for _, s, e in bounds[base]])

    # ---- diff maps and their scores ---------------------------------------
    def score(base_key, key, base):
        D = mel[key][:, :nf] - mel[base_key][:, :nf]
        g = D.mean(0)                       # broadband per-frame gain
        S = D - g                           # gain-free spectral shape
        # score on loud frames only: mel bins in near-silence swing wildly
        m = env[base_key][:nf] > env[base_key][:nf].max() - 25.0
        pw = per_word(key, base) - per_word(base_key, base)
        return {
            "d_total_db": rms(D[:, m]),
            "broadband_gain_db": rms(g[m]),
            "spectral_shape_db": rms(S[:, m]),
            "frac_energy_broadband": float(rms(g[m]) ** 2 / max(rms(D[:, m]) ** 2, 1e-12)),
            "time_concentration": concentration(np.abs(D[:, m]).mean(0)),
            "freq_concentration": concentration(np.abs(D[:, m]).mean(1)),
            "per_word_delta_db": pw.tolist(),
            "per_word_spread_db": float(pw.max() - pw.min()),
            "per_word_rms_db": rms(pw - pw.mean()),
            "mean_gain_db": float(g[m].mean()),
        }, D

    scores, diffs = {}, {}
    for b in ("M1", "F1"):
        for e in EPS_SHOW:
            scores[f"{b}_random_{e:.2f}"], diffs[f"{b}_random_{e:.2f}"] = \
                score(f"{b}_0.00", f"{b}_{e:.2f}", b)
    for e in EPS_SHOW:
        scores[f"M1_towardF1_{e:.2f}"], diffs[f"M1_towardF1_{e:.2f}"] = \
            score("M1_0.00", f"M1_towardF1_{e:.2f}", "M1")
    scores["M1_to_F1_swap"], diffs["M1_to_F1_swap"] = score("M1_0.00", "F1_0.00", "M1")
    out["mel_diff_scores"] = scores

    hdr = (f"{'diff vs base':<22}{'total':>8}{'gain':>8}{'shape':>8}{'fracBB':>8}"
           f"{'timeC':>8}{'freqC':>8}{'wordRMS':>9}{'wordSpr':>9}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for k, v in scores.items():
        print(f"{k:<22}{v['d_total_db']:>8.2f}{v['broadband_gain_db']:>8.2f}"
              f"{v['spectral_shape_db']:>8.2f}{v['frac_energy_broadband']:>8.3f}"
              f"{v['time_concentration']:>8.3f}{v['freq_concentration']:>8.3f}"
              f"{v['per_word_rms_db']:>9.2f}{v['per_word_spread_db']:>9.2f}")

    # ---- the crux, normalised by acoustic change rather than style angle ---
    # The recorded "3-5x more audible" ratio compares directions at matched
    # per-row ANGLE, with delta-ECAPA as the audibility stick. Two different
    # normalisations are needed to read it: identity change per unit of loudness
    # change (does ECAPA under-report random directions for what they do to the
    # audio?), and a magnitude-matched pairing (at equal acoustic disturbance,
    # how much identity does each family buy?).
    rp = os.path.join(OUT_DIR, "prosody_report.json")
    if os.path.exists(rp):
        pr = json.load(open(rp))
        lad, ec = pr["ladders"], pr.get("ecapa_cosine_distance", {})
        rows = []
        for key in ("M1_random", "M1_presetpca", "M1_towardF1", "F1_random", "F1_towardM1"):
            for r in lad[key]:
                dE = float(np.hypot(r["gain_db"], r["d_emphasis_db"]))
                rows.append({"set": key, "eps": r["eps"], "d_energy_db": dE,
                             "d_ecapa": r.get("d_ecapa", float("nan")),
                             "d_spectral_shape_db": r["d_spectral_shape_db"],
                             "d_ltas_shape_db": r["d_ltas_shape_db"],
                             "identity_per_loudness": r.get("d_ecapa", float("nan")) / max(dE, 1e-9),
                             "timbre_per_loudness": r["d_ltas_shape_db"] / max(dE, 1e-9)})
        sw = pr["reference_M1_to_F1_swap"]
        dEs = float(np.hypot(sw["gain_db"], sw["d_emphasis_db"]))
        rows.append({"set": "M1_to_F1_swap", "eps": float("nan"), "d_energy_db": dEs,
                     "d_ecapa": ec.get("M1_to_F1_swap", float("nan")),
                     "d_spectral_shape_db": sw["d_spectral_shape_db"],
                     "d_ltas_shape_db": sw["d_ltas_shape_db"],
                     "identity_per_loudness": ec.get("M1_to_F1_swap", float("nan")) / dEs,
                     "timbre_per_loudness": sw["d_ltas_shape_db"] / dEs})
        out["normalised_crux"] = rows
        h = (f"{'set':<16}{'eps':>6}{'dEnergy':>9}{'dECAPA':>8}{'dLTAS':>8}"
             f"{'ECAPA/dE':>10}{'LTAS/dE':>9}")
        print("\n" + h + "\n" + "-" * len(h))
        for r in rows:
            print(f"{r['set']:<16}{r['eps']:>6.2f}{r['d_energy_db']:>9.2f}{r['d_ecapa']:>8.3f}"
                  f"{r['d_ltas_shape_db']:>8.2f}{r['identity_per_loudness']:>10.4f}"
                  f"{r['timbre_per_loudness']:>9.4f}")
        # magnitude-matched: interpolate the preset-diff ladder onto the random
        # ladder's own acoustic distances, then compare identity at equal change
        rnd = [r for r in rows if r["set"] == "M1_random"]
        pre = [r for r in rows if r["set"] == "M1_towardF1"]
        xs = [r["d_energy_db"] for r in pre]
        ys = [r["d_ecapa"] for r in pre]
        matched = []
        for r in rnd:
            matched.append({"eps_random": r["eps"], "d_energy_db": r["d_energy_db"],
                            "ecapa_random": r["d_ecapa"],
                            "ecapa_presetdiff_at_same_energy": float(np.interp(r["d_energy_db"], xs, ys)),
                            "ratio": float(np.interp(r["d_energy_db"], xs, ys) / max(r["d_ecapa"], 1e-9))})
        out["magnitude_matched_M1"] = matched
        print("\nmagnitude-matched (base M1): equal energy-contour change, how much ECAPA?")
        print(f"{'eps_rand':>9}{'dEnergy':>9}{'ECAPArand':>11}{'ECAPApreset':>13}{'ratio':>8}")
        for m in matched:
            print(f"{m['eps_random']:>9.2f}{m['d_energy_db']:>9.2f}{m['ecapa_random']:>11.3f}"
                  f"{m['ecapa_presetdiff_at_same_energy']:>13.3f}{m['ratio']:>8.2f}")

    # ---- per-word ladder and emphasis-shape stability -----------------------
    # One consistent measure for the M1/F1 asymmetry the listener reported:
    # per-word energy is mean LINEAR power over the word in dB, the utterance
    # level change is plain waveform rms, and the per-word numbers are quoted
    # after that utterance change is removed, so "everything got louder" and
    # "these words got louder relative to those" are separated. The shape
    # correlation asks whether the emphasis pattern REORDERS as eps grows or
    # just scales -- the difference between "which words are emphasised is
    # changing" and "the volume goes up on the emphasised words".
    ladder_eps = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]
    word_ladder = {}
    print("\nper-word energy ladder (dB re base, utterance level change removed)")
    print(f"{'base':<5}{'eps':>6}{'utt dB':>8}{'sd dB':>7}{'range':>7}  "
          + " ".join(f"{w:>5}" for w, _, _ in bounds["M1"]))
    for b in ("M1", "F1"):
        w0 = load16k(os.path.join(RAY_DIR, f"{b}_eps0.00.wav"))
        lin0 = 10.0 ** (logmel(w0) / 10.0)
        e0 = np.array([10 * np.log10(lin0[:, fr(s0):fr(e0_)].mean() + 1e-12)
                       for _, s0, e0_ in bounds[b]])
        r0 = 20 * np.log10(np.sqrt((w0 ** 2).mean()))
        rows = []
        for eps in ladder_eps:
            w = load16k(os.path.join(RAY_DIR, f"{b}_eps{eps:.2f}.wav"))
            lin = 10.0 ** (logmel(w) / 10.0)
            e1 = np.array([10 * np.log10(lin[:, fr(s0):fr(e0_)].mean() + 1e-12)
                           for _, s0, e0_ in bounds[b]])
            utt = float(20 * np.log10(np.sqrt((w ** 2).mean())) - r0)
            dd = (e1 - e0) - utt
            rows.append({"eps": eps, "utterance_level_db": utt,
                         "per_word_db_after_level": dd.tolist(),
                         "per_word_sd_db": float(dd.std()),
                         "per_word_range_db": float(dd.max() - dd.min())})
            print(f"{b:<5}{eps:>6.2f}{utt:>8.2f}{dd.std():>7.2f}{dd.max() - dd.min():>7.2f}  "
                  + " ".join(f"{x:>+5.1f}" for x in dd))
        ref = np.array(rows[2]["per_word_db_after_level"])       # the eps 0.20 shape
        for r_ in rows:
            v = np.array(r_["per_word_db_after_level"])
            r_["shape_corr_with_eps020"] = float(np.corrcoef(ref, v)[0, 1])
        word_ladder[b] = rows
        print(f"{b}: emphasis-shape correlation with the eps 0.20 shape: "
              + " ".join(f"{r_['eps']:.2f}:{r_['shape_corr_with_eps020']:+.2f}" for r_ in rows))
    out["per_word_ladder"] = word_ladder

    # ---- F0 register, from the sweep's own manifest ------------------------
    # The per-frame F0 in phase2b_prosody_analysis.py octave-errors above eps
    # 1.60 (its F0 IQR triples), so the register question is settled with the
    # sweep's already-recorded median F0 instead, which is stable throughout.
    mf = json.load(open(os.path.join(RAY_DIR, "manifest.json")))
    f0 = {}
    for b in ("M1", "F1"):
        rows = [o for o in mf["outputs"] if o["base"] == b]
        f0ref = [o for o in rows if o["eps"] == 0.0][0]["median_f0"]
        f0[b] = [{"eps": o["eps"], "median_f0_hz": o["median_f0"],
                  "shift_cents": float(1200 * np.log2(o["median_f0"] / f0ref)),
                  "f0_iqr_hz": o["f0_iqr"], "wer": o.get("wer"),
                  "pyin_reliable": o["f0_iqr"] < 2 * [r for r in rows if r["eps"] == 0.0][0]["f0_iqr"]}
                 for o in rows]
        print(f"[f0] {b}: " + "  ".join(
            f"eps{r['eps']:.2f}:{r['median_f0_hz']:.0f}Hz({r['shift_cents']:+.0f}ct)"
            + ("" if r["pyin_reliable"] else "*") for r in f0[b]))
    out["f0_register_from_sweep_manifest"] = f0
    out["f0_note"] = ("median F0 from the sweep manifest; * marks clips whose F0 IQR more "
                      "than doubles, i.e. where pyin is octave-erroring and only the "
                      "median is trustworthy. The per-frame F0 contour distances in "
                      "prosody_report.json are unreliable for those clips.")

    # ---- level-matched A/B clips -------------------------------------------
    # A loudness difference dominates any casual A/B, so these are rms-matched
    # to the base: what is left to hear is the emphasis and spectral change.
    ab = "results/listening_sets/phase2b_prosody_ab"
    os.makedirs(ab, exist_ok=True)
    ab_manifest = []

    def level_match(src_path, ref_path, out_name, note):
        x, sr = sf.read(src_path, dtype="float32")
        r, _ = sf.read(ref_path, dtype="float32")
        gdb = 20 * np.log10(np.sqrt((r ** 2).mean()) / np.sqrt((x ** 2).mean()))
        y = x * 10.0 ** (gdb / 20.0)
        sf.write(os.path.join(ab, out_name), y, sr)
        ab_manifest.append({"file": out_name, "source": os.path.basename(src_path),
                            "applied_gain_db": float(gdb), "peak": float(np.abs(y).max()),
                            "note": note})

    base_m1 = os.path.join(RAY_DIR, "M1_eps0.00.wav")
    base_f1 = os.path.join(RAY_DIR, "F1_eps0.00.wav")
    level_match(base_m1, base_m1, "A0_M1_base.wav", "M1 unperturbed, the reference")
    level_match(base_f1, base_f1, "B0_F1_base.wav", "F1 unperturbed, the reference")
    for e in (0.20, 0.80, 3.20):
        level_match(os.path.join(RAY_DIR, f"M1_eps{e:.2f}.wav"), base_m1,
                    f"A_M1_random_eps{e:.2f}_levelmatched.wav",
                    f"M1 + random direction, eps {e:.2f}, rms matched to the base")
        level_match(os.path.join(RAY_DIR, f"F1_eps{e:.2f}.wav"), base_f1,
                    f"B_F1_random_eps{e:.2f}_levelmatched.wav",
                    f"F1 + random direction, eps {e:.2f}, rms matched to the base")
    level_match(os.path.join(PROSODY_DIR, "M1_towardF1_eps0.20.wav"), base_m1,
                "C_M1_towardF1_eps0.20_levelmatched.wav",
                "M1 rotated 11 deg toward F1 -- same rms log-mel change as A eps 3.20")
    level_match(base_f1, base_m1, "C_F1_full_swap_levelmatched.wav",
                "the full M1 to F1 preset swap, the known identity change")
    with open(os.path.join(ab, "manifest.json"), "w") as f:
        json.dump({"experiment": "phase2b_prosody_ab",
                   "text": SENTENCE, "total_step": 8, "speed": 1.05,
                   "style_dp": "held fixed at M1's", "rng_seed": 4242,
                   "note": "rms level-matched to the base so the comparison is not "
                           "decided by loudness; A/B pair of interest is "
                           "A_M1_random_eps3.20 against C_M1_towardF1_eps0.20, which "
                           "carry the same rms log-mel change",
                   "outputs": ab_manifest}, f, indent=2)
    out["ab_clips"] = ab_manifest
    print(f"\nWrote {ab} ({len(ab_manifest)} level-matched clips)")

    # ---- figures -----------------------------------------------------------
    figs = {}
    for b in ("M1", "F1"):
        sym = float(np.percentile(np.abs(diffs[f"{b}_random_3.20"]), 99))
        panels = [(f"{b} base, eps 0.00, log-mel dB", mel[f"{b}_0.00"][:, :nf], False)]
        panels += [(f"diff eps {e:.2f} minus base  (rms {scores[f'{b}_random_{e:.2f}']['d_total_db']:.1f} dB)",
                    diffs[f"{b}_random_{e:.2f}"], True) for e in EPS_SHOW]
        p = os.path.join(OUT_DIR, f"fig_mel_{b}_random_ray.png")
        panel_fig(p, f"{b}: random style_ttl direction, log-mel difference from base",
                  panels, bounds[b], dur, sym,
                  [f"time concentration {scores[f'{b}_random_3.20']['time_concentration']:.2f} "
                   f"(1.0 = spread evenly over the utterance)",
                   f"broadband share of diff energy "
                   f"{scores[f'{b}_random_3.20']['frac_energy_broadband']:.2f}"])
        figs[f"fig_mel_{b}_random_ray.png"] = f"{b} random ray, base + 3 signed diffs, +/-{sym:.0f} dB"
    symA = float(np.percentile(np.abs(diffs["M1_random_3.20"]), 99))
    panels = [("M1 base, log-mel dB", mel["M1_0.00"][:, :nf], False),
              (f"M1 to F1 full preset swap  (rms {scores['M1_to_F1_swap']['d_total_db']:.1f} dB)",
               diffs["M1_to_F1_swap"], True),
              (f"M1 toward F1, eps 0.80  (rms {scores['M1_towardF1_0.80']['d_total_db']:.1f} dB)",
               diffs["M1_towardF1_0.80"], True),
              (f"M1 random, eps 3.20  (rms {scores['M1_random_3.20']['d_total_db']:.1f} dB)",
               diffs["M1_random_3.20"], True)]
    p = os.path.join(OUT_DIR, "fig_mel_timbre_reference.png")
    panel_fig(p, "Timbre reference: preset-aligned change vs random perturbation",
              panels, bounds["M1"], dur, symA,
              ["all three diff panels share one colour scale, so panel area is comparable",
               f"time concentration: swap {scores['M1_to_F1_swap']['time_concentration']:.2f}, "
               f"towardF1 {scores['M1_towardF1_0.80']['time_concentration']:.2f}, "
               f"random {scores['M1_random_3.20']['time_concentration']:.2f}"])
    figs["fig_mel_timbre_reference.png"] = "M1->F1 swap vs towardF1 vs random, one shared scale"

    # matched total change: random eps 3.20 and toward-F1 eps 0.20 happen to
    # produce almost the same rms log-mel difference, so the two maps can be
    # read against each other without a magnitude confound.
    symM = float(np.percentile(np.abs(np.concatenate(
        [diffs["M1_random_3.20"], diffs["M1_towardF1_0.20"]], axis=1)), 99))
    panels = [("M1 base, log-mel dB", mel["M1_0.00"][:, :nf], False),
              (f"random eps 3.20, 73 deg per row  (rms {scores['M1_random_3.20']['d_total_db']:.1f} dB)",
               diffs["M1_random_3.20"], True),
              (f"toward F1 eps 0.20, 11 deg per row  (rms {scores['M1_towardF1_0.20']['d_total_db']:.1f} dB)",
               diffs["M1_towardF1_0.20"], True)]
    panel_fig(os.path.join(OUT_DIR, "fig_mel_matched_magnitude.png"),
              "Matched total change: 6x the per-row angle buys the same rms difference",
              panels, bounds["M1"], dur, symM,
              ["both restripe harmonics; the random panel is red-dominant (net level lift)",
               "with word-scale blocks, the preset panel is balanced red/blue and more even."])
    figs["fig_mel_matched_magnitude.png"] = "random vs preset-diff at equal rms log-mel change"

    # per-word energy ladder, both bases, with the utterance level change
    # removed so the curves show redistribution rather than overall loudness
    from phase2b_prosody_analysis import plot_curves
    for b in ("M1", "F1"):
        x = np.arange(len(bounds[b])).astype(float)
        rows = {r_["eps"]: r_ for r_ in word_ladder[b]}
        plot_curves(os.path.join(OUT_DIR, f"fig_per_word_energy_{b}.png"),
                    f"{b}: per-word energy vs base, random direction, level change removed",
                    [(f"eps {e:.2f} (utt {rows[e]['utterance_level_db']:+.1f} dB)", x,
                      np.array(rows[e]["per_word_db_after_level"]), i * 2 + 1)
                     for i, e in enumerate((0.20, 0.80, 3.20))],
                    "dB re base", " ".join(w for w, _, _ in bounds[b]))
        figs[f"fig_per_word_energy_{b}.png"] = (
            f"{b} per-word dB change across the eps ladder, utterance level removed")

    out["figures"] = figs
    with open(os.path.join(OUT_DIR, "mel_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}")
    for k, v in figs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
