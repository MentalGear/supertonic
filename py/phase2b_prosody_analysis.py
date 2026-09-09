"""Phase 2b follow-up: is a random style_ttl direction prosodic rather than timbral?

The recorded Phase 2b mechanism is "most of style space is nearly inaudible",
inferred from delta-ECAPA between renders. A human listener contradicted it
after hearing the perturbation ray: the clips do differ, more so with
magnitude, and what moves is per-word emphasis / loudness, not voice identity.
ECAPA is a speaker-verification embedding trained to be invariant to prosody,
so it would be blind to exactly that change by construction.

Every clip compared here was rendered from the same text with the vocoder RNG
seeded identically and style_dp pinned, so the waveforms are sample-aligned and
can be compared frame by frame rather than through utterance-level aggregates.
Alignment is verified, not assumed.

The measurement decomposes the log-magnitude spectrogram L[f, t] of each clip
against its base into two orthogonal parts:

    L[f, t] = g[t] + S[f, t],    g[t] = mean_f L[f, t],   mean_f S[f, t] = 0

  * `g` is the frame energy contour -- how loudness is distributed in time.
    Its change splits further into a constant offset (overall gain) and a
    zero-mean residual (emphasis redistribution: which syllables got louder
    relative to the rest).
  * `S` is the per-frame spectral shape, gain-free -- the timbre-carrying part.

Alongside these: F0 contour distance in cents, split the same way into a median
register shift and a contour-shape residual; a long-term average spectrum; and
per-word energy over an energy-based segmentation of the one fixed sentence,
computed once on the base clip and reused for every clip of that base so the
comparison is word for word.

The same battery runs on three direction families of matched per-row geometry
(random, random-inside-the-preset-span, toward another preset) and on the full
M1 -> F1 swap, so "which measurement axis does this direction load on" can be
answered rather than assumed. ECAPA cosine distance is recomputed on the same
clips as a reference stick.

Usage (from py/):  python3 phase2b_prosody_analysis.py [--no-ecapa]
"""

import argparse
import json
import os
import struct
import zlib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

RAY_DIR = "results/listening_sets/phase2b_perturbation_ray"
PROSODY_DIR = "results/listening_sets/phase2b_prosody"
PHASE0_DIR = "results/listening_sets/phase0_linearity"
OUT_DIR = "results/phase2b_prosody"
SR = 16000
NFFT, HOP = 512, 128          # 32 ms window, 8 ms hop at 16 kHz
FMIN, FMAX = 100.0, 7000.0
EPS_LADDER = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20]
SENTENCE = "The quick brown fox jumps over the lazy dog."
WORDS = SENTENCE.rstrip(".").split()


# ---------------------------------------------------------------- feature side

def load16k(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    return resample_poly(w, SR, sr).astype(np.float32) if sr != SR else w


def logspec(w):
    """(bins, frames) log-magnitude spectrogram in dB over the analysis band."""
    f, _, z = stft(w, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    keep = (f >= FMIN) & (f <= FMAX)
    return 20.0 * np.log10(np.abs(z[keep]) + 1e-8), f[keep]


def decompose(L):
    """L -> (g, S) with g the per-frame mean level in dB and S the gain-free shape."""
    g = L.mean(0)
    return g, L - g


def speech_mask(g):
    """Frames within 45 dB of the loudest frame -- speech rather than silence."""
    return g > g.max() - 45.0


def f0_cents(w):
    import librosa
    f0, voiced, _ = librosa.pyin(w, fmin=60, fmax=400, sr=SR,
                                 frame_length=NFFT * 2, hop_length=HOP, center=True)
    c = np.full(f0.shape, np.nan, dtype=float)
    ok = np.isfinite(f0) & voiced
    c[ok] = 1200.0 * np.log2(f0[ok] / 55.0)
    return c


def segment_words(g, mask, base=None):
    """Word units for the fixed sentence, as frame index pairs.

    Prefers the forced word boundaries written by `phase2b_prosody_mel.py`
    (faster-whisper word timestamps, repaired) -- energy thresholding cannot
    split this sentence into nine words because it is spoken without pauses,
    and silently returns two units instead. The energy fallback below is kept
    only so this script runs before the mel one has.
    """
    mp = os.path.join(OUT_DIR, "mel_report.json")
    if base and os.path.exists(mp):
        wb = json.load(open(mp)).get("word_bounds_s", {}).get(base)
        if wb:
            return [(int(round(s * SR / HOP)), int(round(e * SR / HOP))) for _, s, e in wb]
    on = (g > g.max() - 30.0) & mask
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    gap = int(round(0.048 * SR / HOP))
    for i in idx[1:]:
        if i - prev > gap:
            segs.append((start, prev + 1))
            start = i
        prev = i
    segs.append((start, prev + 1))
    minlen = int(round(0.064 * SR / HOP))
    return [s for s in segs if s[1] - s[0] >= minlen]


class Clip:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.w = load16k(path)
        self.L, self.freqs = logspec(self.w)
        self.g, self.S = decompose(self.L)
        self.mask = speech_mask(self.g)
        self._f0 = None

    @property
    def f0(self):
        if self._f0 is None:
            self._f0 = f0_cents(self.w)
        return self._f0

    def ltas(self, mask=None):
        m = self.mask if mask is None else mask
        return self.L[:, m].mean(1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def compare(a, b):
    """Frame-resolved distances from clip `a` (base) to clip `b`."""
    m = a.mask & b.mask
    dg = b.g - a.g
    d = {
        "n_speech_frames": int(m.sum()),
        # energy contour, split into overall gain and emphasis redistribution
        "gain_db": float(dg[m].mean()),
        "d_energy_contour_db": rms(dg[m]),
        "d_emphasis_db": rms(dg[m] - dg[m].mean()),
        # gain-free per-frame spectral shape: the timbre-carrying part
        "d_spectral_shape_db": rms((b.S - a.S)[:, m]),
        # long-term average spectrum, raw and gain-removed
        "d_ltas_db": rms(b.ltas(m) - a.ltas(m)),
        "d_ltas_shape_db": rms((b.ltas(m) - a.ltas(m)) - (b.ltas(m) - a.ltas(m)).mean()),
        # the Phase 0 / direction-audibility stick, for continuity
        "d_logstft_mean_abs": float(np.abs(b.L - a.L).mean()),
    }
    ca, cb = a.f0, b.f0
    ok = np.isfinite(ca) & np.isfinite(cb)
    n = min(ca.size, cb.size, m.size)
    ok = ok[:n] & m[:n]
    if ok.sum() > 10:
        da = ca[:n][ok]
        db = cb[:n][ok]
        d["n_covoiced_frames"] = int(ok.sum())
        d["f0_median_shift_cents"] = float(np.median(db) - np.median(da))
        d["d_f0_contour_cents"] = rms(db - da)
        d["d_f0_shape_cents"] = rms((db - np.median(db)) - (da - np.median(da)))
        d["f0_range_ratio"] = float(np.std(db) / max(np.std(da), 1e-6))
    va = np.isfinite(ca).mean()
    vb = np.isfinite(cb).mean()
    d["voiced_fraction_base"] = float(va)
    d["voiced_fraction_other"] = float(vb)
    return d


def alignment_lag_ms(a, b, max_ms=60.0):
    """Sample-domain cross-correlation lag between two clips, in milliseconds."""
    n = min(a.w.size, b.w.size)
    x = a.w[:n] - a.w[:n].mean()
    y = b.w[:n] - b.w[:n].mean()
    k = int(max_ms * SR / 1000)
    xc = np.correlate(y, x[k:n - k], mode="valid")
    return float((int(np.argmax(np.abs(xc))) - k) * 1000.0 / SR)


# ------------------------------------------------------------------- PNG plots

_FONT = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
    " ": "000000000000000", "e": "000111101111011", "p": "000111101111100",
    "s": "000111100010111", "d": "001011111101111", "B": "110101110101110",
    "F": "111100110100100", "M": "101111111101101", "1": "010110010010111",
    "r": "000111100100100", "a": "000011111101111", "n": "000110101101101",
    "o": "000111101101111", "m": "000111111101101", "t": "010111010010011",
    "w": "000101101111111", "f": "011100110100100", "c": "000111100100111",
    "g": "000111101111001", "y": "000101101111001", "i": "010000010010010",
    "b": "100110101101110", "l": "110010010010111", "h": "100100111101101",
    "u": "000101101101111", "v": "000101101101010", "x": "000101010010101",
    "k": "100101110110101", "z": "000111001010111", "q": "000111101111001",
    "j": "001000001101010", "A": "010101111101101", "C": "011100100100011",
    "E": "111100110100111", "P": "110101110100100", "R": "110101110110101",
    "T": "111010010010010", "S": "011100010001110", "0": "111101101101111",
    "/": "001001010100100", "(": "001010100010001", ")": "100010001010100",
    ":": "000010000010000", "=": "000111000111000", "+": "010010111010010",
    "%": "101001010100101", ",": "000000000010100", "?": "111001010000010",
}


class Canvas:
    def __init__(self, w, h, bg=(250, 250, 248)):
        self.w, self.h = w, h
        self.px = np.full((h, w, 3), bg, dtype=np.uint8)

    def line(self, x0, y0, x1, y1, c, width=1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n * 2)
        ys = np.linspace(y0, y1, n * 2)
        for dx in range(width):
            for dy in range(width):
                xi = np.clip((xs + dx).astype(int), 0, self.w - 1)
                yi = np.clip((ys + dy).astype(int), 0, self.h - 1)
                self.px[yi, xi] = c

    def poly(self, xs, ys, c, width=1):
        for i in range(len(xs) - 1):
            self.line(xs[i], ys[i], xs[i + 1], ys[i + 1], c, width)

    def rect(self, x0, y0, x1, y1, c):
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        self.px[np.clip(y0, 0, self.h):np.clip(y1, 0, self.h),
                np.clip(x0, 0, self.w):np.clip(x1, 0, self.w)] = c

    def text(self, x, y, s, c=(40, 40, 40), scale=2):
        for ch in s:
            bits = _FONT.get(ch, _FONT.get(ch.lower(), _FONT[" "]))
            for r in range(5):
                for col in range(3):
                    if bits[r * 3 + col] == "1":
                        self.rect(x + col * scale, y + r * scale,
                                  x + (col + 1) * scale, y + (r + 1) * scale, c)
            x += 4 * scale

    def save(self, path):
        raw = b"".join(b"\x00" + self.px[i].tobytes() for i in range(self.h))
        def chunk(tag, data):
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
        with open(path, "wb") as f:
            f.write(png)


PALETTE = [(30, 30, 34), (24, 108, 122), (32, 150, 148), (72, 178, 140),
           (150, 190, 90), (214, 168, 58), (222, 118, 52), (198, 62, 60)]


def plot_curves(path, title, curves, ylabel, xlabel, segs=None, seg_labels=None,
                W=1100, H=430):
    """curves: list of (label, x, y, color_index)."""
    cv = Canvas(W, H)
    L, R, T, B = 90, 210, 46, 56
    ys = np.concatenate([c[2][np.isfinite(c[2])] for c in curves])
    xs = np.concatenate([c[1] for c in curves])
    y0, y1 = float(ys.min()), float(ys.max())
    pad = 0.08 * max(y1 - y0, 1e-6)
    y0, y1 = y0 - pad, y1 + pad
    x0, x1 = float(xs.min()), float(xs.max())
    def px(x):
        return L + (x - x0) / max(x1 - x0, 1e-9) * (W - L - R)
    def py(y):
        return H - B - (y - y0) / max(y1 - y0, 1e-9) * (H - T - B)
    if segs:
        for i, (s, e) in enumerate(segs):
            if i % 2 == 0:
                cv.rect(px(s), T, px(e), H - B, (236, 238, 236))
            if seg_labels and i < len(seg_labels):
                cv.text(px(s) + 3, H - B + 8, seg_labels[i][:9], (110, 110, 110), 2)
    cv.rect(L, T, L + 1, H - B, (150, 150, 150))
    cv.rect(L, H - B, W - R, H - B + 1, (150, 150, 150))
    for t in np.linspace(y0, y1, 5):
        cv.text(6, py(t) - 5, f"{t:6.1f}", (110, 110, 110), 2)
        cv.rect(L - 4, py(t), L, py(t) + 1, (150, 150, 150))
    for i, (label, x, y, ci) in enumerate(curves):
        c = PALETTE[ci % len(PALETTE)]
        ok = np.isfinite(y)
        xx, yy = px(np.asarray(x)[ok]), py(np.asarray(y)[ok])
        cv.poly(xx, yy, c, 2)
        cv.rect(W - R + 12, T + 8 + i * 22, W - R + 32, T + 12 + i * 22, c)
        cv.text(W - R + 38, T + 4 + i * 22, label[:16], (60, 60, 60), 2)
    cv.text(L, 14, title[:80], (30, 30, 34), 2)
    cv.text(L, H - 16, xlabel[:60], (110, 110, 110), 2)
    cv.text(6, T - 14, ylabel[:22], (110, 110, 110), 2)
    cv.save(path)


def plot_bars(path, title, groups, series, values, ylabel, W=1100, H=470):
    """values[g][s]; bars grouped by g, coloured by s."""
    cv = Canvas(W, H)
    L, R, T, B = 90, 210, 46, 62
    vmax = max(max(r) for r in values) * 1.15 or 1.0
    gw = (W - L - R) / len(groups)
    bw = gw * 0.8 / len(series)
    for gi, g in enumerate(groups):
        for si in range(len(series)):
            v = values[gi][si]
            x = L + gi * gw + gw * 0.1 + si * bw
            y = H - B - v / vmax * (H - T - B)
            cv.rect(x, y, x + bw - 3, H - B, PALETTE[(si + 1) % len(PALETTE)])
            cv.text(x, y - 14, f"{v:.2f}", (90, 90, 90), 2)
        cv.text(L + gi * gw + 6, H - B + 10, g[:18], (60, 60, 60), 2)
    cv.rect(L, H - B, W - R, H - B + 1, (150, 150, 150))
    for si, s in enumerate(series):
        cv.rect(W - R + 12, T + 8 + si * 22, W - R + 32, T + 12 + si * 22,
                PALETTE[(si + 1) % len(PALETTE)])
        cv.text(W - R + 38, T + 4 + si * 22, s[:16], (60, 60, 60), 2)
    cv.text(L, 14, title[:80], (30, 30, 34), 2)
    cv.text(6, T - 14, ylabel[:22], (110, 110, 110), 2)
    cv.save(path)


# ------------------------------------------------------------------------ main

def ecapa_distances(paths):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/home/user/.cache/ecapa",
                                         run_opts={"device": "cpu"})
    enc.eval()
    out = {}
    for p in paths:
        w = load16k(p)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(w)[None, :]).squeeze().numpy()
        out[os.path.basename(p)] = e / np.linalg.norm(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ecapa", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"sentence": SENTENCE, "sr": SR, "nfft": NFFT, "hop": HOP,
              "band_hz": [FMIN, FMAX]}

    # ---- clips -------------------------------------------------------------
    ray = {b: {e: Clip(os.path.join(RAY_DIR, f"{b}_eps{e:.2f}.wav"))
               for e in [0.0] + EPS_LADDER} for b in ("M1", "F1")}
    fam = {}
    for base, name in (("M1", "towardF1"), ("M1", "presetpca"), ("F1", "towardM1")):
        fam[(base, name)] = {e: Clip(os.path.join(PROSODY_DIR, f"{base}_{name}_eps{e:.2f}.wav"))
                             for e in EPS_LADDER}
    fam[("M1", "random")] = {e: ray["M1"][e] for e in EPS_LADDER}
    fam[("F1", "random")] = {e: ray["F1"][e] for e in EPS_LADDER}
    baseclip = {"M1": ray["M1"][0.0], "F1": ray["F1"][0.0]}

    # ---- 0. alignment ------------------------------------------------------
    lags = {}
    for b in ("M1", "F1"):
        for e in EPS_LADDER:
            lags[f"{b}_eps{e:.2f}"] = alignment_lag_ms(baseclip[b], ray[b][e])
    lags["M1_vs_F1"] = alignment_lag_ms(baseclip["M1"], baseclip["F1"])
    for e in EPS_LADDER:
        lags[f"M1_towardF1_eps{e:.2f}"] = alignment_lag_ms(baseclip["M1"], fam[("M1", "towardF1")][e])
    report["alignment_lag_ms"] = lags
    report["alignment_max_abs_lag_ms"] = float(max(abs(v) for v in lags.values()))
    print(f"[align] max |lag| across all comparisons: "
          f"{report['alignment_max_abs_lag_ms']:.1f} ms", flush=True)

    # ---- 1. word segmentation, computed once per base ----------------------
    segs = {b: segment_words(baseclip[b].g, baseclip[b].mask, b) for b in ("M1", "F1")}
    report["segments"] = {b: [{"i": i, "start_s": s * HOP / SR, "end_s": e * HOP / SR,
                               "label": WORDS[i] if i < len(WORDS) else f"seg{i}"}
                              for i, (s, e) in enumerate(v)]
                          for b, v in segs.items()}
    for b in ("M1", "F1"):
        print(f"[segments] {b}: {len(segs[b])} units "
              f"({', '.join(f'{s*HOP/SR:.2f}-{e*HOP/SR:.2f}' for s, e in segs[b])})", flush=True)

    def seg_profile(clip, base):
        v = np.array([clip.g[s:e].mean() for s, e in segs[base]])
        return v - v.mean()

    # ---- 2. the ladders ----------------------------------------------------
    ladder = {}
    for (base, name), clips in fam.items():
        rows = []
        for e in EPS_LADDER:
            d = compare(baseclip[base], clips[e])
            d["eps"] = e
            d["seg_profile_delta_db"] = (seg_profile(clips[e], base)
                                         - seg_profile(baseclip[base], base)).tolist()
            d["d_seg_emphasis_db"] = rms(np.array(d["seg_profile_delta_db"]))
            rows.append(d)
        ladder[f"{base}_{name}"] = rows
    # the reference timbre change: full M1 -> F1 swap, seed-matched
    swap = compare(baseclip["M1"], baseclip["F1"])
    swap["seg_profile_delta_db"] = (seg_profile(baseclip["F1"], "M1")
                                    - seg_profile(baseclip["M1"], "M1")).tolist()
    swap["d_seg_emphasis_db"] = rms(np.array(swap["seg_profile_delta_db"]))
    report["reference_M1_to_F1_swap"] = swap
    report["ladders"] = ladder

    # ---- 3. content dependence --------------------------------------------
    other = {}
    for ti in (1, 3):
        base_c = Clip(os.path.join(PROSODY_DIR, f"M1t{ti}_random_eps0.00.wav"))
        rows = []
        for e in (0.20, 0.80, 3.20):
            c = Clip(os.path.join(PROSODY_DIR, f"M1t{ti}_random_eps{e:.2f}.wav"))
            d = compare(base_c, c)
            d["eps"] = e
            rows.append(d)
        other[f"text{ti}"] = rows
    report["other_sentences"] = other

    # ---- 4. ECAPA on the same clips ---------------------------------------
    if not args.no_ecapa:
        paths = [os.path.join(RAY_DIR, f"{b}_eps{e:.2f}.wav")
                 for b in ("M1", "F1") for e in [0.0] + EPS_LADDER]
        paths += [os.path.join(PROSODY_DIR, f"{b}_{n}_eps{e:.2f}.wav")
                  for (b, n) in (("M1", "towardF1"), ("M1", "presetpca"), ("F1", "towardM1"))
                  for e in EPS_LADDER]
        emb = ecapa_distances(paths)
        ec = {}
        for (base, name) in fam:
            b0 = emb[f"{base}_eps0.00.wav"]
            key = f"{base}_{name}"
            ec[key] = []
            for e in EPS_LADDER:
                fn = (f"{base}_eps{e:.2f}.wav" if name == "random"
                      else f"{base}_{name}_eps{e:.2f}.wav")
                ec[key].append(float(1.0 - emb[fn] @ b0))
        ec["M1_to_F1_swap"] = float(1.0 - emb["F1_eps0.00.wav"] @ emb["M1_eps0.00.wav"])
        report["ecapa_cosine_distance"] = ec
        for k, rows in ladder.items():
            for i, r in enumerate(rows):
                r["d_ecapa"] = ec[k][i]

    with open(os.path.join(OUT_DIR, "prosody_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- 5. printed tables -------------------------------------------------
    hdr = (f"{'set':<16}{'eps':>6}{'ang':>6}{'gain':>7}{'emph':>7}{'segE':>7}"
           f"{'spec':>7}{'ltas':>7}{'f0sh':>7}{'f0md':>7}{'ecapa':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, rows in ladder.items():
        for r in rows:
            print(f"{k:<16}{r['eps']:>6.2f}"
                  f"{np.degrees(np.arctan(r['eps'])):>6.1f}"
                  f"{r['gain_db']:>7.2f}{r['d_emphasis_db']:>7.2f}"
                  f"{r['d_seg_emphasis_db']:>7.2f}{r['d_spectral_shape_db']:>7.2f}"
                  f"{r['d_ltas_shape_db']:>7.2f}{r.get('d_f0_shape_cents', float('nan')):>7.1f}"
                  f"{r.get('f0_median_shift_cents', float('nan')):>7.1f}"
                  f"{r.get('d_ecapa', float('nan')):>7.3f}")
        print()
    r = swap
    print(f"{'M1->F1 swap':<16}{'-':>6}{'-':>6}"
          f"{r['gain_db']:>7.2f}{r['d_emphasis_db']:>7.2f}{r['d_seg_emphasis_db']:>7.2f}"
          f"{r['d_spectral_shape_db']:>7.2f}{r['d_ltas_shape_db']:>7.2f}"
          f"{r.get('d_f0_shape_cents', float('nan')):>7.1f}"
          f"{r.get('f0_median_shift_cents', float('nan')):>7.1f}"
          f"{report.get('ecapa_cosine_distance', {}).get('M1_to_F1_swap', float('nan')):>7.3f}")

    # ---- 6. the crux: ratio of preset-aligned to random, per metric --------
    metrics = [("d_ecapa", "ECAPA cos"), ("d_emphasis_db", "emphasis dB"),
               ("d_seg_emphasis_db", "per-word dB"), ("d_f0_shape_cents", "F0 shape ct"),
               ("d_spectral_shape_db", "spectral shape dB"), ("d_ltas_shape_db", "LTAS shape dB")]
    crux = {}
    print("\ncrux -- preset-aligned / random, at matched per-row angle (base M1)")
    print(f"{'metric':<20}" + "".join(f"{f'eps{e}':>10}" for e in (0.10, 0.20, 0.40, 0.80)))
    for key, lab in metrics:
        row = []
        for e in (0.10, 0.20, 0.40, 0.80):
            i = EPS_LADDER.index(e)
            rnd = ladder["M1_random"][i].get(key, float("nan"))
            pre = ladder["M1_towardF1"][i].get(key, float("nan"))
            row.append(float(pre / rnd) if rnd else float("nan"))
        crux[key] = {"label": lab, "ratio_towardF1_over_random": row}
        print(f"{lab:<20}" + "".join(f"{v:>10.2f}" for v in row))
    report["crux_ratio_presetdiff_over_random"] = crux
    with open(os.path.join(OUT_DIR, "prosody_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- 7. plots ----------------------------------------------------------
    t = np.arange(baseclip["M1"].g.size) * HOP / SR
    seg_lab = [s["label"] for s in report["segments"]["M1"]]
    plot_curves(os.path.join(OUT_DIR, "fig1_energy_contour_M1_random.png"),
                "M1 random direction: frame energy contour (dB), gain removed",
                [(f"eps {e:.2f}", t, (ray['M1'][e].g - ray['M1'][e].g[baseclip['M1'].mask].mean()), i)
                 for i, e in enumerate([0.0, 0.20, 0.80, 3.20])],
                "level dB", "time s", [(s * HOP / SR, e * HOP / SR) for s, e in segs["M1"]], seg_lab)
    plot_curves(os.path.join(OUT_DIR, "fig2_energy_contour_M1_towardF1.png"),
                "M1 toward F1: frame energy contour (dB), gain removed",
                [("eps 0.00", t, baseclip["M1"].g - baseclip["M1"].g[baseclip["M1"].mask].mean(), 0)]
                + [(f"eps {e:.2f}", t,
                    fam[("M1", "towardF1")][e].g - fam[("M1", "towardF1")][e].g[baseclip["M1"].mask].mean(), i + 1)
                   for i, e in enumerate([0.20, 0.80, 3.20])],
                "level dB", "time s", [(s * HOP / SR, e * HOP / SR) for s, e in segs["M1"]], seg_lab)
    tf = np.arange(baseclip["M1"].f0.size) * HOP / SR
    plot_curves(os.path.join(OUT_DIR, "fig3_f0_contour_M1.png"),
                "M1: F0 contour, random ray vs toward F1",
                [("random 0.00", tf, baseclip["M1"].f0, 0),
                 ("random 0.80", tf, ray["M1"][0.80].f0, 2),
                 ("random 3.20", tf, ray["M1"][3.20].f0, 4),
                 ("towardF1 0.80", tf, fam[("M1", "towardF1")][0.80].f0, 6),
                 ("F1 preset", tf, baseclip["F1"].f0, 7)],
                "cents re 55Hz", "time s")
    fr = baseclip["M1"].freqs
    plot_curves(os.path.join(OUT_DIR, "fig4_ltas_M1.png"),
                "M1: long-term average spectrum, gain removed",
                [(lab, fr, c.ltas() - c.ltas().mean(), i) for i, (lab, c) in enumerate(
                    [("base", baseclip["M1"]), ("random 0.80", ray["M1"][0.80]),
                     ("random 3.20", ray["M1"][3.20]),
                     ("towardF1 0.80", fam[("M1", "towardF1")][0.80]),
                     ("F1 preset", baseclip["F1"])])],
                "dB", "freq Hz")
    groups = ["random", "presetpca", "towardF1", "M1 to F1"]
    i8 = EPS_LADDER.index(0.80)
    series = ["emphasis dB", "per-word dB", "spectral dB", "ECAPA x10"]
    vals = []
    for gname, src in (("random", ladder["M1_random"][i8]), ("presetpca", ladder["M1_presetpca"][i8]),
                       ("towardF1", ladder["M1_towardF1"][i8]), ("M1 to F1", swap)):
        vals.append([src["d_emphasis_db"], src["d_seg_emphasis_db"], src["d_spectral_shape_db"],
                     10 * src.get("d_ecapa", report.get("ecapa_cosine_distance", {}).get("M1_to_F1_swap", 0))])
    plot_bars(os.path.join(OUT_DIR, "fig5_direction_loading.png"),
              "Which axis does each direction load on (eps 0.80, matched per-row angle)",
              groups, series, vals, "distance")
    seg_x = np.arange(len(segs["M1"])).astype(float)
    plot_curves(os.path.join(OUT_DIR, "fig6_per_word_emphasis.png"),
                "Per-word energy change vs base, M1 (word index 0-8)",
                [(f"random {e:.2f}", seg_x, np.array(ladder["M1_random"][EPS_LADDER.index(e)]["seg_profile_delta_db"]), i)
                 for i, e in enumerate([0.20, 0.80, 3.20])]
                + [("towardF1 0.80", seg_x, np.array(ladder["M1_towardF1"][i8]["seg_profile_delta_db"]), 6),
                   ("M1 to F1", seg_x, np.array(swap["seg_profile_delta_db"]), 7)],
                "dB re base", "word index")
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
