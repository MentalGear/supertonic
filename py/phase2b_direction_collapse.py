"""Phase 2b follow-up: do DISTINCT style_ttl directions sound alike?

new-plan.md closes Phase 2b on a synthesis -- "audible does not mean
identifiable" -- whose load-bearing claim is that a 6,120-dimensional
perturbation collapses onto a low-dimensional audible readout (roughly an
utterance level plus a per-word emphasis pattern, "on the order of ten numbers
for this sentence"). If that is true, many different directions produce nearly
the SAME audible consequence, the audio->direction map is many-to-one, and no
probe of any class could invert it. The record flags the claim as inference from
two measurements and names the direct test as NOT RUN:

    "The direct test would be to check whether distinct random directions at
     matched magnitude produce *similar* audible readouts -- energy contour and
     per-word emphasis -- from the same base."

This is that test. K independent random directions from one base at matched
per-row angle, rendered frame-aligned, and their audible readouts compared
AGAINST EACH OTHER rather than against the base.

Design
------
  * base M1, sentence TEXTS[0], `style_dp` pinned at M1's, total_step 8,
    speed 1.05 -- identical to the ray, so the cached word boundaries in
    results/phase2b_prosody/mel_report.json apply unchanged.
  * K = 8 directions, each drawn with the established perturbation model:
    per active row `normalize(P_r + eps*u_r)`, `u_r` uniform on S^255, the 26
    inactive rows left at the base. One independent draw per direction.
  * two magnitudes: eps 0.20 (mid, inside the sampled training range) and
    eps 0.80 (top of it).
  * np.random.seed(RENDER_SEED) before every synthesis, so all clips share one
    vocoder latent and differ only by style.

The three readouts the closure names, compared pairwise:
  * per-word emphasis profile (9 words, utterance level removed),
  * frame energy contour delta over loud frames,
  * the log-mel diff map itself.

Two anchors decide what a correlation means:
  * RELIABILITY CEILING -- the same direction re-rendered under a different
    vocoder seed. Nuisance, not style. Distinct directions cannot be expected to
    agree more than this.
  * NULL -- independent 9-vectors correlate at 0 with sd ~1/sqrt(n-1) = 0.35,
    so a mean pairwise r near 0.35 is indistinguishable from unrelated.

Shared fraction is the summary that does not depend on the null: with profiles
p_k and their mean m, ||m||^2 / mean_k ||p_k||^2 is 1 if every direction
produces the identical readout and ~1/K if they are unrelated.

Usage (from py/):  python3 phase2b_direction_collapse.py [--k 8]
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
from math import comb

from helper import Style, load_text_to_speech, load_voice_style
from phase2b_generate import ACTIVE_ROWS, TEXTS, unit_rows
from phase2b_prosody_analysis import HOP, SR, Canvas, plot_curves, rms
from phase2b_prosody_mel import logmel

OUT_DIR = "results/phase2b_collapse"
LISTEN_DIR = "results/listening_sets/phase2b_direction_collapse"
MEL_REPORT = "results/phase2b_prosody/mel_report.json"
BASE = "M1"
TEXT = TEXTS[0]
EPS_LIST = [0.20, 0.80]
RENDER_SEED = 4242          # the ray's seed: one shared vocoder latent
ALT_SEED = 5151             # second latent, for the reliability ceiling
DIR_SEED = 70000            # direction draws: DIR_SEED + k, independent per k
LOUD_DB = 25.0              # frames within this of the base peak count as loud


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def offdiag(M):
    n = len(M)
    return np.array([M[i, j] for i in range(n) for j in range(i + 1, n)])


def shared_fraction(V):
    """V: (K, d) readouts. ||mean||^2 / mean ||v_k||^2. 1 = identical, ~1/K = unrelated."""
    V = np.asarray(V, float)
    m = V.mean(0)
    return float((m @ m) / max((V ** 2).sum(1).mean(), 1e-12))


def participation_ratio(V):
    """Effective number of dimensions the K readouts span (centred)."""
    Vc = np.asarray(V, float) - np.asarray(V, float).mean(0)
    s = np.linalg.svd(Vc, compute_uv=False) ** 2
    return float(s.sum() ** 2 / max((s ** 2).sum(), 1e-24))


def matrix_fig(path, title, M, labels, vmin, vmax, note):
    """K x K similarity matrix with the numbers printed in the cells."""
    K = len(M)
    cell, L, T = 74, 116, 78
    cv = Canvas(L + K * cell + 40, T + K * cell + 74)
    for i in range(K):
        for j in range(K):
            v = float(M[i, j])
            # diverging: teal for negative, amber for positive, near-white at 0,
            # so "unrelated" reads as blank rather than as a shade of something
            t = np.clip(v / max(abs(vmin), abs(vmax), 1e-9), -1, 1)
            if t >= 0:
                c = (int(250 - 28 * t), int(248 - 96 * t), int(246 - 172 * t))
            else:
                c = (int(250 + 226 * t), int(248 + 140 * t), int(246 + 128 * t))
            c = c if i != j else (222, 222, 220)
            cv.rect(L + j * cell, T + i * cell, L + (j + 1) * cell - 3,
                    T + (i + 1) * cell - 3, c)
            cv.text(L + j * cell + 8, T + i * cell + cell // 2 - 8,
                    ("--" if i == j else f"{v:+.2f}"), (40, 40, 44), 2)
    for i, lb in enumerate(labels):
        cv.text(8, T + i * cell + cell // 2 - 8, lb[:12], (70, 70, 70), 2)
        cv.text(L + i * cell + 8, T - 20, lb[:8], (70, 70, 70), 2)
    cv.text(20, 16, title[:96], (30, 30, 34), 2)
    cv.text(20, 42, note[:110], (110, 110, 110), 2)
    cv.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24,
                    help="directions; K must exceed the readout's dimensionality for "
                         "the participation ratio to be uncensored")
    ap.add_argument("--k-listen", type=int, default=8,
                    help="how many of them to write as level-matched WAVs")
    args = ap.parse_args()
    K = args.k

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)

    bounds = [(w, float(s), float(e))
              for w, s, e in json.load(open(MEL_REPORT))["word_bounds_s"][BASE]]
    words = [w for w, _, _ in bounds]
    print(f"word windows (cached, from the ray's own segmentation): "
          + " ".join(f"{w}[{s:.2f}-{e:.2f}]" for w, s, e in bounds))

    tts = load_text_to_speech("assets/onnx", use_gpu=False)
    base_style = load_voice_style([f"assets/voice_styles/{BASE}.json"])
    base_ttl = base_style.ttl.astype(np.float32)
    dp_ref = load_voice_style(["assets/voice_styles/M1.json"]).dp

    # ---- directions: K independent draws, the established perturbation model
    dirs = []
    for k in range(K):
        g = np.random.default_rng(DIR_SEED + k).standard_normal(
            (len(ACTIVE_ROWS), base_ttl.shape[-1])).astype(np.float32)
        dirs.append(unit_rows(g))
    # how much do the DIRECTIONS themselves overlap? near-orthogonal by design
    Dflat = np.stack([d.reshape(-1) for d in dirs])
    Dn = Dflat / np.linalg.norm(Dflat, axis=1, keepdims=True)
    dir_cos = Dn @ Dn.T
    print(f"\ndirection-space cosine between the {K} draws: "
          f"mean {offdiag(dir_cos).mean():+.4f}  max |{np.abs(offdiag(dir_cos)).max():.4f}| "
          f"(orthogonal by construction in 6144 dims)")

    def perturb(eps, d):
        ttl = base_ttl.copy()
        ttl[0, ACTIVE_ROWS, :] = unit_rows(base_ttl[0, ACTIVE_ROWS, :] + eps * d)
        dev = float(np.abs(np.linalg.norm(ttl, axis=-1) - 1.0).max())
        cos = float((ttl[0, ACTIVE_ROWS] * base_ttl[0, ACTIVE_ROWS]).sum(-1).mean())
        return ttl, dev, float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    def render(ttl, seed=RENDER_SEED):
        np.random.seed(seed)
        wav, dur = tts(TEXT, "en", Style(ttl.astype(np.float32), dp_ref.copy()), 8, 1.05)
        return wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)

    # ---- render ------------------------------------------------------------
    print(f"\nrendering base + {K} directions x {len(EPS_LIST)} magnitudes "
          f"+ {K} reliability re-renders", flush=True)
    clips, meta_rows, worst_dev = {}, [], 0.0
    w_base = render(base_ttl)
    clips["base"] = w_base
    for eps in EPS_LIST:
        for k in range(K):
            ttl, dev, ang = perturb(eps, dirs[k])
            worst_dev = max(worst_dev, dev)
            clips[f"d{k}_eps{eps:.2f}"] = render(ttl)
            clips[f"d{k}_eps{eps:.2f}_altseed"] = render(ttl, ALT_SEED)
            meta_rows.append({"key": f"d{k}_eps{eps:.2f}", "direction": k, "eps": eps,
                              "mean_active_row_angle_deg": ang, "row_norm_max_dev": dev})
        print(f"  eps {eps:.2f}: {K} clips, mean per-row angle "
              f"{np.mean([m['mean_active_row_angle_deg'] for m in meta_rows if m['eps']==eps]):.2f} deg",
              flush=True)
    clips["base_altseed"] = render(base_ttl, ALT_SEED)
    assert worst_dev < 1e-5, f"per-row unit-norm invariant violated: {worst_dev:.2e}"
    print(f"per-row unit-norm invariant: worst deviation {worst_dev:.2e} (< 1e-5 required)")

    lens = {k: v.size for k, v in clips.items()}
    assert len(set(lens.values())) == 1, f"clips are not frame-aligned: {set(lens.values())}"
    print(f"frame alignment: every clip is {next(iter(lens.values()))} samples")

    # ---- readouts ----------------------------------------------------------
    sr = tts.sample_rate
    mel, wav16 = {}, {}
    for k, w in clips.items():
        wav16[k] = load16k_from(w, sr)
        mel[k] = logmel(wav16[k])
    nf = min(m.shape[1] for m in mel.values())
    env_base = mel["base"][:, :nf].mean(0)
    loud = env_base > env_base.max() - LOUD_DB

    def fr(t):
        return int(np.clip(round(t * SR / HOP), 0, nf))

    def per_word_db(key):
        lin = 10.0 ** (mel[key][:, :nf] / 10.0)
        return np.array([10.0 * np.log10(lin[:, fr(s):fr(e)].mean() + 1e-12)
                         for _, s, e in bounds])

    def utt_db(key):
        return float(20 * np.log10(np.sqrt((wav16[key] ** 2).mean())))

    pw0, utt0 = per_word_db("base"), utt_db("base")
    g0 = mel["base"][:, :nf].mean(0)

    def readouts(key, base_key="base"):
        """(emphasis profile in dB after level removal, contour delta, mel diff map)."""
        lvl = utt_db(key) - utt_db(base_key)
        emph = (per_word_db(key) - (pw0 if base_key == "base" else per_word_db(base_key))) - lvl
        D = mel[key][:, :nf] - mel[base_key][:, :nf]
        dg = D.mean(0)[loud]
        return emph, dg - dg.mean(), D[:, loud], lvl

    report = {
        "experiment": "phase2b_direction_collapse",
        "question": ("do K distinct random style_ttl directions at matched per-row angle "
                     "from one base produce SIMILAR audible readouts? (new-plan.md, "
                     "'Still open: whether the collapse account is right')"),
        "base": BASE, "text": TEXT, "k_directions": K, "eps": EPS_LIST,
        "lang": "en", "total_step": 8, "speed": 1.05,
        "style_dp": "held fixed at M1's",
        "render_seed": RENDER_SEED, "alt_seed": ALT_SEED, "direction_seed_base": DIR_SEED,
        "active_rows": ACTIVE_ROWS,
        "words": words, "word_bounds_s": bounds,
        "row_norm_max_dev": worst_dev,
        "n_samples_per_clip": int(next(iter(lens.values()))),
        "direction_cosine_offdiag": {"mean": float(offdiag(dir_cos).mean()),
                                     "max_abs": float(np.abs(offdiag(dir_cos)).max())},
        "per_eps": {}, "reliability_ceiling": {}, "renders": meta_rows,
    }

    for eps in EPS_LIST:
        keys = [f"d{k}_eps{eps:.2f}" for k in range(K)]
        R = [readouts(k_) for k_ in keys]
        E = np.stack([r[0] for r in R])          # (K, n_words) emphasis
        G = np.stack([r[1] for r in R])          # (K, n_loud) contour delta
        M = np.stack([r[2].reshape(-1) for r in R])   # (K, bins*loud) mel diff
        lvl = np.array([r[3] for r in R])

        ce = np.array([[pearson(E[i], E[j]) for j in range(K)] for i in range(K)])
        cg = np.array([[pearson(G[i], G[j]) for j in range(K)] for i in range(K)])
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        cm = Mn @ Mn.T

        # how big is the between-direction difference next to the effect itself?
        pair_rms = [rms(M[i] - M[j]) for i in range(K) for j in range(i + 1, K)]
        own_rms = [rms(M[i]) for i in range(K)]

        row = {
            "utterance_level_db": {"per_direction": lvl.round(3).tolist(),
                                   "mean": float(lvl.mean()), "sd": float(lvl.std()),
                                   "range": float(lvl.max() - lvl.min())},
            "per_word_emphasis_db": {"per_direction": E.round(2).tolist(),
                                     "sd_within_direction": [float(e.std()) for e in E],
                                     "range_within_direction": [float(e.max() - e.min()) for e in E]},
            "pairwise": {
                "emphasis_profile_corr": {"mean": float(offdiag(ce).mean()),
                                          "median": float(np.median(offdiag(ce))),
                                          "min": float(offdiag(ce).min()),
                                          "max": float(offdiag(ce).max()),
                                          "matrix": ce.round(3).tolist()},
                "energy_contour_corr": {"mean": float(offdiag(cg).mean()),
                                        "median": float(np.median(offdiag(cg))),
                                        "min": float(offdiag(cg).min()),
                                        "max": float(offdiag(cg).max()),
                                        "matrix": cg.round(3).tolist()},
                "logmel_diffmap_cosine": {"mean": float(offdiag(cm).mean()),
                                          "median": float(np.median(offdiag(cm))),
                                          "min": float(offdiag(cm).min()),
                                          "max": float(offdiag(cm).max()),
                                          "matrix": cm.round(3).tolist()},
                "diffmap_between_direction_rms_db": float(np.mean(pair_rms)),
                "diffmap_own_rms_db": float(np.mean(own_rms)),
            },
            "shared_fraction": {
                "emphasis": shared_fraction(E),
                "energy_contour": shared_fraction(G),
                "logmel_diffmap": shared_fraction(M),
                "unrelated_reference": 1.0 / K,
            },
            # Effective dimensions the K readouts span. Capped by min(K-1, width)
            # per readout, so each is quoted against its own cap: a low number
            # here means the readout IS low-dimensional, which is compatible
            # with the collapse account's premise -- the pairwise numbers above
            # are what test its conclusion.
            "participation_ratio": {
                "emphasis": participation_ratio(E),
                "emphasis_max": min(K - 1, len(words)),
                "energy_contour": participation_ratio(G),
                "energy_contour_max": min(K - 1, G.shape[1]),
                "logmel_diffmap": participation_ratio(M),
                "logmel_diffmap_max": min(K - 1, M.shape[1]),
            },
        }
        report["per_eps"][f"eps{eps:.2f}"] = row

        print(f"\n=== eps {eps:.2f}, {K} distinct directions, base {BASE} ===")
        print(f"utterance level re base: mean {lvl.mean():+.2f} dB, "
              f"sd {lvl.std():.2f}, range {lvl.max()-lvl.min():.2f}")
        print(f"{'word':<8}" + "".join(f"{f'd{k}':>7}" for k in range(K)))
        for wi, w in enumerate(words):
            print(f"{w:<8}" + "".join(f"{E[k, wi]:>+7.1f}" for k in range(K)))
        print(f"pairwise emphasis-profile r : mean {offdiag(ce).mean():+.3f}  "
              f"median {np.median(offdiag(ce)):+.3f}  "
              f"[{offdiag(ce).min():+.3f}, {offdiag(ce).max():+.3f}]")
        print(f"pairwise energy-contour  r : mean {offdiag(cg).mean():+.3f}  "
              f"[{offdiag(cg).min():+.3f}, {offdiag(cg).max():+.3f}]")
        print(f"pairwise log-mel diffmap cos: mean {offdiag(cm).mean():+.3f}  "
              f"[{offdiag(cm).min():+.3f}, {offdiag(cm).max():+.3f}]")
        print(f"diff-map rms: own {np.mean(own_rms):.2f} dB, "
              f"between two directions {np.mean(pair_rms):.2f} dB")
        print(f"shared fraction  emphasis {row['shared_fraction']['emphasis']:.3f}  "
              f"contour {row['shared_fraction']['energy_contour']:.3f}  "
              f"mel {row['shared_fraction']['logmel_diffmap']:.3f}  "
              f"(1.0 = identical readouts, {1.0/K:.3f} = unrelated)")
        pr = row["participation_ratio"]
        print(f"participation ratio (effective dims the {K} readouts span): "
              f"emphasis {pr['emphasis']:.2f}/{pr['emphasis_max']}  "
              f"contour {pr['energy_contour']:.2f}/{pr['energy_contour_max']}  "
              f"mel {pr['logmel_diffmap']:.2f}/{pr['logmel_diffmap_max']}")

    # ---- reliability ceiling, and the identifiability test --------------------
    # A pairwise correlation near zero only refutes the collapse account if the
    # instrument can resolve anything at all. So the same directions are
    # re-rendered under a second vocoder latent and two things asked:
    #   * reliability -- does one direction's readout reproduce across latents?
    #   * identification -- given a readout from latent B, is its nearest
    #     neighbour among the latent-A readouts the SAME direction? Chance 1/K.
    # Identification above chance means the audible readout carries direction
    # identity, which is precisely what "many-to-one collapse" denies.
    def cos(u, v):
        return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))

    report["reliability_and_identification"] = {}
    for eps in EPS_LIST:
        A = [readouts(f"d{k}_eps{eps:.2f}") for k in range(K)]
        B = [readouts(f"d{k}_eps{eps:.2f}_altseed", "base_altseed") for k in range(K)]
        sims = {}
        for name, f in (("emphasis", lambda r: r[0] - r[0].mean()),
                        ("energy_contour", lambda r: r[1]),
                        ("logmel_diffmap", lambda r: r[2].reshape(-1))):
            C = np.array([[cos(f(A[i]), f(B[j])) for j in range(K)] for i in range(K)])
            nhit = int((np.argmax(C, 1) == np.arange(K)).sum())
            hit = nhit / K
            pval = float(sum(comb(K, t) * (1.0 / K) ** t * (1 - 1.0 / K) ** (K - t)
                             for t in range(nhit, K + 1)))
            rank = float(np.mean([1 + int((C[i] > C[i, i]).sum()) for i in range(K)]))
            sims[name] = {
                "same_direction_across_latents_mean": float(np.mean(np.diag(C))),
                "different_direction_across_latents_mean": float(
                    (C.sum() - np.trace(C)) / (K * (K - 1))),
                "identification_accuracy_1nn": hit,
                "n_correct": nhit, "binomial_p_vs_chance": pval,
                "mean_rank_of_true_direction": rank,
                "chance_accuracy": 1.0 / K, "chance_mean_rank": (K + 1) / 2.0,
                "cross_latent_matrix": C.round(3).tolist(),
            }
        report["reliability_and_identification"][f"eps{eps:.2f}"] = sims
        print(f"\ncross-latent identification, eps {eps:.2f} "
              f"(chance {1/K:.3f}, mean rank {(K+1)/2:.1f})")
        for name, s in sims.items():
            print(f"  {name:<16} same-dir {s['same_direction_across_latents_mean']:+.3f}  "
                  f"diff-dir {s['different_direction_across_latents_mean']:+.3f}  "
                  f"1-NN {s['n_correct']}/{K} = {s['identification_accuracy_1nn']:.3f} "
                  f"(p={s['binomial_p_vs_chance']:.2e})  "
                  f"mean rank {s['mean_rank_of_true_direction']:.2f}")
    n = len(words)  # null reference for the 9-word profile
    report["null_reference"] = {
        "note": f"Pearson r between independent {n}-vectors: mean 0, sd ~1/sqrt(n-1)",
        "sd": float(1.0 / np.sqrt(n - 1)),
    }
    print(f"null reference for a {n}-word profile: r sd = {1/np.sqrt(n-1):.3f}")

    # ---- level-matched listening clips -------------------------------------
    base_rms = float(np.sqrt((clips["base"].astype(np.float64) ** 2).mean()))
    entries = []

    def emit(key, fn, desc, direction, eps):
        w = clips[key]
        gdb = 20 * np.log10(base_rms / max(float(np.sqrt((w.astype(np.float64) ** 2).mean())), 1e-12))
        y = (w * 10.0 ** (gdb / 20.0)).astype(np.float32)
        sf.write(os.path.join(LISTEN_DIR, fn), y, sr)
        entries.append({"file": fn, "description": desc, "direction": direction,
                        "eps": eps, "applied_gain_db": round(float(gdb), 3),
                        "peak": round(float(np.abs(y).max()), 4)})

    emit("base", f"{BASE}_base.wav", f"{BASE} unperturbed, the reference", None, 0.0)
    for eps in EPS_LIST:
        for k in range(min(args.k_listen, K)):
            emit(f"d{k}_eps{eps:.2f}", f"{BASE}_dir{k}_eps{eps:.2f}.wav",
                 f"{BASE} + random direction {k} at eps {eps:.2f}, rms-matched to base",
                 k, eps)

    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump({"experiment": "phase2b_direction_collapse",
                   "base": BASE, "text": TEXT, "lang": "en",
                   "total_step": 8, "speed": 1.05,
                   "style_dp": "held fixed at M1's",
                   "vocoder_seed": RENDER_SEED,
                   "level_matching": "rms-matched to the unperturbed base",
                   "question": report["question"],
                   "outputs": entries}, f, indent=2)

    # ---- figures -----------------------------------------------------------
    figs = []
    segs = [(s, e) for _, s, e in bounds]
    for eps in EPS_LIST:
        Kf = min(args.k_listen, K)
        keys = [f"d{k}_eps{eps:.2f}" for k in range(Kf)]
        E = np.stack([readouts(k_)[0] for k_ in keys])
        x = np.arange(len(words), dtype=float)
        curves = [(f"dir {k}", x, E[k], k) for k in range(Kf)]
        p = os.path.join(OUT_DIR, f"fig_emphasis_profiles_eps{eps:.2f}.png")
        plot_curves(p, f"{BASE}, {Kf} distinct random directions at eps {eps:.2f}: "
                       f"per-word emphasis (dB re base, level removed)",
                    curves, "dB", " ".join(words),
                    segs=[(i - 0.5, i + 0.5) for i in range(len(words))],
                    seg_labels=words)
        figs.append(p)
        cm = np.array([[pearson(E[i], E[j]) for j in range(Kf)] for i in range(Kf)])
        p2 = os.path.join(OUT_DIR, f"fig_emphasis_corr_eps{eps:.2f}.png")
        matrix_fig(p2, f"pairwise emphasis-profile correlation, eps {eps:.2f}",
                   cm, [f"dir {k}" for k in range(Kf)], -1.0, 1.0,
                   f"mean {offdiag(cm).mean():+.2f}; 1.00 = identical readout, "
                   f"0 +- {1/np.sqrt(len(words)-1):.2f} = unrelated")
        figs.append(p2)
    report["figures"] = figs
    report["listening_set"] = LISTEN_DIR

    with open(os.path.join(OUT_DIR, "collapse_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_DIR}/collapse_report.json, {len(figs)} figures, "
          f"and {len(entries)} level-matched clips -> {LISTEN_DIR}")


def load16k_from(w, sr):
    from scipy.signal import resample_poly
    return resample_poly(w, SR, sr).astype(np.float32) if sr != SR else w.astype(np.float32)


if __name__ == "__main__":
    main()
