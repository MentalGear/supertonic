"""Phase 2a -- decensored sibilant ratio: re-validate, then refit.

`phase2a_deessing_direction.py` fit the sibilant ratio
`peak(sibilant frames) / peak(whole clip)` against style_ttl perturbation
coefficients and got held-out R^2 = -0.136: the run correctly halted. The
measure is censored from above -- the denominator INCLUDES the sibilant
frames, so once the sibilant is the loudest thing in the clip the ratio is
exactly 1.0 and cannot report how much further past the rest of the clip it
went. At eps=0.15, 63/120 (52.5%) of the original step-1 draws landed exactly
at 1.0.

This script:
  Task 1 -- decensor the measure (denominator -> peak over NON-sibilant
    frames instead) and re-validate it on the same 20 bench-7 clips
    `phase2a_sibilance.py` already scored (no new renders: reads
    results/phase2a/sibilance.json). Reports both measures' rankings and
    whether decensoring preserves, improves, or degrades separation of the
    listener-flagged clips from the clean ones.
  Task 2 -- rerun step 1 of phase2a_deessing_direction.py (F3, the library
    sentence, speed=1.0 explicit, TOTAL_STEP=8, K=64, seeded) at
    eps in {0.03, 0.07, 0.15}, 80 draws each, with the decensored measure as
    the ridge target. Reports, per eps, the decensored measure's spread, what
    fraction of draws would have saturated (old ratio >= 0.999) under the OLD
    measure, and held-out ridge R^2. Go/no-go stays R^2 > 0.10.

Usage (from py/):
    python3 phase2a_deessing_decensored.py [--smoke]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper import Style, load_text_to_speech, load_voice_style  # noqa: E402
from phase2a_sibilance import analyze_clip  # noqa: E402 -- reused unmodified
from phase2b_generate import ACTIVE_ROWS, LANG, TEXTS, VOICE_STYLE_DIR  # noqa: E402
from phase2b_generate_subspace import build_basis, sample_style  # noqa: E402
from phase2b_probe import fit_ridge, r2_parts, agg_r2  # noqa: E402

OUT_DIR = "results/phase2a_deessing"
AUDIO_DIR = os.path.join(OUT_DIR, "audio_v2")
REPORT_JSON = os.path.join(OUT_DIR, "report_v2.json")
SIBILANCE_JSON = "results/phase2a/sibilance.json"

# ONNX_DIR isn't re-exported cleanly by every module version; resolve it the
# same way phase2a_deessing_direction.py does.
from phase2b_generate import ONNX_DIR  # noqa: E402

SPEED = 1.0
TOTAL_STEP = 8
BASE_PRESET = "F3"
LIBRARY_TEXT_IDX = 4
assert TEXTS[LIBRARY_TEXT_IDX].startswith("The library closes early")

K = 64
EPS_LIST = [0.03, 0.07, 0.15]
N_PER_EPS = 80
TEST_FRAC = 0.25
BASIS_SEED = 20270301           # identical to phase2a_deessing_direction.py
SAMPLE_SEED_BASE = 20270500     # distinct block, one per-eps offset below
VOCODER_SEED_BASE = 20270600    # distinct block from the original script's 20270310+i

R2_GO_THRESHOLD = 0.10
OLD_SATURATION_THRESHOLD = 0.999  # old ratio >= this counts as "saturated"

NAMED_FLAGGED = {
    "pool_F3_t4_seed20361268": "F3 library",
    "pool_F5_t1_seed20361347": "F5 seashells",
    "check_seashells_seed20261449": "M1 seashells",
}


def build_ttl(base_ttl, rows):
    ttl = base_ttl.copy()
    ttl[0, ACTIVE_ROWS, :] = rows.astype(np.float32)
    return ttl


def render(tts, text, ttl, dp, seed):
    np.random.seed(seed)
    wav, dur = tts(text, LANG, Style(ttl, dp.copy()), TOTAL_STEP, SPEED)
    trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
    return trimmed, tts.sample_rate


def both_ratios(result):
    sib = result["sibilant_frames"]["peak"]
    other = result["other_frames"]["peak"]
    whole = result["whole_clip"]["peak"]
    old = sib / max(whole, 1e-12)
    new = sib / max(other, 1e-12)
    return old, new


# --------------------------------------------------------------------------
# Task 1: re-validate on the 20 bench-7 clips already scored by
# phase2a_sibilance.py. No new renders.
# --------------------------------------------------------------------------
def task1_revalidate():
    with open(SIBILANCE_JSON) as f:
        d = json.load(f)

    rows = []
    for c in d["clips"]:
        old, new = both_ratios(c)
        rows.append(dict(stem=c["stem"], verdict=c["verdict"], old=old, new=new))

    def ranked(measure):
        srt = sorted(rows, key=lambda r: -r[measure])
        return [dict(rank=i + 1, stem=r["stem"], verdict=r["verdict"], value=round(r[measure], 4))
                for i, r in enumerate(srt)]

    old_ranked = ranked("old")
    new_ranked = ranked("new")

    old_rank_of = {r["stem"]: r["rank"] for r in old_ranked}
    new_rank_of = {r["stem"]: r["rank"] for r in new_ranked}
    rank_preserved = all(old_rank_of[s] == new_rank_of[s] for s in old_rank_of)

    # strict flagged (per phase2a_sibilance.VERDICTS) vs strict clean
    strict_flagged = [r for r in rows if r["verdict"] == "flagged"]
    strict_clean = [r for r in rows if r["verdict"] == "clean"]

    def spread(rs, key):
        vals = [r[key] for r in rs]
        return dict(min=round(min(vals), 4), max=round(max(vals), 4)) if vals else None

    named_positions = {
        name: dict(
            stem=stem,
            listener_verdict=next(c["verdict"] for c in d["clips"] if c["stem"] == stem),
            old_rank=old_rank_of[stem], old_value=round(next(r["old"] for r in rows if r["stem"] == stem), 4),
            new_rank=new_rank_of[stem], new_value=round(next(r["new"] for r in rows if r["stem"] == stem), 4),
        )
        for stem, name in NAMED_FLAGGED.items()
    }

    return dict(
        n_clips=len(rows),
        old_ranked=old_ranked,
        new_ranked=new_ranked,
        rank_order_identical_old_vs_new=rank_preserved,
        strict_flagged_old_spread=spread(strict_flagged, "old"),
        strict_flagged_new_spread=spread(strict_flagged, "new"),
        strict_clean_old_spread=spread(strict_clean, "old"),
        strict_clean_new_spread=spread(strict_clean, "new"),
        named_clip_positions=named_positions,
        honesty_note=(
            "The prior docstring claim 'flagged 0.975-1.000, clean tops out at 0.646' is "
            "only half right: the 3 strict-flagged clips DO occupy ranks 1-3 of 20 under "
            "both measures, with old-measure values 0.9751-1.0000 (new: 0.9751-1.3270 -- "
            "F3's value moves off the 1.0 ceiling, nothing else changes since it was the "
            "only clip actually saturated). But strict-clean does NOT top out at 0.646: "
            "the real strict-clean maximum is pool_M4_t4_seed20361077 at 0.9587 (old) / "
            "0.9587 (new, unsaturated so unchanged), with pool_F2_t1_seed20361205 at 0.8268 "
            "close behind. So the true old-measure margin between flagged-min (0.9751) and "
            "clean-max (0.9587) is only 0.0164, not the ~0.33 the prior docstring implied. "
            "Rank order is nonetheless clean (flagged strictly occupies ranks 1-3, all 20 "
            "clips), and it is IDENTICAL between the old and new measure -- decensoring "
            "changes only F3's raw value (the one clip that was actually at the ceiling), "
            "moving it from 1.0 to 1.327 without altering any rank. Per the pre-declared "
            "bar: this is preservation, not improvement -- treated as such below, not as "
            "a win. The M1-seashells clip (check_seashells_seed20261449) was flagged in "
            "bench 7, but phase2a_sibilance.py's own VERDICTS classifies it clean_for_"
            "sibilance (its listener flag was primarily time-compression, sibilance only a "
            "secondary remark), and both measures rank it mid-pack (rank 8/20, ~0.59) -- "
            "consistent with that classification, not with the other two flagged clips."
        ),
    )


# --------------------------------------------------------------------------
# Task 2: refit with the decensored target at reduced eps.
# --------------------------------------------------------------------------
def task2_refit(wmodel, tts, n_per_eps):
    base_style = load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{BASE_PRESET}.json")])
    base_ttl = base_style.ttl.astype(np.float32)
    dp_ref = base_style.dp.copy()
    library_text = TEXTS[LIBRARY_TEXT_IDX]

    B, P = build_basis(base_ttl, n_dims=K, seed=BASIS_SEED)
    off_diag = float(np.abs(B @ B.T - np.eye(K)).max())
    print(f"basis orthonormality max|BB^T-I| = {off_diag:.2e}")

    os.makedirs(AUDIO_DIR, exist_ok=True)

    per_eps = {}
    for eps_idx, eps in enumerate(EPS_LIST):
        n = n_per_eps
        print(f"\n=== eps={eps} : {n} draws ===", flush=True)
        rng = np.random.default_rng(SAMPLE_SEED_BASE + eps_idx * 1000)
        records = []
        for i in range(n):
            rows, c_drawn, c_realized, frac = sample_style(rng, B, P, K, eps)
            ttl = build_ttl(base_ttl, rows)
            seed = VOCODER_SEED_BASE + eps_idx * 1000 + i
            wav, sr = render(tts, library_text, ttl, dp_ref, seed)
            stem = f"eps{eps:.2f}_{i:04d}_F3_seed{seed}"
            path = os.path.join(AUDIO_DIR, f"{stem}.wav")
            sf.write(path, wav, sr, subtype="PCM_16")
            result = analyze_clip(wmodel, path)
            old, new = both_ratios(result)
            records.append(dict(i=i, seed=seed, old=old, new=new, c_drawn=c_drawn.tolist(),
                                 in_subspace_fraction=frac))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{n}] old={old:.4f} new={new:.4f}", flush=True)

        old_arr = np.array([r["old"] for r in records])
        new_arr = np.array([r["new"] for r in records])
        sat_frac = float((old_arr >= OLD_SATURATION_THRESHOLD).mean())

        X = np.array([r["c_drawn"] for r in records])
        y = new_arr
        perm = np.random.default_rng(SAMPLE_SEED_BASE + eps_idx * 1000 + 1).permutation(n)
        n_test = max(1, int(round(n * TEST_FRAC)))
        ite, itr = perm[:n_test], perm[n_test:]
        Pte, alpha = fit_ridge(X[itr], y[itr], X[ite])
        ss_res, ss_tot = r2_parts(y[ite], Pte, y[itr].mean(0))
        r2_heldout = agg_r2(ss_res, ss_tot)

        print(f"  saturation_frac(old>=0.999)={sat_frac:.3f}  "
              f"new_spread=[{new_arr.min():.4f},{new_arr.max():.4f}]  "
              f"held-out R^2={r2_heldout:.4f} (alpha={alpha:.3g})")

        per_eps[str(eps)] = dict(
            eps=eps, n=n,
            new_measure_spread=dict(
                mean=float(new_arr.mean()), std=float(new_arr.std()),
                min=float(new_arr.min()), max=float(new_arr.max()),
                p10=float(np.percentile(new_arr, 10)), p50=float(np.percentile(new_arr, 50)),
                p90=float(np.percentile(new_arr, 90)),
            ),
            old_measure_spread=dict(
                mean=float(old_arr.mean()), min=float(old_arr.min()), max=float(old_arr.max()),
            ),
            old_measure_saturation_fraction=sat_frac,
            ridge_fit=dict(r2_heldout_trainmean_baseline=float(r2_heldout), alpha=alpha,
                            n_train=int(len(itr)), n_test=int(len(ite))),
            clears_threshold=bool(r2_heldout > R2_GO_THRESHOLD),
            records=records,
        )

    return per_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_per_eps = 6 if args.smoke else N_PER_EPS

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Task 1: re-validate decensored measure on bench-7 clips ===", flush=True)
    task1 = task1_revalidate()
    print(json.dumps(task1["honesty_note"]))
    for k in ("strict_flagged_old_spread", "strict_flagged_new_spread",
              "strict_clean_old_spread", "strict_clean_new_spread"):
        print(f"  {k}: {task1[k]}")
    print(f"  rank_order_identical_old_vs_new = {task1['rank_order_identical_old_vs_new']}")

    print("\nLoading faster-whisper tiny.en ...", flush=True)
    from faster_whisper import WhisperModel
    wmodel = WhisperModel("Systran/faster-whisper-tiny.en", device="cpu", compute_type="int8",
                           local_files_only=True)

    print("Loading TTS engine ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)

    print("\n=== Task 2: refit at reduced eps with decensored target ===", flush=True)
    task2 = task2_refit(wmodel, tts, n_per_eps)

    any_clears = any(v["clears_threshold"] for v in task2.values())
    if any_clears:
        best_eps = max((v for v in task2.values() if v["clears_threshold"]),
                        key=lambda v: v["ridge_fit"]["r2_heldout_trainmean_baseline"])
        verdict = (f"eps={best_eps['eps']} clears the R^2>{R2_GO_THRESHOLD} bar "
                   f"(R^2={best_eps['ridge_fit']['r2_heldout_trainmean_baseline']:.4f}); "
                   f"descent + guards would proceed there per the brief.")
    else:
        verdict = (f"No eps in {EPS_LIST} clears the pre-declared R^2>{R2_GO_THRESHOLD} bar. "
                   f"Stopping -- no direction fitted, no descent, no transfer test.")
    print("\n" + verdict)

    report = dict(
        experiment="phase2a_deessing_decensored",
        date=datetime.date.today().isoformat(),
        task1_revalidation=task1,
        task2_refit={k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in task2.items()},
        task2_any_eps_clears_threshold=any_clears,
        verdict=verdict,
    )
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
