"""Bench generator: Phase 2a capacity (what an R^2 sounds like).

Phase 2a (`phase2b_subspace_probe.py`) measured how well a WavLM-fed ridge
probe recovers a K-dimensional style_ttl subspace perturbation from audio,
reporting R^2 = 0.912 / 0.608 / 0.232 at K = 4 / 16 / 64 (eps=0.20). Nobody
had heard what those numbers mean. This bench renders, for the best/typical/
worst test-split sample at each K, a level-matched, seed-matched triple:

  1. true style       -- the actual perturbed style tensor for that sample
  2. probe prediction  -- base + (predicted c) @ B[:K], then unit_rows
  3. base preset       -- M1, unperturbed -- the "recovered nothing" control

This script both regenerates the audio (fitting the same ridge probe as
`phase2b_subspace_probe.py`, byte-for-byte the same R^2 numbers) and writes
the bench page, unlike most other bench generators in this directory, which
only render HTML from an already-rendered listening set. Everything Phase 2a
needs -- manifest, subspace basis, WavLM features, the probe report -- lives
under `results/phase2b_subspace/` and `results/phase2a/`, all already
present from the Phase 2a measurement run.

Run from `py/`:
    cd py && python3 benches/phase2a_capacity_bench.py

Requires `results/phase2b_subspace/{manifest.json,subspace.npz,wavlm_feats.npz}`
and `results/phase2a/{subspace_probe.json,subspace_probe_matched.json}`, an
ONNX engine under `assets/onnx/`, and voice styles under `assets/voice_styles/`
-- all present after `phase2b_generate_subspace.py` + `phase2b_subspace_embed.py`
+ `phase2b_subspace_probe.py` have been run. Writes WAVs + manifest.json to
`results/listening_sets/phase2a_capacity/` and the page to
`results/benches/phase2a_capacity.html` (gitignored, embeds WAVs as base64).
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

from bench_common import PLAYER_PAUSE_SCRIPT, b64

# helper.py, phase2b_probe.py, phase2b_subspace_probe.py live in py/, one level up
# from this benches/ script -- unlike the bench_common-only generators, this one
# also re-fits the probe, so it needs py/ on sys.path regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import Style, load_text_to_speech, load_voice_style  # noqa: E402
from phase2b_probe import fit_ridge  # noqa: E402
from phase2b_subspace_probe import build_xy  # noqa: E402

SUBSPACE_DIR = "results/phase2b_subspace"
REPORT_PATH = "results/phase2a/subspace_probe.json"
MATCHED_REPORT_PATH = "results/phase2a/subspace_probe_matched.json"
LISTEN_DIR = "results/listening_sets/phase2a_capacity"
OUT_HTML = "results/benches/phase2a_capacity.html"
VOICE_STYLE_DIR = "assets/voice_styles"
ONNX_DIR = "assets/onnx"
LAYERS = [3, 4, 5]
K_LIST = [4, 16, 64]
RANKS = [("best", -1), ("typical", "median"), ("worst", 0)]


def unit_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def level_match(w, ref_rms):
    r = float(np.sqrt((w.astype(np.float64) ** 2).mean()))
    g = ref_rms / max(r, 1e-12)
    return (w * g).astype(np.float32), 20 * np.log10(g)


def style_cos(a_active, b_active):
    num = (a_active * b_active).sum(-1)
    den = np.linalg.norm(a_active, axis=-1) * np.linalg.norm(b_active, axis=-1)
    return float((num / den.clip(min=1e-12)).mean())


def main():
    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    with open(os.path.join(SUBSPACE_DIR, "manifest.json")) as f:
        meta = json.load(f)
    records = meta["records"]
    subspace = np.load(os.path.join(SUBSPACE_DIR, "subspace.npz"))
    feats = np.load(os.path.join(SUBSPACE_DIR, "wavlm_feats.npz"))
    report = json.load(open(REPORT_PATH))
    matched_report = json.load(open(MATCHED_REPORT_PATH))

    B = subspace["basis"].astype(np.float64)  # (256, 6144)
    base_ttl_full = subspace["base_ttl"].astype(np.float32)  # (50, 256)
    active_rows = subspace["active_rows"].tolist()
    P = base_ttl_full[active_rows, :].astype(np.float64)  # (24, 256)

    texts = meta["texts"]
    lang, total_step, speed = meta["lang"], meta["total_step"], meta["speed"]
    base_preset = meta["base_preset"]
    dp_ref = load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{base_preset}.json")]).dp

    print("Loading TTS engine ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    sr = tts.sample_rate
    base_style_ttl = base_ttl_full[None].astype(np.float32)  # (1, 50, 256), unperturbed M1

    def render(text, ttl_1_50_256, seed):
        np.random.seed(seed)  # vocoder latent RNG -- shared across a triple, see docstring
        wav, dur = tts(text, lang, Style(ttl_1_50_256.astype(np.float32), dp_ref.copy()),
                        total_step, speed)
        return wav[0, : int(sr * dur[0].item())].astype(np.float32)

    k_summaries = {}
    triples_meta = []

    for K in K_LIST:
        recs_k = [r for r in records if int(r["K"]) == K]
        X, Y, rms_pre, itr, ite, n_matched, missing = build_xy(K, recs_k, feats, subspace, LAYERS)
        assert missing == 0, f"K={K}: {missing} manifest records missing from wavlm_feats.npz"
        kept = recs_k  # aligned 1:1 with X/Y rows (build_xy filters recs_k by feats presence; missing==0 above)

        Xtr, Xte = X[itr], X[ite]
        Ytr, Yte = Y[itr], Y[ite]
        Ypred, alpha = fit_ridge(Xtr, Ytr, Xte)

        baseline = Yte.mean(0)  # same baseline phase2b_subspace_probe.py uses for the reported R^2
        ss_res = ((Yte - Ypred) ** 2).sum(1)
        ss_tot = ((Yte - baseline) ** 2).sum(1)
        r2_sample = 1.0 - ss_res / ss_tot

        comp_r2_mean = float((1.0 - ((Yte - Ypred) ** 2).sum(0) / ((Yte - baseline) ** 2).sum(0)).mean())
        rep_val = report["results"][str(K)]["per_component_r2_mean"]
        assert abs(comp_r2_mean - rep_val) < 1e-6, (
            f"K={K}: re-fit R^2 {comp_r2_mean} does not match report {rep_val} -- probe fit is not reproducing"
        )

        order = np.argsort(r2_sample)
        picks = {"worst": int(order[0]), "typical": int(order[len(order) // 2]), "best": int(order[-1])}

        c_true_all = subspace[f"c_realized_K{K}"]
        rows_this_k = []
        for rank_label in ("best", "typical", "worst"):
            pos = picks[rank_label]
            row_i = int(ite[pos])
            rec = kept[row_i]
            idx = int(rec["idx"])
            text = texts[rec["text_idx"]]
            seed = int(rec["seed"])
            sample_r2 = float(r2_sample[pos])

            true_ttl = subspace[f"ttl_K{K}"][idx][None].astype(np.float32)

            c_pred = Ypred[pos]
            d_pred = (c_pred @ B[:K]).reshape(len(active_rows), 256)
            rows_pred = unit_rows(P + d_pred)
            pred_ttl_full = base_ttl_full.copy()
            pred_ttl_full[active_rows, :] = rows_pred.astype(np.float32)
            pred_ttl = pred_ttl_full[None]

            cos_pred = style_cos(true_ttl[0, active_rows, :], pred_ttl[0, active_rows, :])
            cos_base = style_cos(true_ttl[0, active_rows, :], base_style_ttl[0, active_rows, :])

            wav_true = render(text, true_ttl, seed)
            wav_pred = render(text, pred_ttl, seed)
            wav_base = render(text, base_style_ttl, seed)
            n = min(len(wav_true), len(wav_pred), len(wav_base))
            wav_true, wav_pred, wav_base = wav_true[:n], wav_pred[:n], wav_base[:n]

            ref_rms = float(np.sqrt((wav_true.astype(np.float64) ** 2).mean()))
            lvl_pred, gdb_pred = level_match(wav_pred, ref_rms)
            lvl_base, gdb_base = level_match(wav_base, ref_rms)

            stem = f"K{K}_{rank_label}_idx{idx:05d}"
            sf.write(os.path.join(LISTEN_DIR, f"{stem}_true.wav"), wav_true, sr, subtype="PCM_16")
            sf.write(os.path.join(LISTEN_DIR, f"{stem}_pred.wav"), lvl_pred, sr, subtype="PCM_16")
            sf.write(os.path.join(LISTEN_DIR, f"{stem}_base.wav"), lvl_base, sr, subtype="PCM_16")

            row = {
                "K": K, "rank_label": rank_label, "idx": idx, "text": text, "seed": seed,
                "sample_r2": sample_r2, "style_cosine_pred_vs_true": cos_pred,
                "style_cosine_base_vs_true": cos_base,
                "gain_db_pred": gdb_pred, "gain_db_base": gdb_base,
                "duration_sec": n / sr,
            }
            rows_this_k.append(row)
            triples_meta.append(row)
            print(f"K={K:<3} {rank_label:8s} idx={idx:5d} sample_r2={sample_r2:+.4f} "
                  f"style_cos(pred)={cos_pred:.4f} style_cos(base)={cos_base:.4f} seed={seed}", flush=True)

        k_summaries[K] = {
            "n_test": int(len(ite)),
            "achieved_mean_r2": report["results"][str(K)]["per_component_r2_mean"],
            "oracle_ceiling_r2": report["results"][str(K)]["oracle_ceiling"]["r2_testmean_baseline"],
            "shuffled_control_r2": report["results"][str(K)]["shuffled_target_control_r2"],
            "loudness_control_r2": report["results"][str(K)]["loudness_control_r2"],
            "sample_r2_best": rows_this_k[0]["sample_r2"],
            "sample_r2_typical": rows_this_k[1]["sample_r2"],
            "sample_r2_worst": rows_this_k[2]["sample_r2"],
        }

    manifest_out = {
        "experiment": "phase2a_capacity_bench",
        "subspace_source": SUBSPACE_DIR,
        "report_source": REPORT_PATH,
        "layers_1based": LAYERS,
        "k_list": K_LIST,
        "per_sample_r2_definition": (
            "1 - sum_j(Yte[i,j]-Ypred[i,j])^2 / sum_j(Yte[i,j]-Yte.mean(0)[j])^2 for test sample i, summed over the "
            "K subspace coefficients j. Uses the same baseline (the test-set coefficient mean) as the official "
            "per_component_r2 / pooled R^2 in results/phase2a/subspace_probe.json -- summing this per-sample "
            "quantity's numerator and denominator separately over all test samples reproduces that pooled R^2 "
            "exactly. This is a per-sample decomposition of the same metric, not a different one."
        ),
        "level_matching": "pred and base clips rms-matched to the true render of their own triple",
        "vocoder_seed": "np.random.seed(record's original corpus seed) immediately before each of the 3 renders "
                         "in a triple, so the seed is identical within a triple and differs across triples",
        "k_summaries": {str(k): v for k, v in k_summaries.items()},
        "rows": triples_meta,
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"\nWrote {len(triples_meta)} rows (x3 clips) -> {LISTEN_DIR}")

    build_html(k_summaries, triples_meta, matched_report, report)


def build_html(k_summaries, triples_meta, matched_report, report):
    def clip(path, label, stat, note, anchor=False):
        return f"""          <article class="clip{' anchor' if anchor else ''}">
            <div class="clip-head"><span class="clip-label">{label}</span><span class="clip-stat">{stat}</span></div>
            <audio controls preload="metadata" src="data:audio/wav;base64,{b64(path)}"></audio>
            <p class="note">{note}</p>
          </article>"""

    RANK_ORDER = {"best": 0, "typical": 1, "worst": 2}
    RANK_TITLE = {
        "best": "Best case", "typical": "Typical case", "worst": "Worst case",
    }

    k_sections = []
    for K in K_LIST:
        s = k_summaries[K]
        rows_k = sorted([r for r in triples_meta if r["K"] == K], key=lambda r: RANK_ORDER[r["rank_label"]])
        ladders = []
        for row in rows_k:
            stem = f"K{K}_{row['rank_label']}_idx{row['idx']:05d}"
            cells = [
                clip(f"{LISTEN_DIR}/{stem}_true.wav", "true style", "ground truth",
                     "The actual perturbed style tensor for this sample -- what the probe was asked to recover.",
                     anchor=True),
                clip(f"{LISTEN_DIR}/{stem}_pred.wav", "probe prediction",
                     f"sample R&sup2; {row['sample_r2']:+.3f}",
                     "The style tensor reconstructed from the probe's predicted subspace coefficients: "
                     "base + (predicted c) &middot; B[:K], renormalized."),
                clip(f"{LISTEN_DIR}/{stem}_base.wav", "the base voice with no perturbation applied",
                     "control",
                     "M1 unperturbed. The control: if the prediction is not clearly closer to true than this is, "
                     "the probe recovered nothing audible here, regardless of its R&sup2;."),
            ]
            ladders.append(f"""      <div class="ladder">
        <h3>{RANK_TITLE[row['rank_label']]} &middot; sample R&sup2; {row['sample_r2']:+.3f} &middot; &ldquo;{row['text']}&rdquo;</h3>
        <div class="clips three">
{chr(10).join(cells)}
        </div>
      </div>""")

        k_sections.append(f"""      <div class="idgroup">
        <h4>K = {K} <span class="idval">achieved R&sup2; {s['achieved_mean_r2']:.3f} &middot; oracle ceiling {s['oracle_ceiling_r2']:.3f} &middot; n_test={s['n_test']}</span></h4>
        <p class="taskline"><strong>Restated</strong>What does R&sup2;&nbsp;=&nbsp;{s['achieved_mean_r2']:.3f} sound like at K={K}? Compare
          prediction against true and against the no-perturbation control below.</p>
        <div class="ladders">
{chr(10).join(ladders)}
        </div>
      </div>""")

    summary_rows = "\n".join(
        f'          <tr><td>K={K}</td><td>{k_summaries[K]["n_test"]}</td>'
        f'<td class="ok">{k_summaries[K]["achieved_mean_r2"]:.3f}</td>'
        f'<td>{k_summaries[K]["oracle_ceiling_r2"]:.3f}</td>'
        f'<td>{k_summaries[K]["shuffled_control_r2"]:+.3f}</td>'
        f'<td>{k_summaries[K]["loudness_control_r2"]:+.3f}</td></tr>'
        for K in K_LIST
    )

    m4 = matched_report["results"]["4"]["per_component_r2_mean"]
    m16 = matched_report["results"]["16"]["per_component_r2_mean"]
    m64 = report["results"]["64"]["per_component_r2_mean"]  # K=64's default eps IS 0.20, no separate matched run

    HTML = """<title>Capacity Bench</title>
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
  .wrap { max-width:940px; margin:0 auto; display:flex; flex-direction:column; gap:40px; }
  .eyebrow { font-family:var(--mono); font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); }
  h1 { font-family:var(--sans); font-weight:600; font-size:clamp(28px,4.4vw,40px); letter-spacing:-.02em; line-height:1.12; text-wrap:balance; margin:10px 0 14px; }
  .deck { color:var(--ink-2); max-width:64ch; margin:0; }
  h2 { font-family:var(--sans); font-weight:600; font-size:15px; margin:0 0 4px; }
  h3 { font-family:var(--sans); font-weight:600; font-size:13.5px; margin:0 0 11px; color:var(--ink-2); }
  h4 { font-family:var(--sans); font-weight:600; font-size:14px; margin:0 0 10px; display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; }
  .idval { font-family:var(--mono); font-size:11px; color:var(--accent); font-weight:500; font-variant-numeric:tabular-nums; }
  .sect-note { font-size:14.5px; color:var(--ink-3); margin:0 0 18px; max-width:64ch; }
  .task { background:var(--accent-soft); border:1px solid var(--accent); border-radius:3px; padding:18px 20px; }
  .task .tag { font-family:var(--mono); font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent); margin:0 0 9px; font-weight:500; }
  .task ol { margin:0; padding-left:20px; color:var(--ink); }
  .task li { margin-bottom:7px; }
  .task li:last-child { margin-bottom:0; }
  .task .why { margin:12px 0 0; font-size:14px; color:var(--ink-2); }
  .taskline { background:var(--accent-soft); border-left:3px solid var(--accent); border-radius:2px; padding:11px 14px; margin:0 0 16px; font-size:14.5px; color:var(--ink); }
  .taskline strong { font-family:var(--sans); font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); display:block; margin-bottom:4px; }
  .correction { background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--flag); border-radius:3px; padding:18px 20px; }
  .correction .was { font-family:var(--mono); font-size:11.5px; color:var(--flag); letter-spacing:.06em; text-transform:uppercase; margin:0 0 7px; }
  .correction p { margin:0; }

  .tablebox { overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:3px; }
  table { border-collapse:collapse; width:100%; min-width:480px; font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:9px 14px; border-bottom:1px solid var(--rule); }
  th:first-child,td:first-child { text-align:left; }
  th { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:500; }
  tr:last-child td { border-bottom:none; }
  td.ok { color:var(--accent); }
  figcaption.tcap { font-size:13.5px; color:var(--ink-3); padding:12px 16px; border-top:1px solid var(--rule); margin:0; }

  .idgroups { display:flex; flex-direction:column; gap:28px; }
  .idgroup { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:16px 18px 18px; }
  .ladders { display:flex; flex-direction:column; gap:16px; }
  .ladder + .ladder { border-top:1px solid var(--rule); padding-top:16px; }
  .clips { display:grid; grid-template-columns:repeat(auto-fit,minmax(205px,1fr)); gap:12px; }
  .clips.three { grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:13px 15px; display:flex; flex-direction:column; gap:9px; min-width:0; }
  .idgroup .clip { background:var(--ground); }
  .clip.anchor { background:var(--sunk); }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:13px; }
  .clip-stat { font-family:var(--mono); font-size:10px; color:var(--ink-3); font-variant-numeric:tabular-nums; }
  audio { width:100%; height:36px; display:block; }
  .note { font-size:12.5px; color:var(--ink-3); margin:0; }

  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:68ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2a &middot; Capacity</div>
    <h1>What 0.91 Sounds Like</h1>
    <p class="deck">Phase 2a measured how well a WavLM probe recovers a K-dimensional style perturbation from audio,
      as R&sup2; against the true subspace coefficients: 0.912 at K=4, 0.608 at K=16, 0.232 at K=64. Nobody has heard
      what those numbers mean. This bench renders the true style, the probe's prediction, and an unperturbed control
      side by side, so the R&sup2; ladder can be judged by ear instead of taken on faith.</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li><strong>For each triple below, compare the probe prediction to the true clip and to the control.</strong>
          Does the prediction sound like a recognizable step toward the true clip, or does it sound like the
          unperturbed control &mdash; i.e. like nothing was recovered?</li>
        <li>This is <em>not</em> "which sounds better." The control is not a competitor; it is the null hypothesis.
          A prediction that is indistinguishable from the control means the R&sup2; above it is not audible, however
          large the number looks.</li>
      </ol>
      <p class="why">This decides whether the R&sup2; numbers Phase 2a is using as its capacity metric correspond to
        audible recovery, and at what value on the K=4/16/64 ladder recovery stops being audible.</p>
    </div>
  </section>

  <section>
    <h2>The R&sup2; ladder</h2>
    <p class="sect-note">Achieved R&sup2; is the mean per-component test-set R&sup2; from
      <code>results/phase2a/subspace_probe.json</code> &mdash; the same number Phase 2a reports. Oracle ceiling is the
      best any linear readout of the training targets could do at this K (near 1.0 confirms the shortfall below it is
      a real audio-readout limit, not a training-set-size artifact). Shuffled and loudness controls should sit near 0;
      if they do not, the achieved number is contaminated.</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>K</th><th>n_test</th><th>achieved R&sup2;</th><th>oracle ceiling</th><th>shuffled control</th><th>loudness control</th></tr></thead>
        <tbody>
__SUMMARY_ROWS__
        </tbody>
      </table>
      <figcaption class="tcap">Oracle ceilings sit close to 1.0 at every K, so every shortfall below them is
        something the estimator was able to express and did not &mdash; rank starvation, the artifact that
        invalidated Phase 2b, is ruled out here. That is all they rule out. They do <em>not</em> establish that
        the decline from K=4 to K=64 is a limit on what the audio carries: the amplitude-matched control below
        shows most of it is per-direction amplitude, and the K=4 to K=16 step additionally holds n_train at 240
        while the target quadruples, which costs estimation efficiency even where rank is not binding.
        Shuffled-target and loudness controls sit at or below 0 at every K.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The clips</h2>
    <p class="sect-note">Three test-split samples per K &mdash; best, typical, and worst by per-sample R&sup2;
      (defined in <code>manifest.json</code>: the same test-mean-baseline decomposition behind the official R&sup2;,
      computed per sample instead of pooled) &mdash; so the range Phase 2a measured is represented, not just its
      average. Every clip is rendered on the same text with the same diffusion settings; the prediction and control
      clips are rms level-matched to their triple's true clip, and all three clips in a triple share one vocoder
      RNG seed, so what you hear differs only by style, not by sampling noise.</p>
    <div class="idgroups">
__K_SECTIONS__
    </div>
  </section>

  <section>
    <h2>The amplitude-matched control</h2>
    <p class="sect-note">A separate run held per-direction perturbation amplitude fixed across K instead of holding
      total perturbation energy fixed (which spreads the same energy over more directions as K grows, shrinking each
      one). Under matched amplitude the K-decline mostly disappears:</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>condition</th><th>R&sup2;</th></tr></thead>
        <tbody>
          <tr><td>K=4, eps=0.05</td><td class="ok">__M4__</td></tr>
          <tr><td>K=16, eps=0.10</td><td class="ok">__M16__</td></tr>
          <tr><td>K=64, eps=0.20 (default corpus, no separate matched run needed)</td><td class="ok">__M64__</td></tr>
        </tbody>
      </table>
      <figcaption class="tcap">This showed the K-decline in the headline table above is largely a per-direction
        amplitude effect, not evidence of a hard dimensional limit on what the audio can carry &mdash; at matched
        per-direction amplitude, R&sup2; stays roughly flat across K instead of collapsing from 0.91 to 0.23.</figcaption>
    </figure>
  </section>

  <section>
    <div class="correction">
      <p class="was">Scope</p>
      <p>Every style rendered for this bench is engine-generated. &ldquo;True style&rdquo; means the tensor
        constructed for the corpus and rendered from directly &mdash; a perturbed <code>style_ttl</code> the script
        built, not a recorded human voice. The base preset (M1) and the perturbation subspace are both synthetic
        constructions; nothing here is a real speaker.</p>
    </div>
  </section>

  <footer>
    Scope. Text and <code>style_dp</code> are held fixed within every triple, so what varies is <code>style_ttl</code>
    alone. Best/typical/worst are chosen by a per-sample decomposition of the test-set R&sup2; (see
    <code>manifest.json</code>), not by a separate metric &mdash; ranking by this quantity is self-consistent with
    the headline number but is not itself the number Phase 2a reports. Predictions are reconstructed from the ridge
    probe's coefficients exactly as the corpus's own realized styles were built (project onto B[:K], add to base,
    <code>unit_rows</code>), so any prediction error already includes the probe's out-of-subspace loss alongside its
    in-subspace error. Active-row style cosine was computed for every clip here but is not shown: bench 5 calibrated
    that same metric and found it does not separate same-speaker from different-speaker pairs, so displaying it next
    to a clip would present an uncalibrated number as evidence. The sample R&sup2; above each prediction is the
    calibrated quantity under test.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
"""

    HTML = (HTML
            .replace("__SUMMARY_ROWS__", summary_rows)
            .replace("__K_SECTIONS__", "\n".join(k_sections))
            .replace("__M4__", f"{m4:.3f}")
            .replace("__M16__", f"{m16:.3f}")
            .replace("__M64__", f"{m64:.3f}")
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
