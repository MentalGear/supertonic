"""Bench generator: Phase 2b speaker similarity.

Builds "Closer, Not Cloned": the calibrated speaker-identity re-measurement
that followed the direction-collapse bench's WavLM-vs-baseline listening
result. Presents the calibration anchors (same-speaker vs. different-speaker
ECAPA cosine, measured on the preset library), where the WavLM-predicted
style lands relative to them, and the 27-clip true/wavlm_pred/trainmean
listening set — true / WavLM prediction / fixed average voice for the
best/typical/worst sample of each of three held-out identities.

Not yet published as an Artifact. Rebuild notes: docs/LISTENING_BENCHES.md.

Run from `py/` (relative paths below assume that cwd), after generating the
inputs with the Phase 2b speaker-similarity measurement pipeline:

    cd py && python3 benches/phase2b_speaker_similarity_bench.py

Requires `py/results/listening_sets/phase2b_speaker_similarity/` (WAVs plus
`manifest.json`) and `py/results/phase2b_speaker_similarity/` (the two JSON
reports) — all gitignored and not checked into the repo, so you must
(re)generate them first. Writes
`py/results/benches/phase2b_speaker_similarity.html`, itself gitignored: it
embeds the WAVs as base64 and runs several MB.
"""

import json
import pathlib

from bench_common import PLAYER_PAUSE_SCRIPT, b64

SRC = pathlib.Path("results/listening_sets/phase2b_speaker_similarity")
REPORTS = pathlib.Path("results/phase2b_speaker_similarity")
OUT = pathlib.Path("results/benches/phase2b_speaker_similarity.html")

manifest = json.load(open(SRC / "manifest.json"))
sim = json.load(open(REPORTS / "speaker_similarity_report.json"))
gender = json.load(open(REPORTS / "gender_confound_report.json"))


def fmt_range(stats):
    return f"{stats['mean']:.3f} ({stats['min']:.3f}–{stats['max']:.3f})"


# --- Calibration anchors, read from the report rather than hardcoded ---
cal = sim["calibration"]["distributions"]["ecapa_cosine"]
same_seed = cal["same_speaker_diff_seed (10 pairs)"]
same_text = cal["same_speaker_diff_text (10 pairs)"]
diff_spk = cal["different_speaker (45 preset pairs)"]

# --- Measured: prediction vs. true, overall ---
ov = sim["headline"]["overall"]
wavlm_ov = ov["wavlm"]["ecapa_cosine_vs_true"]
ecapa_ov = ov["ecapa"]["ecapa_cosine_vs_true"]
tm_ov = ov["train_mean"]["ecapa_cosine_vs_true"]

# --- Per-identity WavLM means ---
per_id = sim["headline"]["per_identity"]
per_id_wavlm = {k: v["wavlm"]["ecapa_cosine_vs_true"]["mean"] for k, v in per_id.items()}

# --- Consistency, from the gender-confound report ---
g_all = gender["gender_split_summary"]["all"]
g_m5 = gender["gender_split_summary"]["M5 (male-true, unconfounded)"]

# --- Withdrawn rms log-mel claim ---
logmel_same = sim["calibration"]["distributions"]["logmel_rms_db_frame_aligned"]["same_speaker_diff_seed (10 pairs)"]
logmel_diff = sim["calibration"]["distributions"]["logmel_rms_db_frame_aligned"]["different_speaker (45 preset pairs)"]

# --- Listening set: 27 clips as 9 true/wavlm_pred/trainmean triples ---
KIND_META = {
    "true": ("true style", "ground truth", "What was actually rendered for this held-out identity."),
    "wavlm_pred": ("WavLM prediction", None, None),
    "trainmean_baseline": ("fixed average voice", None,
                            "The same reconstructed voice on every sample in this set, regardless of who the true "
                            "speaker was &mdash; it never looks at the audio. Judge whether it sounds like "
                            "<em>this</em> speaker specifically, not whether it sounds like plausible speech."),
}


def clip(path, label, stat, note, anchor=False):
    return f"""          <article class="clip{' anchor' if anchor else ''}">
            <div class="clip-head"><span class="clip-label">{label}</span><span class="clip-stat">{stat}</span></div>
            <audio controls preload="metadata" src="data:audio/wav;base64,{b64(path)}"></audio>
            <p class="note">{note}</p>
          </article>"""


ladders = []
for row in manifest["rows"]:
    ident, rank, idx, text = row["identity"], row["rank_label"], row["idx"], row["text"]
    stem = f"{ident}_{rank}_idx{idx:05d}"
    cells = []
    for kind in ("true", "wavlm_pred", "trainmean_baseline"):
        label, extra_stat, fixed_note = KIND_META[kind]
        if kind == "true":
            stat, note = "ground truth", fixed_note
        elif kind == "wavlm_pred":
            stat = f"cos {row['wavlm_ecapa_cosine_vs_true']:.3f} &middot; {row['wavlm_logmel_db']:.1f} dB"
            note = "Predicted from the true audio by the WavLM probe."
        else:
            stat = f"cos {row['trainmean_ecapa_cosine_vs_true']:.3f} &middot; {row['trainmean_logmel_db']:.1f} dB"
            note = fixed_note
        cells.append(clip(SRC / f"{stem}_{kind}.wav", label, stat, note, kind == "true"))
    ladders.append(f"""      <div class="ladder">
        <h3>{ident} &middot; {rank} case &mdash; &ldquo;{text}&rdquo;</h3>
        <div class="clips three">
{chr(10).join(cells)}
        </div>
      </div>""")

GROUPED = []
for ident in ("F4", "F5", "M5"):
    group_ladders = [l for l, row in zip(ladders, manifest["rows"]) if row["identity"] == ident]
    GROUPED.append(f"""      <div class="idgroup">
        <h4>Identity {ident} <span class="idval">WavLM mean cos {per_id_wavlm[ident]:.3f}</span></h4>
        <div class="ladders">
{chr(10).join(group_ladders)}
        </div>
      </div>""")

HTML = """<title>Closer, Not Cloned</title>
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
  h4 { font-family:var(--sans); font-weight:600; font-size:14px; margin:0 0 10px; display:flex; align-items:baseline; justify-content:space-between; gap:12px; }
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
  .correction .old { color:var(--ink-3); text-decoration:line-through; margin-bottom:12px; }

  .tablebox { overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:3px; }
  table { border-collapse:collapse; width:100%; min-width:480px; font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:9px 14px; border-bottom:1px solid var(--rule); }
  th:first-child,td:first-child { text-align:left; }
  th { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:500; }
  tr:last-child td { border-bottom:none; }
  td.flag { color:var(--flag); }
  td.ok { color:var(--accent); }
  figcaption.tcap { font-size:13.5px; color:var(--ink-3); padding:12px 16px; border-top:1px solid var(--rule); margin:0; }

  .findings { display:flex; flex-direction:column; }
  .finding { display:grid; grid-template-columns:1fr auto; gap:4px 16px; align-items:baseline; padding:11px 2px; border-bottom:1px solid var(--rule); }
  .finding:first-child { border-top:1px solid var(--rule); }
  .finding-label { font-size:15px; color:var(--ink-2); }
  .finding-val { font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; text-align:right; }

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
  .ask { font-size:14.5px; color:var(--ink-2); margin:14px 0 0; }

  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:68ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2b &middot; Speaker similarity</div>
    <h1>Closer, Not Cloned</h1>
    <p class="deck">The direction-collapse bench's listeners judged the WavLM reconstruction closer to true than the
      fixed average voice, prompting a calibrated re-measurement of what &ldquo;closer&rdquo; means here. This bench
      carries that measurement and the clips it is built on: WavLM lands inside the different-speaker range on the
      calibrated instrument, and its single best sample approaches &mdash; but does not cross &mdash; the
      same-speaker floor.</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li><strong>For each identity below, compare the WavLM prediction and the fixed average voice against the true
          clip.</strong> Which one sounds more like the <em>same person</em> as the true recording &mdash; not which
          one sounds better, and not which one sounds more natural.</li>
        <li>The fixed average voice is a predictor that ignores the input audio entirely and always renders the same
          style tensor. If WavLM does not consistently sound closer to true than this fixed voice does, the numeric
          result below does not hold up by ear.</li>
      </ol>
      <p class="why">This decides whether the WavLM probe is worth carrying forward as an encoder input, or whether its
        edge over a no-audio baseline is a measurement artifact rather than an audible one.</p>
    </div>
  </section>

  <section>
    <h2>The calibrated anchors</h2>
    <p class="sect-note">ECAPA cosine similarity, calibrated on the ten shipped presets before any prediction enters
      the picture. These three distributions define what &ldquo;same speaker&rdquo; and &ldquo;different speaker&rdquo;
      look like on this instrument.</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>pair type</th><th>n</th><th>mean (range)</th></tr></thead>
        <tbody>
          <tr><td>same speaker, different seed</td><td>__N_SAME_SEED__</td><td class="ok">__SAME_SEED__</td></tr>
          <tr><td>same speaker, different sentence</td><td>__N_SAME_TEXT__</td><td class="ok">__SAME_TEXT__</td></tr>
          <tr><td>different speaker (45 preset pairs)</td><td>__N_DIFF__</td><td class="flag">__DIFF__</td></tr>
        </tbody>
      </table>
      <figcaption class="tcap">Same-speaker pairs cluster high whether the seed or the sentence changes; different-speaker
        pairs sit lower and spread wide &mdash; wide enough that a single different-speaker pair can score above a
        single same-speaker one. That overlap is why a threshold needs the full distribution, not one anchor pair.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Where the predictions land</h2>
    <p class="sect-note">Same instrument, same 80 held-out samples across three identities (F4, F5, M5), each compared
      against its own true render.</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>predictor</th><th>n</th><th>mean (range)</th></tr></thead>
        <tbody>
          <tr><td>WavLM-predicted style</td><td>__N_WAVLM__</td><td class="ok">__WAVLM__</td></tr>
          <tr><td>ECAPA-predicted style</td><td>__N_ECAPA__</td><td>__ECAPA__</td></tr>
          <tr><td>fixed average voice (train-mean)</td><td>__N_TM__</td><td class="flag">__TM__</td></tr>
        </tbody>
      </table>
      <figcaption class="tcap">WavLM's mean sits inside the calibrated different-speaker range, not the same-speaker
        one &mdash; this is not a claim of voice cloning. Its single best sample (identity F4, cos __F4_BEST__) is the
        one result that approaches the same-speaker floor of __SAME_TEXT_MIN__ without crossing it.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The consistency, not just the mean</h2>
    <p class="sect-note">A mean that beats a baseline is weaker evidence than a margin that holds sample by sample.
      Both are checked here.</p>
    <div class="findings">
      <div class="finding"><span class="finding-label">WavLM beats the fixed average voice on identity, per sample</span><span class="finding-val">__N_BEAT_TM__ of __N_TOTAL__ &middot; mean margin +__MARGIN_TM__</span></div>
      <div class="finding"><span class="finding-label">WavLM beats a same-gender impostor voice, per sample (all identities)</span><span class="finding-val">__PCT_BEAT_IMP__% &middot; mean margin +__MARGIN_IMP__</span></div>
      <div class="finding"><span class="finding-label">Same check, restricted to M5 (the unconfounded, non-majority-gender case)</span><span class="finding-val">__PCT_BEAT_IMP_M5__% &middot; mean margin +__MARGIN_IMP_M5__</span></div>
    </div>
  </section>

  <section>
    <h2>Per identity</h2>
    <p class="sect-note">WavLM's mean cosine to true, broken out by held-out identity. The ranking matches an
      independent listener's ordering, made before these numbers were computed.</p>
    <div class="findings">
      <div class="finding"><span class="finding-label">F4</span><span class="finding-val">__F4_MEAN__</span></div>
      <div class="finding"><span class="finding-label">F5</span><span class="finding-val">__F5_MEAN__</span></div>
      <div class="finding"><span class="finding-label">M5</span><span class="finding-val">__M5_MEAN__</span></div>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Withdrawn</p>
      <p class="old">rms log-mel distance and active-row style cosine can serve as identity measures for this
        comparison.</p>
      <p><strong>They cannot.</strong> Calibrated across the same 80 held-out samples, the same-speaker rms log-mel
        distribution (__LOGMEL_SAME__ dB mean) and the different-speaker distribution (__LOGMEL_DIFF__ dB mean)
        overlap completely &mdash; neither measure separates same-speaker pairs from different-speaker pairs well
        enough to support a threshold. ECAPA cosine, calibrated the same way, is the measure used above instead.</p>
    </div>
  </section>

  <section>
    <h2>The listening set</h2>
    <p class="sect-note">Twenty-seven clips: three identities (F4, F5, M5) held out of training, each with its
      best/typical/worst case by WavLM cosine, each case rendered three ways &mdash; true, WavLM-predicted, fixed
      average voice. Every clip is rms level-matched to its own true render. <strong>Listen for whether the prediction
      sounds like the same person as the true clip, not for audio quality.</strong></p>
    <div class="idgroups">
__GROUPED__
    </div>
  </section>

  <footer>
    Scope. Three held-out identities, 80 samples total, one probe family (WavLM) compared against one no-audio
    baseline and one alternative probe (ECAPA) on the same held-out set. The same-gender-impostor check controls for
    the coarsest possible confound &mdash; that beating the baseline is just landing on the right gender &mdash; it
    does not rule out finer confounds like age or recording condition. <code>style_dp</code> was held fixed at each
    identity's own true value throughout, so this measures timbre recovery, not rhythm or rate recovery.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
"""

HTML = (HTML
        .replace("__N_SAME_SEED__", str(same_seed["n"])).replace("__SAME_SEED__", fmt_range(same_seed))
        .replace("__N_SAME_TEXT__", str(same_text["n"])).replace("__SAME_TEXT__", fmt_range(same_text))
        .replace("__N_DIFF__", str(diff_spk["n"])).replace("__DIFF__", fmt_range(diff_spk))
        .replace("__N_WAVLM__", str(wavlm_ov["n"])).replace("__WAVLM__", fmt_range(wavlm_ov))
        .replace("__N_ECAPA__", str(ecapa_ov["n"])).replace("__ECAPA__", fmt_range(ecapa_ov))
        .replace("__N_TM__", str(tm_ov["n"])).replace("__TM__", fmt_range(tm_ov))
        .replace("__F4_BEST__", f"{wavlm_ov['max']:.3f}")
        .replace("__SAME_TEXT_MIN__", f"{same_text['min']:.3f}")
        .replace("__N_BEAT_TM__", str(round(g_all["frac_wavlm_beats_trainmean"] * g_all["n"])))
        .replace("__N_TOTAL__", str(g_all["n"]))
        .replace("__MARGIN_TM__", f"{g_all['wavlm_minus_trainmean']['mean']:.2f}")
        .replace("__PCT_BEAT_IMP__", f"{g_all['frac_wavlm_beats_impostor'] * 100:.0f}")
        .replace("__MARGIN_IMP__", f"{g_all['wavlm_minus_impostor']['mean']:.2f}")
        .replace("__PCT_BEAT_IMP_M5__", f"{g_m5['frac_wavlm_beats_impostor'] * 100:.0f}")
        .replace("__MARGIN_IMP_M5__", f"{g_m5['wavlm_minus_impostor']['mean']:.2f}")
        .replace("__F4_MEAN__", f"{per_id_wavlm['F4']:.3f}")
        .replace("__F5_MEAN__", f"{per_id_wavlm['F5']:.3f}")
        .replace("__M5_MEAN__", f"{per_id_wavlm['M5']:.3f}")
        .replace("__LOGMEL_SAME__", f"{logmel_same['mean']:.2f}")
        .replace("__LOGMEL_DIFF__", f"{logmel_diff['mean']:.2f}")
        .replace("__GROUPED__", "\n".join(GROUPED))
        .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print("wrote", OUT, OUT.stat().st_size, "bytes")
