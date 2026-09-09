"""Bench generator: Phase 2b direction collapse / WavLM.

Builds "Narrow, Not Many-to-One": the eight-direction orthogonality set that
tests whether distinct style_ttl directions collapse to one audible effect,
plus the WavLM/ECAPA/train-mean probe-recovery triples for three held-out
voices.

Published: https://claude.ai/code/artifact/b9d634e6-78cd-4190-940f-830adb910346
Verdict, inputs, and rebuild notes: docs/LISTENING_BENCHES.md.

Run from `py/` (relative paths below assume that cwd), after generating the
inputs with the Phase 2b direction-collapse / probe-recovery pipeline:

    cd py && python3 benches/phase2b_direction_collapse_bench.py

Requires `py/results/listening_sets/phase2b_direction_collapse/`,
`py/results/listening_sets/phase2b_probe_recovery_wavlm/` (WAVs), and
`py/results/phase2b_collapse/` (figures) — all gitignored and not checked
into the repo, so you must (re)generate them first. Writes
`py/results/benches/phase2b_direction_collapse.html`, itself gitignored: it
embeds WAVs and PNGs as base64 and runs several MB.
"""

import pathlib

from bench_common import PLAYER_PAUSE_SCRIPT, b64

COL = pathlib.Path("results/listening_sets/phase2b_direction_collapse")
REC = pathlib.Path("results/listening_sets/phase2b_probe_recovery_wavlm")
FIG = pathlib.Path("results/phase2b_collapse")
OUT = pathlib.Path("results/benches/phase2b_direction_collapse.html")


def clip(path, label, stat, note, anchor=False):
    return f"""          <article class="clip{' anchor' if anchor else ''}">
            <div class="clip-head"><span class="clip-label">{label}</span><span class="clip-stat">{stat}</span></div>
            <audio controls preload="metadata" src="data:audio/wav;base64,{b64(path)}"></audio>
            <p class="note">{note}</p>
          </article>"""


# --- Job 1: direction collapse ---
DIRS_80 = [("dir0", "r = -0.17 with dir2"), ("dir2", "r = -0.01 with dir7"),
           ("dir7", "r = -0.54 with dir0"), ("dir4", "another draw")]
j1_80 = "\n".join(clip(COL / f"M1_{d}_eps0.80.wav", d, "38.5&deg;", n) for d, n in DIRS_80)
j1_20 = "\n".join(clip(COL / f"M1_{d}_eps0.20.wav", d, "11.3&deg;", "Smaller step, same direction as above.")
                  for d in ("dir0", "dir2"))
j1_base = clip(COL / "M1_base.wav", "M1 base", "unperturbed", "The common starting point for every direction.", True)

# --- Job 2: probe recovery triples ---
SAMPLES = [
    ("01200", "F5", "Nine hungry travellers waited quietly under the old stone bridge.",
     [("true", "true style", "ground truth", "What the engine actually rendered."),
      ("ecapa_pred", "ECAPA predicted", "cos 0.875 &middot; 15.9 dB", "R&sup2; 0.067."),
      ("wavlm_pred", "WavLM predicted", "cos 0.900 &middot; 15.7 dB", "R&sup2; 0.142 &mdash; twice ECAPA's."),
      ("trainmean_baseline", "fixed average voice", "cos 0.873 &middot; 19.2 dB", "Ignores the audio entirely &mdash; same style tensor every time. The deciding arm.")]),
    ("01201", "F5", "Bright yellow flowers grew along the muddy path near the river.",
     [("true", "true style", "ground truth", "What the engine actually rendered."),
      ("ecapa_pred", "ECAPA predicted", "cos 0.884 &middot; 18.6 dB", "R&sup2; 0.067."),
      ("wavlm_pred", "WavLM predicted", "cos 0.901 &middot; 14.1 dB", "WavLM's clearest win of the three."),
      ("trainmean_baseline", "fixed average voice", "cos 0.873 &middot; 19.7 dB", "Ignores the audio entirely. Identical style tensor to the one below.")]),
    ("01202", "M5", "Please call the doctor before the meeting starts at noon.",
     [("true", "true style", "ground truth", "What the engine actually rendered."),
      ("ecapa_pred", "ECAPA predicted", "cos 0.902 &middot; 21.5 dB", "Here ECAPA edges WavLM on cosine."),
      ("wavlm_pred", "WavLM predicted", "cos 0.901 &middot; 19.0 dB", "Closer in dB, level on cosine."),
      ("trainmean_baseline", "fixed average voice", "cos 0.887 &middot; 22.3 dB", "Ignores the audio entirely. Identical style tensor to the one above.")]),
]
j2 = []
for idx, base, text, arms in SAMPLES:
    cells = "\n".join(clip(REC / f"eps0.20_idx{idx}_{k}.wav", lbl, st, nt, k == "true") for k, lbl, st, nt in arms)
    j2.append(f"""      <div class="ladder">
        <h3>Held-out voice {base} &mdash; &ldquo;{text}&rdquo;</h3>
        <div class="clips four">
{cells}
        </div>
      </div>""")

FIGS = [
    ("fig_emphasis_profiles_eps0.80.png", "Eight directions, eight signatures",
     "The refuted account predicts one curve. There are eight, spreading about 8 dB on &ldquo;dog&rdquo;."),
    ("fig_emphasis_corr_eps0.80.png", "Pairwise correlation, and mostly blank",
     "Every pair of directions, correlated against every other. Mean +0.06. The blankness is the result."),
]
figs = "\n".join(
    f"""      <figure class="fig">
        <img src="data:image/png;base64,{b64(FIG / f)}" alt="{t}">
        <figcaption><strong>{t}.</strong> {c}</figcaption>
      </figure>""" for f, t, c in FIGS)

HTML = """<title>Narrow, Not Many-to-One</title>
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
  .ladders { display:flex; flex-direction:column; gap:22px; }
  .clips { display:grid; grid-template-columns:repeat(auto-fit,minmax(205px,1fr)); gap:12px; }
  .clips.four { grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:13px 15px; display:flex; flex-direction:column; gap:9px; min-width:0; }
  .clip.anchor { background:var(--sunk); }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:13px; }
  .clip-stat { font-family:var(--mono); font-size:10px; color:var(--ink-3); font-variant-numeric:tabular-nums; }
  audio { width:100%; height:36px; display:block; }
  .note { font-size:12.5px; color:var(--ink-3); margin:0; }
  .ask { font-size:14.5px; color:var(--ink-2); margin:14px 0 0; }
  .fig { margin:0; background:var(--surface); border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  .fig img { display:block; width:100%; height:auto; }
  .fig figcaption { font-size:13.5px; color:var(--ink-3); padding:12px 16px; border-top:1px solid var(--rule); }
  .figs { display:flex; flex-direction:column; gap:16px; }
  .findings { display:flex; flex-direction:column; }
  .finding { display:grid; grid-template-columns:1fr auto; gap:4px 16px; align-items:baseline; padding:11px 2px; border-bottom:1px solid var(--rule); }
  .finding:first-child { border-top:1px solid var(--rule); }
  .finding-label { font-size:15px; color:var(--ink-2); }
  .finding-val { font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; text-align:right; }
  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:68ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2b &middot; Second correction</div>
    <h1>Narrow, Not Many-to-One</h1>
    <p class="deck">The record explained Phase 2b's null by saying many different style directions sound nearly alike,
      and flagged the direct test as unrun. The test has now been run. They do not sound alike &mdash; they sound
      near-orthogonal, and a machine can tell which is which. The null survives for a different reason, and the fix
      for the encoder changes with it.</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li><strong>Do the eight directions sound like eight different things?</strong> Same voice, same sentence,
          same step size &mdash; only the direction differs. Listen for which <em>words</em> carry the stress.</li>
        <li><strong>In the recovery triples, can you pick out the true speaker?</strong> Compare each prediction against
          the <em>fixed average voice</em> &mdash; a predictor that ignores the audio and always outputs the same style.
          The question is not which sounds better, but whether either prediction sounds like <em>the same person</em> as
          the true clip.</li>
      </ol>
      <p class="why">Question 1 is already answered by measurement and your ear is the check on it. Question 2 is the
        one that decides whether Phase 2b stays closed: if neither probe beats a fixed average at identifying the
        speaker, no audio encoder of this class recovers voice identity from a single utterance.</p>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Recorded as inference, now refuted by measurement</p>
      <p class="old">Many different high-dimensional directions produce nearly the same low-dimensional audible
        consequence, so the inverse is many-to-one.</p>
      <p><strong>The inverse is narrow, not many-to-one.</strong> Per utterance and per base, the audio exposes on the
        order of 5&ndash;10 direction-specific dimensions out of 6,120 &mdash; and those dimensions are cleanly
        direction-specific rather than collapsed. Inverting a ~7-dimensional readout perfectly still recovers about
        0.1% of the target, which is why the null stands.</p>
      <p style="margin-top:12px;">This matters for the encoder. The limit is dimensional and <strong>additive across
        utterances</strong>, not a property of the representation &mdash; so the lever for Phase 2a is
        <strong>many utterances per style</strong>, not a better single-utterance encoder. That is a different
        instruction from the one the refuted account licensed.</p>
    </div>
  </section>

  <section>
    <h2>Eight directions from one voice</h2>
    <p class="taskline"><strong>Your task</strong>Do these sound like eight different things, or eight versions of one
      thing? Listen for which words carry the stress.</p>
    <p class="sect-note">Same base, same sentence, same seed, same per-row angle &mdash; only the direction differs,
      and the directions are mutually orthogonal in style space. All level-matched to the base, since magnitude alone
      buys loudness. If the refuted account were right these would be hard to tell apart.</p>
    <div class="ladders">
      <div class="ladder">
        <h3>eps 0.80 &mdash; 38.5&deg; per row</h3>
        <div class="clips">
__J1BASE__
__J180__
        </div>
        <p class="ask">Listen for <em>which</em> words carry the stress. The measurement says these emphasis patterns
          correlate at +0.06 on average &mdash; 55% of all pairs fall inside the null band &mdash; and that two
          directions differ from each other by more than either differs from the base.</p>
      </div>
      <div class="ladder">
        <h3>eps 0.20 &mdash; 11.3&deg; per row, the same two directions</h3>
        <div class="clips">
__J120__
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2>The numbers behind that</h2>
    <div class="findings">
      <div class="finding"><span class="finding-label">Pairwise emphasis correlation between directions (276 pairs)</span><span class="finding-val">+0.16 / +0.06</span></div>
      <div class="finding"><span class="finding-label">Pairs falling inside the null band</span><span class="finding-val">55%</span></div>
      <div class="finding"><span class="finding-label">Each direction moves the log-mel from base</span><span class="finding-val">4.8 / 9.6 dB</span></div>
      <div class="finding"><span class="finding-label">Two directions differ from <em>each other</em> by</span><span class="finding-val">5.5 / 10.4 dB</span></div>
      <div class="finding"><span class="finding-label">Picking the right direction from 24, across a fresh vocoder draw</span><span class="finding-val">70.8%, chance 4.2%</span></div>
      <div class="finding"><span class="finding-label">Effective dimensionality of the audible readout</span><span class="finding-val">~5 &ndash; 7 of 6,120</span></div>
    </div>
  </section>

  <section>
    <h2>Seen, not summarised</h2>
    <div class="figs">
__FIGS__
    </div>
  </section>

  <section>
    <h2>Does twice the R&sup2; sound like anything?</h2>
    <p class="taskline"><strong>Your task</strong>Not &ldquo;which sounds better&rdquo; &mdash; which sounds like
      <em>the same person</em> as the true clip. Judge each prediction against the fixed average voice.</p>
    <p class="sect-note">WavLM scores roughly double ECAPA. Here is what each actually reconstructed, for voices held
      out of training entirely &mdash; same text and seed, only the style tensor differs. The fourth arm decides it:
      a predictor that ignores the input audio completely and always emits the average of the seven training voices.
      It still renders as speech, so you will hear a voice &mdash; the point is that it is the <em>same</em> voice on
      every sample, whoever the true speaker was. That is why it can sound male where the true voice is female:
      averaging seven mixed-gender presets lands where it lands.</p>
    <div class="ladders">
__J2__
    </div>
    <p class="ask">For scale: a full M1&rarr;F1 identity swap is 16.6 dB. Both predictions sit 14&ndash;21 dB from the
      true style &mdash; as far as, or further than, a different speaker. Expect a blander, generic voice rather than
      the right voice with error on it. If neither probe beats the no-audio baseline at identifying the speaker, the
      closure is confirmed by ear.</p>
  </section>

  <section>
    <h2>An admission worth recording</h2>
    <p class="sect-note" style="max-width:66ch;">The within-family residual control &mdash; R&sup2; of +0.0000, cited
      in the record as decisive &mdash; is exactly what <em>both</em> accounts predict. It never had the power to tell
      them apart, so it cannot be evidence for collapse, and it is not evidence against recoverability either. The
      grain of truth in the old account survives in one place: the <em>energy contour</em> genuinely is the shared part
      across directions. It was the per-word emphasis pattern and the spectral detail that turned out to be
      direction-specific. The right list of readouts, and the wrong claim about them.</p>
  </section>

  <footer>
    Scope. One base voice and one sentence, so the dimensionality figure is per base, per sentence. At eps 0.20 the
    nine-number emphasis profile sits near the vocoder-noise floor, which is why identification is weaker there; the
    log-mel diff map is the strong instrument and is unambiguous at both magnitudes. Sixteen of the twenty-four
    measured directions are not rendered here &mdash; the measurement uses all of them, the bench caps at what stays
    A/B-able. And <code>style_dp</code> remains pinned throughout, so speech rate is still outside all of this.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
"""

HTML = (HTML.replace("__J1BASE__", j1_base).replace("__J180__", j1_80)
            .replace("__J120__", j1_20).replace("__J2__", "\n".join(j2)).replace("__FIGS__", figs)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print("wrote", OUT, OUT.stat().st_size, "bytes")
