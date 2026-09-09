"""Bench generator: Phase 0 row locality.

Builds the "Which Rows Carry the Voice" listening bench: row-band hybrids of
M1/F1 `style_ttl`, single-row swaps, and the 50-row per-row delta-share chart.

Published: https://claude.ai/code/artifact/744225a4-dd6a-4b73-a46c-c349c851a2b4
Verdict, inputs, and rebuild notes: docs/LISTENING_BENCHES.md.

Run from `py/` (relative paths below assume that cwd), after generating the
inputs with `phase0_row_locality.py`:

    cd py && python3 benches/phase0_row_locality_bench.py

Requires `py/results/listening_sets/phase0_row_locality/` (WAVs plus
`manifest.json`), gitignored and not checked into the repo — you must
(re)generate it first. Writes `py/results/benches/phase0_row_locality.html`,
itself gitignored: it embeds the WAVs as base64 and runs 2.9 MB+.
"""

import json
import pathlib

from bench_common import PLAYER_PAUSE_SCRIPT, b64

SRC = pathlib.Path("results/listening_sets/phase0_row_locality")
OUT = pathlib.Path("results/benches/phase0_row_locality.html")

m = json.load(open(SRC / "manifest.json"))
prof = m["per_row_profile"]
share = prof["delta_share_pct"]
cos = prof["cos_a_vs_b"]
active = set(prof["active_rows"])

CLIPS = [
    ("endpoint_M1.wav", "M1", "anchor", "Unmodified preset. The base every hybrid starts from."),
    ("endpoint_F1.wav", "F1", "anchor", "Unmodified preset. The destination."),
    ("set_active_from_F1.wav", "24 active rows", "key", "The rows that move across all ten presets, taken from F1. Carries 99.8% of the delta."),
    ("set_inactive_from_F1.wav", "26 frozen rows", "key", "The complement. Carries 0.2% of the delta — and should still sound like M1."),
    ("topk10_from_F1.wav", "top 10 rows", "sparse", "The ten rows holding most of the delta. 61% of it, in a fifth of the grid."),
    ("topk20_from_F1.wav", "top 20 rows", "sparse", "Twenty rows, 98.8% of the delta — where a sparse edit effectively arrives."),
    ("single_row/row15_from_F1.wav", "row 15 only", "single", "The single strongest row. One row of fifty, and it moves."),
    ("single_row/row37_from_F1.wav", "row 37 only", "single", "The deadest row. Constant across every released voice; should do nothing."),
]

GROUPS = [
    ("Anchors", "anchor", "The two presets, unmodified."),
    ("The money pair", "key", "Split by which rows actually move across the ten shipped voices. If the first sounds like F1 and the second still sounds like M1, localization is settled."),
    ("How sparse can it get", "sparse", "Rows ranked by share of the M1→F1 delta, taken from F1 in bulk."),
    ("Single rows", "single", "One row of fifty swapped, strongest against deadest."),
]

by_key = {}
for fn, label, grp, note in CLIPS:
    b64_audio = b64(SRC / fn)
    o = next(x for x in m["outputs"] if x["file"] == fn)
    by_key.setdefault(grp, []).append((label, note, b64_audio, o))

idx = 0
sections = []
for title, key, blurb in GROUPS:
    rows = []
    for label, note, b64_audio, o in by_key[key]:
        idx += 1
        travel = o.get("travel", 0.0)
        sh = o.get("delta_share_pct", 0.0)
        rows.append(f"""        <article class="clip">
          <div class="clip-head">
            <span class="clip-label">{label}</span>
            <span class="clip-stat">travel {travel:.2f} &middot; {sh:.1f}% of delta</span>
          </div>
          <div class="travel" aria-hidden="true"><span style="width:{travel*100:.1f}%"></span></div>
          <audio controls preload="metadata" data-key="{idx}" src="data:audio/wav;base64,{b64_audio}"></audio>
          <p class="note">{note}</p>
        </article>""")
    sections.append(f"""      <section class="group">
        <h3>{title}</h3>
        <p class="group-note">{blurb}</p>
        <div class="clips">
{chr(10).join(rows)}
        </div>
      </section>""")

# chart geometry
W, H = 880, 300
PAD_L, PAD_R, PAD_T, PAD_B = 46, 12, 14, 54
PW = W - PAD_L - PAD_R
PH = H - PAD_T - PAD_B
YMAX = 8.0
slot = PW / 50
bw = slot - 2.6

bars, ticks, rug = [], [], []
for i, v in enumerate(share):
    h = max(2.0, v / YMAX * PH)
    x = PAD_L + i * slot + 1.3
    y = PAD_T + PH - h
    a = "1" if i in active else "0"
    bars.append(
        f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{h:.2f}" rx="2.5" '
        f'data-row="{i}" data-share="{v:.3f}" data-cos="{cos[i]:.4f}" data-active="{a}"></rect>'
    )
    if i in active:
        rug.append(f'<rect class="rug" x="{x:.2f}" y="{PAD_T+PH+9:.2f}" width="{bw:.2f}" height="4" rx="1.5"></rect>')

for t in range(0, 9, 2):
    y = PAD_T + PH - (t / YMAX * PH)
    ticks.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+PW}" y2="{y:.1f}"></line>')
    ticks.append(f'<text class="ytick" x="{PAD_L-9}" y="{y+3.5:.1f}" text-anchor="end">{t}</text>')

xlab = []
for i in list(range(0, 50, 5)) + [49]:
    x = PAD_L + i * slot + slot / 2
    xlab.append(f'<text class="xtick" x="{x:.1f}" y="{PAD_T+PH+30:.1f}" text-anchor="middle">{i}</text>')

CHART = f"""<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of the M1 to F1 style_ttl delta carried by each of the 50 rows. Twenty-four rows carry essentially all of it; the remaining twenty-six sit at the baseline.">
        {''.join(ticks)}
        <line class="axis" x1="{PAD_L}" y1="{PAD_T+PH}" x2="{PAD_L+PW}" y2="{PAD_T+PH}"></line>
        {''.join(bars)}
        {''.join(rug)}
        {''.join(xlab)}
        <text class="axlabel" x="{PAD_L}" y="{H-6}">row index &rarr;</text>
      </svg>"""

HTML = """<title>Which Rows Carry the Voice</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>
  :root {
    color-scheme: light;
    --ground: #F2F5F6;
    --surface: #FFFFFF;
    --sunk: #E4EAEC;
    --ink: #0D1417;
    --ink-2: #46565E;
    --ink-3: #7A8990;
    --rule: #D2DADD;
    --accent: #0E7C73;
    --accent-soft: #D5E9E6;
    --grid: #E7ECEE;
    --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans: "IBM Plex Sans", system-ui, -apple-system, Segoe UI, sans-serif;
    --serif: "Newsreader", Georgia, "Times New Roman", serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --ground: #0B1013; --surface: #141B1F; --sunk: #090D10;
      --ink: #E5EDEF; --ink-2: #9CADB5; --ink-3: #6A7B84;
      --rule: #223037; --accent: #3EB9AD; --accent-soft: #0E312E; --grid: #1C272C;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ground: #0B1013; --surface: #141B1F; --sunk: #090D10;
    --ink: #E5EDEF; --ink-2: #9CADB5; --ink-3: #6A7B84;
    --rule: #223037; --accent: #3EB9AD; --accent-soft: #0E312E; --grid: #1C272C;
  }

  body { background: var(--ground); color: var(--ink); font-family: var(--serif); font-size: 16px; line-height: 1.6; padding: 40px 22px 72px; }
  .wrap { max-width: 920px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }
  .eyebrow { font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); }
  h1 { font-family: var(--sans); font-weight: 600; font-size: clamp(28px, 4.4vw, 40px); letter-spacing: -0.02em; line-height: 1.12; text-wrap: balance; margin: 10px 0 14px; }
  .deck { color: var(--ink-2); max-width: 64ch; margin: 0; }
  h2 { font-family: var(--sans); font-weight: 600; font-size: 15px; margin: 0 0 4px; }
  h3 { font-family: var(--sans); font-weight: 600; font-size: 14px; margin: 0 0 3px; }
  .sect-note, .group-note { font-size: 14.5px; color: var(--ink-3); margin: 0 0 18px; max-width: 64ch; }
  .group-note { margin-bottom: 14px; }

  .headline { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 3px; overflow: hidden; }
  .headline div { background: var(--surface); padding: 14px 16px; }
  .headline .n { font-family: var(--mono); font-size: 25px; font-weight: 500; line-height: 1; color: var(--accent); }
  .headline .k { font-size: 13.5px; color: var(--ink-3); margin-top: 7px; }

  .figure { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 18px 18px 10px; }
  .figure figcaption { font-size: 13.5px; color: var(--ink-3); margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--rule); }
  .chartbox { overflow-x: auto; }
  svg { display: block; width: 100%; min-width: 620px; height: auto; }
  .bar { fill: var(--accent); }
  .bar:hover { fill: var(--ink); }
  .rug { fill: var(--accent); opacity: 0.45; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--rule); stroke-width: 1; }
  .ytick, .xtick, .axlabel { font-family: var(--mono); font-size: 10.5px; fill: var(--ink-3); }
  .axlabel { font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; }
  .legend { display: flex; flex-wrap: wrap; gap: 18px; margin: 12px 0 0; font-size: 13px; color: var(--ink-2); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 11px; height: 11px; border-radius: 2px; background: var(--accent); }
  .swatch.frozen { background: var(--ink-3); }
  .swatch.rugsw { height: 4px; width: 14px; opacity: 0.45; }
  #tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s; background: var(--ink); color: var(--ground); font-family: var(--mono); font-size: 11.5px; line-height: 1.5; padding: 7px 9px; border-radius: 3px; z-index: 9; white-space: pre; }

  .group { display: flex; flex-direction: column; }
  .clips { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
  .clip { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 14px 16px; display: flex; flex-direction: column; gap: 9px; min-width: 0; }
  .clip-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .clip-label { font-family: var(--sans); font-weight: 600; font-size: 14px; }
  .clip-stat { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
  .travel { height: 3px; background: var(--rule); border-radius: 2px; overflow: hidden; }
  .travel span { display: block; height: 100%; background: var(--accent); }
  audio { width: 100%; height: 36px; display: block; }
  .note { font-size: 13.5px; color: var(--ink-3); margin: 0; }

  .findings { display: flex; flex-direction: column; }
  .finding { display: grid; grid-template-columns: 1fr auto; gap: 4px 16px; align-items: baseline; padding: 11px 2px; border-bottom: 1px solid var(--rule); }
  .finding:first-child { border-top: 1px solid var(--rule); }
  .finding-label { font-size: 15px; color: var(--ink-2); }
  .finding-val { font-family: var(--mono); font-size: 13px; font-variant-numeric: tabular-nums; text-align: right; }
  footer { border-top: 1px solid var(--rule); padding-top: 18px; font-size: 14px; color: var(--ink-3); max-width: 68ch; }
  code { font-family: var(--mono); font-size: 0.88em; background: var(--accent-soft); color: var(--ink); padding: 1px 5px; border-radius: 2px; }
  kbd { font-family: var(--mono); font-size: 11px; border: 1px solid var(--rule); border-bottom-width: 2px; border-radius: 3px; padding: 1px 5px; color: var(--ink-2); }
  audio:focus-visible, .bar:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 0 companion &middot; Row locality</div>
    <h1>Which Rows Carry the Voice</h1>
    <p class="deck">The style block is 50 rows of 256 numbers. If what it encodes were spread evenly across them, every
      axis we ever build would have to move all 12,800 parameters at once. It isn't spread evenly. Roughly half the
      rows are frozen solid across every voice Supertone ships.</p>
  </header>

  <section>
    <div class="headline">
      <div><div class="n">24</div><div class="k">rows that move across the ten presets</div></div>
      <div><div class="n">61%</div><div class="k">of the M1&rarr;F1 delta held by ten rows</div></div>
      <div><div class="n">0.2%</div><div class="k">held by the other twenty-six combined</div></div>
      <div><div class="n">0.916</div><div class="k">correlation, tensor delta to audio movement</div></div>
    </div>
  </section>

  <section>
    <h2>The profile</h2>
    <p class="sect-note">Each bar is one row's share of the total M1&rarr;F1 difference. Hover any bar for its
      numbers.</p>
    <figure class="figure">
      <div class="chartbox">
        __CHART__
      </div>
      <div class="legend">
        <span><i class="swatch rugsw"></i>Marked below the axis: the 24 rows that move across all ten shipped presets</span>
      </div>
      <figcaption>Share of the M1&rarr;F1 delta, by row. Bars at the baseline are drawn at a 2&nbsp;px minimum so a
        near-zero row stays distinguishable from a gap; those rows sit around 0.006% each. Two things to read here.
        The tall bars are scattered rather than clustered, which is why contiguous row bands turned out to be the
        wrong way to cut this grid. And the marker strip &mdash; derived independently, from spread across all ten
        presets &mdash; lands under almost every tall bar. Rows 5 and 13 are the interesting disagreement: marked
        active across the preset library, yet barely involved in this particular pair.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The listening set</h2>
    <p class="sect-note">Every clip is M1's style block with the named rows replaced verbatim by F1's, same sentence
      and settings throughout, vocoder RNG seeded so nothing differs but the tensor. Press <kbd>1</kbd>&ndash;<kbd>8</kbd>
      to play; starting one stops the others. <em>Travel</em> is how far the audio moved from M1 toward F1.</p>
    <div style="display:flex; flex-direction:column; gap:26px;">
__SECTIONS__
    </div>
  </section>

  <section>
    <h2>What the numbers say</h2>
    <p class="sect-note">All of this is measurable without listening. The clips are the confirmation, not the
      evidence.</p>
    <div class="findings">
      <div class="finding"><span class="finding-label">Per-row cosine similarity between M1 and F1</span><span class="finding-val">0.796 &ndash; 0.9999</span></div>
      <div class="finding"><span class="finding-label">Ten rows carry most of the delta (uniform would be 20%)</span><span class="finding-val">61%</span></div>
      <div class="finding"><span class="finding-label">Single-row swaps that move the audio less than 2%</span><span class="finding-val">21 of 50</span></div>
      <div class="finding"><span class="finding-label">Strongest single row reaches only part way to F1</span><span class="finding-val">row 15 &middot; 0.30</span></div>
      <div class="finding"><span class="finding-label">Twenty rows get almost all the way there</span><span class="finding-val">0.78 of 1.00</span></div>
      <div class="finding"><span class="finding-label">Worst deviation from unit row norm, across all 66 tensors</span><span class="finding-val">2.4e-07</span></div>
    </div>
  </section>

  <section>
    <h2>What this changes downstream</h2>
    <p class="sect-note" style="max-width:66ch;">Derive future axes on the active rows and hold the rest fixed: a
      6,144-parameter fit instead of 12,800, better conditioned, and &ldquo;change age without changing identity&rdquo;
      gets a structural handle rather than needing orthogonalization after the fact. Two cautions. Swapping only the
      frozen rows still moved the audio a little, so they are not free to hard-zero without a listening check. And no
      single row is the gender switch &mdash; the strongest one gets less than a third of the way &mdash; so this is
      about twenty scattered components, not one, which cuts against the Eigenvoice precedent the plan cites.</p>
  </section>

  <footer>
    Scope: one voice pair, one sentence, one set of inference settings, and a spectral distance that saturates &mdash;
    single-row distances sum to far more than the endpoint distance, so <em>travel</em> ranks clips, it does not measure
    a fraction of the way. The frozen/active split is the one finding here drawn from all ten presets rather than from
    M1 and F1 alone.
  </footer>
</div>
<div id="tip" role="status" aria-live="polite"></div>

<script>
__PLAYER_PAUSE_SCRIPT__
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    var i = parseInt(e.key, 10);
    if (i >= 1 && i <= players.length) {
      var t = players[i - 1];
      if (t.paused) { t.play(); } else { t.pause(); }
      t.scrollIntoView({ block: 'nearest' });
    }
  });

  var tip = document.getElementById('tip');
  document.querySelectorAll('.bar').forEach(function (b) {
    b.addEventListener('mouseenter', function () {
      tip.textContent = 'row ' + b.dataset.row + '\\n' + b.dataset.share + '% of delta\\ncos ' +
        b.dataset.cos + '\\n' + (b.dataset.active === '1' ? 'active' : 'frozen');
      tip.style.opacity = '1';
    });
    b.addEventListener('mousemove', function (e) {
      tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 130) + 'px';
      tip.style.top = (e.clientY + 16) + 'px';
    });
    b.addEventListener('mouseleave', function () { tip.style.opacity = '0'; });
  });
</script>
"""

HTML = (HTML.replace("__CHART__", CHART)
            .replace("__SECTIONS__", "\n".join(sections))
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print("wrote", OUT, OUT.stat().st_size, "bytes")
