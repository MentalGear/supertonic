"""Bench generator: Phase 0 linearity gate.

Builds the "M1 -> F1 Linearity Gate" listening bench: five renders of one
sentence with `style_ttl` interpolated between the M1 and F1 presets at
weights 0.00/0.25/0.50/0.75/1.00, plus the endpoint identity checks.

Published: https://claude.ai/code/artifact/f12d5550-0804-4cc0-8ab4-c7c551aad253
Verdict, inputs, and rebuild notes: docs/LISTENING_BENCHES.md.

Run from `py/` (relative paths below assume that cwd), after generating the
inputs with `phase0_linearity_gate.py`:

    cd py && python3 benches/phase0_linearity_gate_bench.py

Requires `py/results/listening_sets/phase0_linearity/interp_w*.wav`, which
are gitignored and not checked into the repo — you must (re)generate them
first. Writes `py/results/benches/phase0_linearity_gate.html`, itself
gitignored: it embeds the WAVs as base64 and runs 1.8 MB+.
"""

import pathlib

from bench_common import PLAYER_PAUSE_SCRIPT, b64

SRC = pathlib.Path("results/listening_sets/phase0_linearity")
OUT = pathlib.Path("results/benches/phase0_linearity_gate.html")

WEIGHTS = ["0.00", "0.25", "0.50", "0.75", "1.00"]
ROLES = {
    "0.00": ("anchor", "reference &middot; M1", "Preset M1, untouched."),
    "0.25": ("blend", "blend", "Quarter step toward F1."),
    "0.50": ("blend", "blend", "Midpoint &mdash; the hardest case."),
    "0.75": ("blend", "blend", "Three quarters toward F1."),
    "1.00": ("anchor", "reference &middot; F1", "Preset F1, recovered through the blend."),
}

rows = []
for i, w in enumerate(WEIGHTS):
    b64_audio = b64(SRC / f"interp_w{w}.wav")
    cls, tag, note = ROLES[w]
    pct = float(w) * 100
    rows.append(f"""      <article class="row {cls}">
        <div class="row-head">
          <span class="w">{w}</span>
          <span class="tag">{tag}</span>
        </div>
        <div class="stack">
          <div class="meter" aria-hidden="true"><span style="width:{pct}%"></span></div>
          <audio controls preload="metadata" data-key="{i+1}" src="data:audio/wav;base64,{b64_audio}"></audio>
          <p class="note">{note}</p>
        </div>
      </article>""")

HTML = """<title>M1 &rarr; F1 Linearity Gate</title>
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
    --pass: #2C6B45;
    --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans: "IBM Plex Sans", system-ui, -apple-system, Segoe UI, sans-serif;
    --serif: "Newsreader", Georgia, "Times New Roman", serif;
  }
  :root:not([data-theme="light"]) { }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --ground: #0B1013;
      --surface: #141B1F;
      --sunk: #090D10;
      --ink: #E5EDEF;
      --ink-2: #9CADB5;
      --ink-3: #6A7B84;
      --rule: #223037;
      --accent: #3EB9AD;
      --accent-soft: #0E312E;
      --pass: #63B98A;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ground: #0B1013;
    --surface: #141B1F;
    --sunk: #090D10;
    --ink: #E5EDEF;
    --ink-2: #9CADB5;
    --ink-3: #6A7B84;
    --rule: #223037;
    --accent: #3EB9AD;
    --accent-soft: #0E312E;
    --pass: #63B98A;
  }

  body {
    background: var(--ground);
    color: var(--ink);
    font-family: var(--serif);
    font-size: 16px;
    line-height: 1.6;
    padding: 40px 22px 72px;
  }
  .wrap { max-width: 880px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }

  .eyebrow {
    font-family: var(--mono);
    font-size: 11.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
  }
  h1 {
    font-family: var(--sans);
    font-weight: 600;
    font-size: clamp(28px, 4.4vw, 40px);
    letter-spacing: -0.02em;
    line-height: 1.12;
    text-wrap: balance;
    margin: 10px 0 14px;
  }
  .deck { color: var(--ink-2); max-width: 63ch; margin: 0; }
  h2 {
    font-family: var(--sans);
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.01em;
    margin: 0 0 4px;
  }
  .sect-note { font-size: 14.5px; color: var(--ink-3); margin: 0 0 20px; max-width: 62ch; }

  .spec {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
    gap: 1px;
    background: var(--rule);
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
  }
  .spec div { background: var(--surface); padding: 12px 14px; }
  .spec dt {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 5px;
  }
  .spec dd {
    font-family: var(--mono);
    font-size: 13.5px;
    color: var(--ink);
    margin: 0;
    font-variant-numeric: tabular-nums;
    word-break: break-word;
  }

  .sweep { display: flex; flex-direction: column; gap: 2px; }
  .row {
    display: grid;
    grid-template-columns: 96px 1fr;
    gap: 0 22px;
    align-items: start;
    padding: 18px 20px;
    background: var(--surface);
    border: 1px solid var(--rule);
  }
  .row + .row { border-top: none; }
  .row:first-of-type { border-radius: 3px 3px 0 0; }
  .row:last-of-type { border-radius: 0 0 3px 3px; }
  .row.anchor { background: var(--sunk); }

  .row-head { display: flex; flex-direction: column; gap: 6px; }
  .w {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
    line-height: 1;
  }
  .anchor .w { color: var(--ink-2); }
  .tag {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-3);
  }
  .blend .tag { color: var(--accent); }

  .stack { display: flex; flex-direction: column; gap: 11px; min-width: 0; }
  .meter { height: 3px; background: var(--rule); border-radius: 2px; overflow: hidden; }
  .meter span { display: block; height: 100%; background: var(--accent); }
  .anchor .meter span { background: var(--ink-3); }
  audio { width: 100%; height: 38px; display: block; }
  .note { font-size: 14px; color: var(--ink-3); margin: 0; }

  .checks { display: flex; flex-direction: column; gap: 0; }
  .check {
    display: grid;
    grid-template-columns: 20px 1fr auto;
    gap: 0 12px;
    align-items: baseline;
    padding: 11px 2px;
    border-bottom: 1px solid var(--rule);
  }
  .check:first-child { border-top: 1px solid var(--rule); }
  .mark { color: var(--pass); font-family: var(--mono); font-size: 13px; }
  .check-label { font-size: 15px; color: var(--ink-2); }
  .check-val {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }

  .verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .branch { border-left: 2px solid var(--rule); padding: 4px 0 4px 16px; }
  .branch.go { border-left-color: var(--accent); }
  .branch h3 {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin: 0 0 7px;
  }
  .branch.go h3 { color: var(--accent); }
  .branch p { margin: 0; font-size: 15px; color: var(--ink-2); }

  footer {
    border-top: 1px solid var(--rule);
    padding-top: 18px;
    font-size: 14px;
    color: var(--ink-3);
    max-width: 66ch;
  }
  code {
    font-family: var(--mono);
    font-size: 0.88em;
    background: var(--accent-soft);
    color: var(--ink);
    padding: 1px 5px;
    border-radius: 2px;
  }
  kbd {
    font-family: var(--mono);
    font-size: 11px;
    border: 1px solid var(--rule);
    border-bottom-width: 2px;
    border-radius: 3px;
    padding: 1px 5px;
    color: var(--ink-2);
  }
  audio:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  @media (max-width: 560px) {
    .row { grid-template-columns: 1fr; gap: 12px; }
    .row-head { flex-direction: row; align-items: baseline; gap: 10px; }
  }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 0 &middot; Linearity Gate</div>
    <h1>M1 &rarr; F1 Linearity Gate</h1>
    <p class="deck">Five renders of one sentence, with <code>style_ttl</code> interpolated between two presets. The
      question this set answers is not whether the endpoints work &mdash; the tensors already prove that. It is whether
      the three points <em>between</em> them are voices at all, or smear. That verdict gates the parametric
      approach.</p>
  </header>

  <section>
    <h2>Run parameters</h2>
    <p class="sect-note">Held identical across all five renders.</p>
    <dl class="spec">
      <div><dt>Text</dt><dd>The quick brown fox jumps over the lazy dog.</dd></div>
      <div><dt>Voice A &rarr; B</dt><dd>M1 &rarr; F1</dd></div>
      <div><dt>Language</dt><dd>en</dd></div>
      <div><dt>total_step</dt><dd>8</dd></div>
      <div><dt>speed</dt><dd>1.05</dd></div>
      <div><dt>include_duration</dt><dd>False</dd></div>
      <div><dt>style_dp</dt><dd>fixed at M1</dd></div>
      <div><dt>Clip length</dt><dd>3.10 s &middot; 44.1 kHz</dd></div>
    </dl>
  </section>

  <section>
    <h2>The sweep</h2>
    <p class="sect-note">In order, low to high. Press <kbd>1</kbd>&ndash;<kbd>5</kbd> to play a row; starting one stops
      the others, so nothing overlaps while you compare.</p>
    <div class="sweep">
__ROWS__
    </div>
  </section>

  <section>
    <h2>Already settled, numerically</h2>
    <p class="sect-note">These needed no ears, and all passed. They are why the listening test is the only thing
      left.</p>
    <div class="checks">
      <div class="check"><span class="mark">&check;</span><span class="check-label">w = 0.00 recovers M1's tensor
          bit-for-bit</span><span class="check-val">max |&Delta;| = 0.0</span></div>
      <div class="check"><span class="mark">&check;</span><span class="check-label">w = 1.00 recovers F1's tensor to
          float32 rounding</span><span class="check-val">max |&Delta;| = 2.98e-07</span></div>
      <div class="check"><span class="mark">&check;</span><span class="check-label">Both presets carry unit-norm rows,
          as assumed</span><span class="check-val">1.0000000 &plusmn; 2.4e-07</span></div>
      <div class="check"><span class="mark">&check;</span><span class="check-label">Blended rows stay unit-norm at
          w = 0.50</span><span class="check-val">0.9999998 &ndash; 1.0000002</span></div>
      <div class="check"><span class="mark">&check;</span><span class="check-label">No clipping, no NaNs in any
          render</span><span class="check-val">peaks &lt; 1.0</span></div>
    </div>
  </section>

  <section>
    <h2>What your ears decide</h2>
    <p class="sect-note">Judge the three blends, not the anchors.</p>
    <div class="verdict">
      <div class="branch go">
        <h3>Proceed</h3>
        <p>The blends read as clean, plausible, distinct voices &mdash; recognisably between M1 and F1 rather than
          either one. Phase 1 continues as written.</p>
      </div>
      <div class="branch">
        <h3>Re-scope</h3>
        <p>The blends smear, buzz, or collapse onto a pole. Linear travel through style space is not safe, and the plan
          moves to a learned manifold.</p>
      </div>
    </div>
  </section>

  <footer>
    One caveat worth carrying into later phases: <code>sample_noisy_latent()</code> draws an unseeded
    <code>randn</code>, so the vocoder is stochastic per call. Two renders of the same tensor differ as waveforms.
    Identity is therefore checked on tensors, never on audio &mdash; any future &ldquo;same as before?&rdquo; comparison
    needs a fixed seed to mean anything.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    var i = parseInt(e.key, 10);
    if (i >= 1 && i <= players.length) {
      var t = players[i - 1];
      if (t.paused) { t.play(); } else { t.pause(); }
    }
  });
</script>
"""

HTML = HTML.replace("__ROWS__", "\n".join(rows)).replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print("wrote", OUT, OUT.stat().st_size, "bytes")
