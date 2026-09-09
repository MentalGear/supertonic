"""Bench generator: Phase 2b prosody correction.

Builds "The Metric That Could Not Hear" listening bench: the perturbation
ladder that an utterance-level ECAPA/spectral-flatness battery reported as
acoustically flat, alongside the log-mel diffs that show the emphasis moving
word by word instead.

Published: https://claude.ai/code/artifact/d00d367b-18f0-41ba-85c8-a192c58b41e3
Verdict, inputs, and rebuild notes: docs/LISTENING_BENCHES.md.

Run from `py/` (relative paths below assume that cwd), after generating the
inputs with the Phase 2b prosody-correction pipeline:

    cd py && python3 benches/phase2b_prosody_correction_bench.py

Requires `py/results/listening_sets/phase2b_prosody_ab/` (WAVs) and
`py/results/phase2b_prosody/` (mel-diff PNGs), both gitignored and not
checked into the repo — you must (re)generate them first. Writes
`py/results/benches/phase2b_prosody_correction.html`, itself gitignored: it
embeds WAVs and PNGs as base64 and runs several MB.
"""

import pathlib

from bench_common import PLAYER_PAUSE_SCRIPT, b64

AB = pathlib.Path("results/listening_sets/phase2b_prosody_ab")
FIG = pathlib.Path("results/phase2b_prosody")
OUT = pathlib.Path("results/benches/phase2b_prosody_correction.html")


def wav(n):
    return b64(AB / n)


def png(n):
    return b64(FIG / n)


def clip(name, label, stat, note, anchor=False):
    return f"""          <article class="clip{' anchor' if anchor else ''}">
            <div class="clip-head"><span class="clip-label">{label}</span><span class="clip-stat">{stat}</span></div>
            <audio controls preload="metadata" src="data:audio/wav;base64,{wav(name)}"></audio>
            <p class="note">{note}</p>
          </article>"""


DECISIVE = f"""      <div class="ladder decisive">
        <h3>Same total spectral change &mdash; 11.6 dB against 11.5 dB &mdash; same loudness</h3>
        <div class="clips">
{clip('A_M1_random_eps3.20_levelmatched.wav', 'random, 73&deg;/row', 'eps 3.20', 'A big step in a random direction.')}
{clip('C_M1_towardF1_eps0.20_levelmatched.wav', 'toward F1, 11&deg;/row', 'eps 0.20', 'A small step toward another preset.')}
        </div>
        <p class="ask">Both moved the spectrogram by the same amount. If one sounds like the same person emphasising
          different words and the other like a different person, that is the finding &mdash; six times the distance
          travelled buys a different <em>kind</em> of change, not more of the same one.</p>
      </div>"""

M1 = f"""      <div class="ladder">
        <h3>M1 &mdash; the emphasis pattern reorders</h3>
        <div class="clips">
{clip('A0_M1_base.wav', 'base', 'eps 0.00', 'Unperturbed.', True)}
{clip('A_M1_random_eps0.20_levelmatched.wav', 'eps 0.20', '11.3&deg;', 'Per-word range 1.6 dB.')}
{clip('A_M1_random_eps0.80_levelmatched.wav', 'eps 0.80', '38.5&deg;', 'Per-word range 3.7 dB.')}
{clip('A_M1_random_eps3.20_levelmatched.wav', 'eps 3.20', '72.2&deg;', 'Per-word range 10.1 dB. Ranking has reordered.')}
        </div>
      </div>"""

F1 = f"""      <div class="ladder">
        <h3>F1 &mdash; the same pattern, scaled</h3>
        <div class="clips">
{clip('B0_F1_base.wav', 'base', 'eps 0.00', 'Unperturbed.', True)}
{clip('B_F1_random_eps0.20_levelmatched.wav', 'eps 0.20', '11.3&deg;', 'Per-word range 2.4 dB.')}
{clip('B_F1_random_eps0.80_levelmatched.wav', 'eps 0.80', '38.5&deg;', 'Per-word range 4.0 dB.')}
{clip('B_F1_random_eps3.20_levelmatched.wav', 'eps 3.20', '72.2&deg;', 'Range 5.0 dB, and saturating.')}
        </div>
      </div>"""

REF = f"""      <div class="ladder">
        <h3>For scale &mdash; a known identity change</h3>
        <div class="clips">
{clip('C_F1_full_swap_levelmatched.wav', 'M1 to F1, full swap', '16.6 dB', 'What changing speaker actually sounds like.')}
        </div>
      </div>"""

FIGS = [
    ("fig_mel_M1_random_ray.png", "M1, random direction",
     "At eps 0.20 nearly the whole difference is one broadband block on &ldquo;The quick&rdquo;, with the rest of the sentence near-white. A change localised to particular words is what a prosodic effect looks like."),
    ("fig_mel_timbre_reference.png", "The timbre reference",
     "The preset-aligned panels are dense harmonic striping across every voiced frame &mdash; pitch and formants moved. The random panel is red-dominant in word-scale blocks &mdash; levels moved."),
    ("fig_mel_matched_magnitude.png", "Matched magnitude",
     "Random at 73&deg;/row against toward-F1 at 11&deg;/row, carrying the same 11.5 dB change. Six times the angle buys the same total movement, distributed completely differently."),
    ("fig_per_word_energy_M1.png", "M1 per-word energy",
     "The curves cross as magnitude grows: the emphasis ranking itself reorders."),
    ("fig_per_word_energy_F1.png", "F1 per-word energy",
     "The curves stay roughly parallel: the same pattern, louder."),
]
figblocks = "\n".join(
    f"""      <figure class="fig">
        <img src="data:image/png;base64,{png(f)}" alt="{t}">
        <figcaption><strong>{t}.</strong> {c}</figcaption>
      </figure>""" for f, t, c in FIGS
)

HTML = """<title>The Metric That Could Not Hear</title>
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
  .correction { background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--flag); border-radius:3px; padding:18px 20px; }
  .correction .was { font-family:var(--mono); font-size:12px; color:var(--flag); letter-spacing:.06em; text-transform:uppercase; margin:0 0 7px; }
  .correction p { margin:0; }
  .correction .old { color:var(--ink-3); text-decoration:line-through; margin-bottom:12px; }
  .ladders { display:flex; flex-direction:column; gap:22px; }
  .decisive { background:var(--surface); border:1px solid var(--accent); border-radius:3px; padding:16px 18px; }
  .clips { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:13px 15px; display:flex; flex-direction:column; gap:9px; min-width:0; }
  .decisive .clip { background:var(--sunk); }
  .clip.anchor { background:var(--sunk); }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:13.5px; }
  .clip-stat { font-family:var(--mono); font-size:10.5px; color:var(--ink-3); font-variant-numeric:tabular-nums; }
  audio { width:100%; height:36px; display:block; }
  .note { font-size:13px; color:var(--ink-3); margin:0; }
  .ask { font-size:14.5px; color:var(--ink-2); margin:14px 0 0; }
  .fig { margin:0; background:var(--surface); border:1px solid var(--rule); border-radius:3px; overflow:hidden; }
  .fig img { display:block; width:100%; height:auto; }
  .fig figcaption { font-size:13.5px; color:var(--ink-3); padding:12px 16px; border-top:1px solid var(--rule); }
  .figs { display:flex; flex-direction:column; gap:16px; }
  .tablebox { overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:3px; }
  table { border-collapse:collapse; width:100%; min-width:560px; font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:9px 14px; border-bottom:1px solid var(--rule); }
  th:first-child,td:first-child { text-align:left; }
  th { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:500; }
  tr:last-child td { border-bottom:none; }
  td.flag { color:var(--flag); }
  td.ok { color:var(--accent); }
  figcaption.tcap { font-size:13.5px; color:var(--ink-3); padding:12px 16px; border-top:1px solid var(--rule); margin:0; }
  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:68ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2b &middot; Correction</div>
    <h1>The Metric That Could Not Hear</h1>
    <p class="deck">A battery of acoustic measurements called this perturbation ladder flat, and concluded that most of
      style space is inaudible. A listener played the same clips and immediately heard the emphasis moving. The listener
      was right, and the reason is worth more than the original result.</p>
  </header>

  <section>
    <div class="correction">
      <p class="was">Recorded, and wrong</p>
      <p class="old">Most of style space is nearly inaudible, so the audio-to-style inverse is ill-posed in nearly
        every direction.</p>
      <p><strong>Style space is anisotropic in <em>what</em> a direction changes, not in whether it changes anything.</strong>
        A random direction is identity-poor and emphasis-rich: at matched angle it moves pitch register 20x less than a
        preset-aligned one, but the spectrogram only 1.7x less. The probe's null means ECAPA could not see most of what
        a style direction does &mdash; not that most directions do nothing.</p>
    </div>
  </section>

  <section>
    <h2>The measurement that misled, and why</h2>
    <p class="sect-note">Preset-aligned change divided by random change, at matched per-row angle. The first row is the
      instrument the original conclusion was built on; the rest measure the same thing with tools that can hear
      prosody.</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>measure</th><th>eps 0.10</th><th>eps 0.20</th><th>eps 0.40</th><th>eps 0.80</th></tr></thead>
        <tbody>
          <tr><td>&Delta;ECAPA <em>(the original stick)</em></td><td class="flag">4.15</td><td class="flag">6.51</td><td class="flag">4.83</td><td class="flag">3.51</td></tr>
          <tr><td>emphasis contour, dB</td><td class="ok">1.11</td><td class="ok">1.80</td><td class="ok">1.65</td><td class="ok">1.74</td></tr>
          <tr><td>per-word emphasis, dB</td><td class="ok">1.51</td><td class="ok">2.22</td><td class="ok">2.35</td><td class="ok">3.15</td></tr>
          <tr><td>per-frame spectral shape, dB</td><td class="ok">1.12</td><td class="ok">1.70</td><td class="ok">1.75</td><td class="ok">1.67</td></tr>
          <tr><td>LTAS shape, dB</td><td class="ok">1.53</td><td class="ok">2.30</td><td class="ok">2.40</td><td class="ok">1.95</td></tr>
        </tbody>
      </table>
      <figcaption class="tcap">ECAPA is trained to be invariant to prosody &mdash; that invariance is the whole point of
        a speaker-verification embedding. Used as an audibility meter it reports 3.5 to 6.5x where acoustic measures
        report 1.1 to 3.2x. The &ldquo;3-5x more audible&rdquo; figure was a property of the instrument.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The pair that settles it</h2>
    <p class="sect-note">Every clip below is rms level-matched to its base, because magnitude buys up to +4.9 dB of
      plain loudness and an unmatched comparison would be decided by that alone. Your task: is the decisive pair below a
      difference in <em>emphasis</em> or a difference in <em>speaker</em>?</p>
    <div class="ladders">
__DECISIVE__
__M1__
__F1__
__REF__
    </div>
  </section>

  <section>
    <h2>Seen instead of summarised</h2>
    <p class="sect-note">The same clips as 2D log-mel differences. The renders are frame-aligned &mdash; identical
      sample counts, re-rendering reproduces them to 16-bit quantisation &mdash; so these are genuine sample-by-sample
      comparisons rather than statistics.</p>
    <div class="figs">
__FIGS__
    </div>
  </section>

  <section>
    <h2>What survives, and what does not</h2>
    <p class="sect-note" style="max-width:66ch;">The ECAPA null stands &mdash; a speaker embedding still does not
      predict the style tensor, and the direct encoder is still the route. What changes is the reason, and the reason
      matters: the old wording implied a hard information ceiling on <em>any</em> encoder. There is no such ceiling.
      The information is in the audio; it was not in ECAPA. A prosody-bearing input deserves testing before anyone
      concludes the inverse is ill-posed.</p>
    <p class="sect-note" style="max-width:66ch;">The strong version of the two-subspace idea &mdash; cleanly separable
      timbre and prosody subspaces &mdash; is <strong>not</strong> supported. The broadband-versus-spectral split of the
      difference is essentially the same for every direction family, and at matched total change the localisation
      advantage disappears. Random directions do eventually move pitch and timbre; they just buy far less of it per
      unit travelled.</p>
  </section>

  <footer>
    Scope. One seeded random direction per base, so n=1 in direction space, though four independent directions agree on
    the ECAPA side. Word boundaries come from a small ASR model with one hand-repaired boundary. M1's
    &ldquo;dog&rdquo; sits at -26 dB in the base, so its large relative deltas are partly a small-denominator effect.
    F0 contours above eps 1.60 are unreliable through octave errors. And <code>style_dp</code> was pinned throughout,
    so speech rate and timing &mdash; a first-order emotion cue &mdash; lie outside everything measured here.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
"""

HTML = (HTML.replace("__DECISIVE__", DECISIVE).replace("__M1__", M1)
            .replace("__F1__", F1).replace("__REF__", REF).replace("__FIGS__", figblocks)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print("wrote", OUT, OUT.stat().st_size, "bytes")
