"""Bench generator: Phase 2a -- speed root-cause.

Two listener notes on bench 6's unperturbed M1 control clips described a
TIME-COMPRESSION symptom: "'chuck' after woodchuck sounds
condensed/hiccuped", "the final word 'morning' is pronounced too quickly so
it sounds like a hiccup". `phase2a_speed_rootcause.py` tested whether
`helper.py`'s default `speed=1.05` -- which shrinks the duration
predictor's own time budget by ~5% and (per the repo-wide render idiom)
trims the vocoder output to that shrunk duration -- is the cause. This bench
turns those renders into a page a listener can judge: does the artifact
they flagged survive at speed=1.00/0.90, does trimming discard audible
signal, and does more denoising (steps 8 -> 32) change anything at all.

Run from `py/`, after `phase2a_speed_rootcause.py` and
`phase2a_speed_word_durations.py` have produced their JSON + WAVs:

    cd py && python3 benches/phase2a_speed_rootcause_bench.py

Requires `py/results/phase2a/speed_rootcause.json`,
`py/results/phase2a/speed_word_durations.json`, and
`py/results/listening_sets/phase2a_speed_rootcause/` (WAVs, gitignored).
Writes `py/results/benches/phase2a_speed_rootcause.html` (gitignored: embeds
WAVs as base64). Not published as an Artifact.
"""

import json
import os

import numpy as np
import soundfile as sf

from bench_common import PLAYER_PAUSE_SCRIPT, b64

RESULTS = "results/phase2a"
LISTEN_DIR = "results/listening_sets/phase2a_speed_rootcause"
OUT_HTML = "results/benches/phase2a_speed_rootcause.html"

report = json.load(open(os.path.join(RESULTS, "speed_rootcause.json")))
word_durs = json.load(open(os.path.join(RESULTS, "speed_word_durations.json")))

SENTENCE_TEXT = {
    "woodchuck": "How much wood would a woodchuck chuck if it could chuck wood?",
    "seashells": "She sells seashells by the sea shore every summer morning.",
}
FLAGGED_SEED = {"woodchuck": 20261069, "seashells": 20261449}
FLAGGED_WORD_LABEL = {"woodchuck": "chuck (first occurrence, right after “woodchuck”)",
                       "seashells": "morning (utterance-final, only occurrence)"}
LISTENER_QUOTE = {
    "woodchuck": "“the 'chuck' after woodchuck sounds condensed/hiccuped”",
    "seashells": "“the final word 'morning' is pronounced too quickly so it sounds like a hiccup”",
}


def level_match(w, ref_rms):
    r = float(np.sqrt((w.astype(np.float64) ** 2).mean()))
    g = ref_rms / max(r, 1e-12)
    return (w * g).astype(np.float32), 20 * np.log10(g)


def read_wav(name):
    y, sr = sf.read(os.path.join(LISTEN_DIR, name))
    return y.astype(np.float32), sr


def write_tmp(name, y, sr):
    path = os.path.join(LISTEN_DIR, f"_lm_{name}")
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def find_a(sentence, seed, speed):
    for r in report["speed_sweep"]:
        if r["sentence"] == sentence and r["seed"] == seed and abs(r["speed"] - speed) < 1e-6:
            return r
    raise KeyError((sentence, seed, speed))


def find_c(sentence, seed, total_step):
    for r in report["steps_sweep"]:
        if r["sentence"] == sentence and r["seed"] == seed and r["total_step"] == total_step:
            return r
    raise KeyError((sentence, seed, total_step))


def find_word_row(sentence, seed, speed):
    for r in word_durs:
        if r["sentence"] == sentence and r["seed"] == seed and abs(r["speed"] - speed) < 1e-6:
            return r
    return None


def clip_card(path_wav, sr, label, stat, note, anchor=False):
    return f"""            <article class="clip{' anchor' if anchor else ''}">
              <div class="clip-head"><span class="clip-label">{label}</span><span class="clip-stat">{stat}</span></div>
              <audio controls preload="metadata" src="data:audio/wav;base64,{b64(path_wav)}"></audio>
              <p class="note">{note}</p>
            </article>"""


ARTIFACT_OPTS = [
    ("yes_clear", "Yes -- the artifact is clearly still there"),
    ("yes_mild", "Somewhat -- softer, but I can still hear it"),
    ("no", "No -- it's gone, this clip sounds clean"),
]


def radio_group(field, name):
    inputs = "\n".join(
        f'              <label><input type="radio" name="{name}" value="{val}" disabled> {text}</label>'
        for val, text in ARTIFACT_OPTS
    )
    return f'            <div class="choices" data-field="{field}">\n{inputs}\n            </div>'


def verdict_block(cid):
    return f"""          <div class="verdict" data-clip="{cid}">
            <div class="q-block">
              <p class="q-label">Is the artifact you flagged before still present in this clip? <span class="tick" data-tick data-clip="{cid}" data-field="artifact"></span></p>
{radio_group('artifact', f'{cid}_artifact')}
            </div>
            <div class="q-block">
              <p class="q-label">Notes <span class="tick" data-tick data-clip="{cid}" data-field="notes"></span></p>
              <textarea data-field="notes" data-clip="{cid}" disabled rows="2"
                placeholder="Anything specific -- where, how strong?"></textarea>
            </div>
          </div>"""


def main():
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    sentence_sections = []
    clip_ids_for_js = []
    instr_rows = []

    for sentence in ["woodchuck", "seashells"]:
        text = SENTENCE_TEXT[sentence]
        seed = FLAGGED_SEED[sentence]
        instr = report["instrumentation"][sentence]
        instr_rows.append((sentence, instr))

        # reference level = the flagged-seed, speed=1.05, 8-step trimmed clip
        ref_rec = find_a(sentence, seed, 1.05)
        ref_wav, sr = read_wav(ref_rec["trimmed_file"])
        ref_rms = float(np.sqrt((ref_wav.astype(np.float64) ** 2).mean()))

        # --- Group A: speed ladder, trimmed, level-matched ---
        speed_cells = []
        speed_word_rows = []
        for speed in [1.05, 1.00, 0.90]:
            rec = find_a(sentence, seed, speed)
            wav, _ = read_wav(rec["trimmed_file"])
            lvl, gdb = level_match(wav, ref_rms)
            path = write_tmp(f"{sentence}_speed{speed:.2f}_lm.wav", lvl, sr)
            cid = f"{sentence}_speed_{int(round(speed * 100)):03d}"
            clip_ids_for_js.append({"id": cid, "sentence": sentence, "group": "speed", "speed": speed})
            wr = find_word_row(sentence, seed, speed)
            fw = wr["flagged_word"]["duration"] if wr and wr["flagged_word"] else None
            stat = f"speed={speed:.2f}{' (default)' if speed == 1.05 else ''} &middot; gain {gdb:+.1f} dB"
            note = (f"Clip duration {rec['dur_after_speed_sec']:.3f}s. "
                    + (f"Flagged word duration &asymp; {fw:.2f}s (faster-whisper)." if fw else "Flagged word not detected in transcript."))
            speed_cells.append(clip_card(path, sr, f"speed {speed:.2f}", stat, note, anchor=(speed == 1.05)))
            speed_word_rows.append((speed, rec["dur_after_speed_sec"], fw))
            speed_cells[-1] = speed_cells[-1][:-len("</article>")] + verdict_block(cid) + "\n            </article>"

        # --- Group B: trim test at speed=1.05 ---
        full_wav, _ = read_wav(ref_rec["full_file"])
        trimmed_wav, _ = read_wav(ref_rec["trimmed_file"])
        # already at native level -- no level-match needed, same underlying render
        full_path = os.path.join(LISTEN_DIR, ref_rec["full_file"])
        trimmed_path = os.path.join(LISTEN_DIR, ref_rec["trimmed_file"])
        discarded_ms = instr["discarded_ms"]
        discarded_dbfs = instr["discarded_tail_dbfs"]
        clip_dbfs = instr["trimmed_clip_dbfs"]
        cid_full = f"{sentence}_trim_full"
        cid_trim = f"{sentence}_trim_trimmed"
        clip_ids_for_js += [{"id": cid_full, "sentence": sentence, "group": "trim", "which": "full"},
                             {"id": cid_trim, "sentence": sentence, "group": "trim", "which": "trimmed"}]
        trim_cells = [
            clip_card(trimmed_path, sr, "trimmed", "as normally rendered",
                      f"Cut to int(sr &middot; duration) = {instr['trimmed_samples']} samples, the repo-wide render idiom.",
                      anchor=True) [:-len("</article>")] + verdict_block(cid_trim) + "\n            </article>",
            clip_card(full_path, sr, "full (untrimmed)", f"+{discarded_ms:.0f} ms",
                      f"Raw vocoder output, {instr['discarded_samples']} extra samples kept. "
                      f"Discarded tail measures {discarded_dbfs:.1f} dBFS vs {clip_dbfs:.1f} dBFS for the clip "
                      f"itself ({clip_dbfs - discarded_dbfs:.0f} dB below) -- by this measure it is silence, not "
                      f"speech, so if you hear a difference here it is not the trimmed-off tail.")
            [:-len("</article>")] + verdict_block(cid_full) + "\n            </article>",
        ]

        # --- Group C: steps ladder at speed=1.05 ---
        steps_cells = []
        for total_step in [8, 32]:
            rec = find_c(sentence, seed, total_step)
            wav, _ = read_wav(rec["trimmed_file"])
            lvl, gdb = level_match(wav, ref_rms)
            path = write_tmp(f"{sentence}_steps{total_step}_lm.wav", lvl, sr)
            cid = f"{sentence}_steps_{total_step:02d}"
            clip_ids_for_js.append({"id": cid, "sentence": sentence, "group": "steps", "total_step": total_step})
            stat = f"TOTAL_STEP={total_step}{' (default)' if total_step == 8 else ''} &middot; gain {gdb:+.1f} dB"
            note = "Denoising step count only -- speed and duration are unchanged from the default render."
            card = clip_card(path, sr, f"{total_step} steps", stat, note, anchor=(total_step == 8))
            steps_cells.append(card[:-len("</article>")] + verdict_block(cid) + "\n            </article>")

        speed_table_rows = "\n".join(
            f'              <tr><td>{sp:.2f}</td><td>{d:.3f}</td>'
            f'<td>{(f"{fw:.2f}" if fw is not None else "n/a")}</td></tr>'
            for sp, d, fw in speed_word_rows
        )

        sentence_sections.append(f"""  <section class="sentence">
    <h2>&ldquo;{text}&rdquo;</h2>
    <p class="taskline"><strong>What was flagged</strong>Seed {seed} (unperturbed M1, bench 6's control clip):
      {LISTENER_QUOTE[sentence]} &mdash; flagged word: {FLAGGED_WORD_LABEL[sentence]}.</p>

    <h3>A &middot; Speed sweep &mdash; does the artifact soften as speed drops?</h3>
    <p class="sect-note">Same seed, same text, same 8 denoising steps. Level-matched to the speed=1.05 clip
      (rms). If <code>speed</code> is the cause, the flagged word should audibly lengthen and the artifact should
      soften or disappear by speed=0.90.</p>
    <div class="clips three">
{chr(10).join(speed_cells)}
    </div>
    <figure class="tablebox">
      <table>
        <thead><tr><th>speed</th><th>clip duration (s)</th><th>flagged word duration (s)</th></tr></thead>
        <tbody>
{speed_table_rows}
        </tbody>
      </table>
      <figcaption class="tcap">Flagged word duration from faster-whisper tiny.en word timestamps on this
        exact clip -- see full table across 3 seeds in the footer discussion.</figcaption>
    </figure>

    <h3>B &middot; Trim test &mdash; does trimming discard audible signal?</h3>
    <p class="sect-note">At speed=1.05 (default), the trimmed clip (as every render script in this repo
      produces it) vs. the full untrimmed vocoder output. Not level-matched -- it is the same render, the full
      clip is just longer.</p>
    <div class="clips two">
{chr(10).join(trim_cells)}
    </div>

    <h3>C &middot; Denoising steps &mdash; is more diffusion enough on its own?</h3>
    <p class="sect-note">Speed held at the default 1.05. TOTAL_STEP=8 (repo default) vs. 32. Level-matched to
      the speed=1.05/8-step clip. This isolates step count from timing: if the artifact is a noise problem, more
      steps should fix it without touching speed.</p>
    <div class="clips two">
{chr(10).join(steps_cells)}
    </div>
  </section>""")

    instr_table_rows = "\n".join(
        f'''          <tr><td>{s}</td><td>{i['raw_dur_sec']:.3f}</td><td>{i['dur_after_speed_sec']:.3f}</td>
            <td>{i['latent_len']}</td><td>{i['raw_vocoder_samples']}</td><td>{i['trimmed_samples']}</td>
            <td>{i['discarded_ms']:.0f}</td><td>{i['discarded_tail_dbfs']:.1f}</td><td>{i['trimmed_clip_dbfs']:.1f}</td></tr>'''
        for s, i in instr_rows
    )

    HTML = """<title>Speed Root-Cause Bench</title>
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
  h2 { font-family:var(--sans); font-weight:600; font-size:19px; margin:0 0 8px; }
  h3 { font-family:var(--sans); font-weight:600; font-size:14px; margin:22px 0 6px; color:var(--ink-2); }
  .sect-note { font-size:13.5px; color:var(--ink-3); margin:0 0 14px; max-width:70ch; }
  .task { background:var(--accent-soft); border:1px solid var(--accent); border-radius:3px; padding:18px 20px; }
  .task .tag { font-family:var(--mono); font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent); margin:0 0 9px; font-weight:500; }
  .task ol, .task ul { margin:0; padding-left:20px; color:var(--ink); }
  .task li { margin-bottom:7px; }
  .task li:last-child { margin-bottom:0; }
  .task .why { margin:12px 0 0; font-size:14px; color:var(--ink-2); }
  .taskline { background:var(--accent-soft); border-left:3px solid var(--accent); border-radius:2px; padding:11px 14px; margin:0 0 16px; font-size:14.5px; color:var(--ink); }
  .taskline strong { font-family:var(--sans); font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); display:block; margin-bottom:4px; }

  .tablebox { overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:3px; margin:14px 0 0; }
  table { border-collapse:collapse; width:100%; min-width:420px; font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:9px 14px; border-bottom:1px solid var(--rule); }
  th:first-child,td:first-child { text-align:left; }
  th { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:500; }
  tr:last-child td { border-bottom:none; }
  figcaption.tcap { font-size:12.5px; color:var(--ink-3); padding:10px 14px; border-top:1px solid var(--rule); margin:0; }

  .sentence { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:20px 22px 22px; }
  .clips { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:0 0 4px; }
  .clips.two { grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }
  .clip { background:var(--ground); border:1px solid var(--rule); border-radius:3px; padding:13px 15px; display:flex; flex-direction:column; gap:9px; min-width:0; }
  .clip.anchor { background:var(--sunk); }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:13px; }
  .clip-stat { font-family:var(--mono); font-size:10px; color:var(--ink-3); font-variant-numeric:tabular-nums; }
  audio { width:100%; height:36px; display:block; }
  .note { font-size:12.5px; color:var(--ink-3); margin:0; }

  .verdict { margin-top:6px; padding-top:12px; border-top:1px dashed var(--rule); display:flex; flex-direction:column; gap:10px; }
  .q-block { display:flex; flex-direction:column; gap:6px; }
  .q-label { font-family:var(--sans); font-weight:600; font-size:12px; color:var(--ink); margin:0; display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .choices { display:flex; flex-direction:column; gap:5px; }
  .choices label { display:flex; align-items:flex-start; gap:7px; font-family:var(--sans); font-size:12px; color:var(--ink-2); cursor:pointer; }
  .choices input[type="radio"] { margin-top:2px; accent-color:var(--accent); flex-shrink:0; }
  textarea { width:100%; font-family:var(--sans); font-size:12px; color:var(--ink); background:var(--ground); border:1px solid var(--rule); border-radius:3px; padding:7px 9px; resize:vertical; box-sizing:border-box; }
  textarea::placeholder { color:var(--ink-3); }
  textarea:disabled, .choices input:disabled { opacity:.55; }
  .tick { font-family:var(--mono); font-size:10px; letter-spacing:.04em; color:var(--ink-3); font-weight:400; }
  .tick.saving { color:var(--ink-3); }
  .tick.saved { color:var(--accent); }
  .tick.err { color:var(--flag); }

  .progress { font-family:var(--mono); font-size:12px; color:var(--ink-3); margin:0; }
  .progress.ready { color:var(--accent); }
  .progress.nodb { color:var(--flag); }

  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:70ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2a &middot; Speed root-cause</div>
    <h1>Is speed=1.05 the hiccup?</h1>
    <p class="deck">A listener flagged a compression-sounding artifact on two unperturbed M1 control clips --
      "chuck" condensed in "woodchuck", the final word "morning" rushed like a hiccup. <code>helper.py</code>'s
      default <code>speed=1.05</code> shrinks the duration predictor's own time estimate by ~5% before anything is
      rendered, so the acoustic model must fit the utterance into fewer frames than its own predictor asked for.
      That default is <strong>upstream's</strong>, not this fork's &mdash; commit <code>8518b839</code> introduced it
      across all nine language bindings at once &mdash; so it applies to stock Supertonic everywhere, and none of
      the clips here contain anything from this fork. This page asks whether that artifact survives when speed drops
      toward 1.00/0.90, and whether more denoising steps alone fix it. It also tests a trim hypothesis that the
      measurements have already refuted: the discarded tail is about 80&nbsp;dB below the clip, digital silence
      rather than cut-off speech, so the trimmed and untrimmed clips are expected to sound identical &mdash; they
      are included as a check on that measurement, not as a live hypothesis.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li><strong>For each clip below, say whether the specific artifact you flagged on the original control
          clip is still present</strong> -- not whether the clip sounds good in general, and not "which do you
          prefer." The speed=1.05 clip in group A is the same render (same seed, same text, unperturbed M1) as the
          control clip you already flagged.</li>
        <li>Groups A/B/C isolate three different candidate causes one at a time: A changes only the time budget,
          B changes only whether the post-render trim is applied, C changes only denoising step count. Everything
          else in each group is identical.</li>
      </ol>
      <p class="why">This decides whether <code>speed</code>'s default value is the root cause, a contributing
        cause, or unrelated to the compression artifact -- and whether fixing it requires changing the default,
        adding more denoising steps, or something else entirely.</p>
    </div>
  </section>

  <section>
    <h2 style="font-size:15px;margin-bottom:10px;">Instrumentation &middot; flagged seed, defaults (speed=1.05, TOTAL_STEP=8)</h2>
    <p class="sect-note">One render per sentence at its flagged seed, every stage of the time-budget pipeline.
      "discarded tail dBFS" is the rms level of the samples the standard trim removes; "trimmed clip dBFS" is the
      rms level of the kept clip, for comparison -- the gap between them says whether the removed tail is audio or
      near-silence.</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>sentence</th><th>raw dur (s)</th><th>dur/speed (s)</th><th>latent_len</th>
          <th>raw vocoder (samples)</th><th>trimmed (samples)</th><th>discarded (ms)</th>
          <th>discarded tail (dBFS)</th><th>trimmed clip (dBFS)</th></tr></thead>
        <tbody>
__INSTR_ROWS__
        </tbody>
      </table>
      <figcaption class="tcap">Discarded tail sits roughly 80 dB below the clip's own level in both sentences --
        by this measure, what the standard trim removes is near-digital-silence, not truncated speech.</figcaption>
    </figure>
  </section>

__SECTIONS__

  <footer>
    Scope. Text and vocoder seed are held fixed within each group (A/B/C), so what varies is exactly the one
    condition named. "Flagged word duration" in group A's table comes from faster-whisper tiny.en word timestamps
    run on that exact rendered clip; across the flagged seed plus 2 additional seeds per sentence (not all shown
    here -- see <code>results/phase2a/speed_word_durations.json</code>), the flagged word and the utterance-final
    word lengthen as speed drops from 1.05 toward 0.90 in the large majority of (sentence, seed) pairs, and in
    several cases lengthen <em>faster</em> than the clip's overall duration does -- i.e. more than a uniform
    time-stretch would predict, consistent with that word being disproportionately compressed at the default speed.
    faster-whisper's word-boundary timestamps have their own jitter (tens of ms), so treat single-clip deltas as
    indicative, not exact. Vocoder sampling is unseeded across different (text, style) pairs but held fixed via
    <code>np.random.seed</code> within each comparison here, per <code>CLAUDE.md</code>.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
<script>
var CLIPS = __CLIPS_JSON__;
(function () {
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  var verdictEls = $all('.verdict');
  var state = {}, pending = {}, timers = {}, db = null;
  verdictEls.forEach(function (v) { var cid = v.getAttribute('data-clip'); state[cid] = {}; pending[cid] = {}; });

  function setTick(cid, field, mode) {
    var el = document.querySelector('[data-tick][data-clip="' + cid + '"][data-field="' + field + '"]');
    if (!el) return;
    el.className = 'tick' + (mode ? ' ' + mode : '');
    el.textContent = mode === 'saving' ? 'saving…' : mode === 'saved' ? 'saved ✓' : mode === 'err' ? 'not saved' : '';
  }
  function clipMeta(cid) { for (var i = 0; i < CLIPS.length; i++) { if (CLIPS[i].id === cid) return CLIPS[i]; } return null; }
  function save(cid) {
    if (!db) return;
    var fields = Object.keys(pending[cid]);
    if (!fields.length) return;
    pending[cid] = {};
    var meta = clipMeta(cid);
    var body = Object.assign({}, state[cid], { updated_at: new Date().toISOString() });
    if (meta) body.meta = meta;
    db.doc('speed_verdicts/' + cid).set(body).then(function () {
      fields.forEach(function (f) { setTick(cid, f, 'saved'); });
      updateProgress();
    }).catch(function () { fields.forEach(function (f) { setTick(cid, f, 'err'); }); });
  }
  function onFieldChange(cid, field, value) {
    state[cid][field] = value; pending[cid][field] = true; setTick(cid, field, 'saving');
    clearTimeout(timers[cid]); timers[cid] = setTimeout(function () { save(cid); }, 600);
  }
  verdictEls.forEach(function (v) {
    var cid = v.getAttribute('data-clip');
    $all('.choices', v).forEach(function (group) {
      var field = group.getAttribute('data-field');
      $all('input[type=radio]', group).forEach(function (input) {
        input.addEventListener('change', function () { if (input.checked) onFieldChange(cid, field, input.value); });
      });
    });
    $all('textarea', v).forEach(function (ta) {
      var field = ta.getAttribute('data-field');
      ta.addEventListener('input', function () { onFieldChange(cid, field, ta.value); });
    });
  });
  function updateProgress() {
    var el = document.getElementById('progress-line');
    if (!el) return;
    if (!db) { el.textContent = 'Responses cannot be saved in this view.'; el.className = 'progress nodb'; return; }
    var n = CLIPS.filter(function (c) { return state[c.id] && state[c.id].artifact; }).length;
    el.textContent = n + ' of ' + CLIPS.length + ' clips answered';
    el.className = 'progress ready';
  }
  function enableInputs() { $all('.verdict input, .verdict textarea').forEach(function (el) { el.disabled = false; }); }
  function hydrate() {
    var ids = CLIPS.map(function (c) { return c.id; });
    return Promise.all(ids.map(function (cid) {
      return db.doc('speed_verdicts/' + cid).get().then(function (snap) {
        if (!snap.exists) return;
        var data = snap.data() || {};
        state[cid] = data;
        var v = document.querySelector('.verdict[data-clip="' + cid + '"]');
        if (!v) return;
        Object.keys(data).forEach(function (field) {
          if (field === 'updated_at' || field === 'meta') return;
          var val = data[field];
          if (val === undefined || val === null || val === '') return;
          var radio = v.querySelector('.choices[data-field="' + field + '"] input[value="' + val + '"]');
          if (radio) { radio.checked = true; setTick(cid, field, 'saved'); return; }
          var ta = v.querySelector('textarea[data-field="' + field + '"]');
          if (ta) { ta.value = val; setTick(cid, field, 'saved'); }
        });
      }).catch(function () {});
    }));
  }
  (async function () {
    var claude = window.claude;
    db = (claude && typeof claude.use === 'function') ? await claude.use('db') : null;
    if (!db) { updateProgress(); return; }
    await hydrate(); enableInputs(); updateProgress();
  })();
})();
</script>
"""
    HTML = (HTML
            .replace("__INSTR_ROWS__", instr_table_rows)
            .replace("__SECTIONS__", "\n\n".join(sentence_sections))
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(clip_ids_for_js)))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes, {len(clip_ids_for_js)} clips")


if __name__ == "__main__":
    main()
