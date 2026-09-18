"""Bench generator: does the 4-11 kHz sibilant-band ratio track an audible
over-driven 's', or does it just score high on any fricative-heavy sentence
regardless of how it sounds?

`phase2a_sibilance_latent.py`'s Part C (`reference_sample_check`) ran our
sibilant-band-ratio measure over upstream's OWN showcase files in
`assets/audio_samples/` and found upstream's zero-shot renders scoring as
high as, or higher than, our flagged clips: keld_supertonic3.wav = 0.459,
luna_supertonic3.wav = 0.617, versus our flagged F3 (0.543, Part A
main-comparison render) and F5 (0.700, same). Twelve automated measures
(three latent, four waveform, tried across two scripts) have now failed to
separate clips a listener flagged from clips the same listener called clean
on our own pool -- see `phase2a_sibilance_latent.py`'s
`bench7_pool_reproduction.flagged_vs_clean_validation`, where every
validation entry overlaps. This bench asks a listener directly, blind, to
settle two questions the metric alone cannot:
  1. Is upstream's own showcase audio ALSO over-driven (making this an
     engine-wide fricative-synthesis property, just most audible on F3/F5),
     or does it sound clean despite a high ratio (meaning the ratio doesn't
     measure the artifact at all -- explaining all twelve failures at once)?
  2. Do OUR flagged clips actually sound worse than our clean ones, blind,
     without the listener knowing which pool they came from?

Renders NOTHING new. Every clip already exists on disk:
  - upstream's showcase renders and their human reference recordings, from
    `assets/audio_samples/` (classification below is sourced from
    `assets/README.md`'s own "Reference voice" / "Supertonic 3 output"
    table -- not inferred from filenames);
  - our flagged and clean bench-7 pool clips, from
    `results/listening_sets/phase2a_baseline/`.

Source classification of `assets/audio_samples/*.wav` (see `assets/README.md`,
"Custom Voices and Audio Samples" -- a table of "Reference voice" /
"Supertonic 3 output" pairs, mirroring upstream's own audio-sample demo):
  - `*_reference.wav`  = the human voice-cloning reference recording (real
    speech), played in the table's "Reference voice" column.
  - `*_supertonic3.wav` = Supertonic 3's zero-shot clone of that reference
    voice (synthesis), played in the table's "Supertonic 3 output" column.
This is documentation, not a guess from filenames. Only `keld`, `luna`, and
`watson`'s reference recordings contain any sibilant content by our
detector (`alphonse`/`moka`/`nora` reference recordings have none), so only
those three are used as the "real speech" anchor.

Every clip is RMS level-matched to the set's median loudness (heterogeneous
sources, so there is no single natural reference clip to match to) and
played in shuffled, blind order. CRITICAL: unlike this project's other bench
generators, NO reveal panel is included and the CLIPS array embedded in the
page's own <script> carries only clip ids -- no kind/source flag anywhere in
visible text or client code, per this task's blinding requirement. The
source mapping exists ONLY in this script's `CLIP_SPECS`, in the written
`manifest.json`, and (if this bench is ever published) in the artifact's
`verdicts` db collection, written server-side, never in page source.

Limitation, stated on the page too: different sources means different
speakers, sentences, and (for the reference recordings) different
recording conditions/mics. This is unavoidable given what exists on disk
and is not something the level-matching or blinding can fix -- only the
open, per-clip question can still be answered meaningfully across it.

Run from `py/`:
    cd py && python3 benches/phase2a_sibilance_sources_bench.py

Requires `assets/audio_samples/` (repo-shipped, not gitignored) and
`results/listening_sets/phase2a_baseline/` (from
`benches/phase2a_baseline_bench.py`, already on disk from bench 7). Writes
leveled WAVs + manifest.json to
`results/listening_sets/phase2a_sibilance_sources/` and the page to
`results/benches/phase2a_sibilance_sources.html` (gitignored, embeds WAVs as
base64). Does NOT publish; that is left to whoever reviews the output.
"""

import json
import os
import random
import sys

import numpy as np
import soundfile as sf

from bench_common import PLAYER_PAUSE_SCRIPT, b64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2b_speaker_similarity import level_match  # noqa: E402

ASSETS_DIR = os.path.normpath(os.path.join("..", "assets", "audio_samples"))
BASELINE_DIR = "results/listening_sets/phase2a_baseline"
LISTEN_DIR = "results/listening_sets/phase2a_sibilance_sources"
OUT_HTML = "results/benches/phase2a_sibilance_sources.html"

SHUFFLE_SEED = 20360918  # blind display order

# Every clip that goes into the bench, and why. `ratio` is the 4-11 kHz
# sibilant-band ratio already measured in results/phase2a/sibilance_latent.json
# (Part C for assets/audio_samples/, bench7_pool_reproduction for our pool
# clips) -- kept here for the manifest/report only, never surfaced on the
# page. `kind` is the hidden source label: upstream_synth, upstream_reference,
# ours_flagged, ours_clean.
CLIP_SPECS = [
    # --- upstream's own showcase renders (Supertonic 3 zero-shot clones) ---
    dict(kind="upstream_synth", path=os.path.join(ASSETS_DIR, "alphonse_supertonic3.wav"),
         ratio=0.0702, note="upstream showcase (elder character voice, Korean text)"),
    dict(kind="upstream_synth", path=os.path.join(ASSETS_DIR, "keld_supertonic3.wav"),
         ratio=0.4592, note="upstream showcase (news voice, English text) -- named in the task brief"),
    dict(kind="upstream_synth", path=os.path.join(ASSETS_DIR, "luna_supertonic3.wav"),
         ratio=0.6168, note="upstream showcase (audiobook voice, English text) -- named in the task brief"),
    dict(kind="upstream_synth", path=os.path.join(ASSETS_DIR, "moka_supertonic3.wav"),
         ratio=0.2722, note="upstream showcase (character voice, Japanese text)"),
    dict(kind="upstream_synth", path=os.path.join(ASSETS_DIR, "nora_supertonic3.wav"),
         ratio=None, note="upstream showcase (call-center voice, English text); no sibilant "
                           "words detected by our locator -- a null-content foil"),
    dict(kind="upstream_synth", path=os.path.join(ASSETS_DIR, "watson_supertonic3.wav"),
         ratio=0.3443, note="upstream showcase (audiobook voice, Japanese text)"),
    # --- upstream's own human reference recordings (real speech) ---
    dict(kind="upstream_reference", path=os.path.join(ASSETS_DIR, "keld_reference.wav"),
         ratio=0.4322, note="human reference recording behind the keld voice clone"),
    dict(kind="upstream_reference", path=os.path.join(ASSETS_DIR, "luna_reference.wav"),
         ratio=0.3556, note="human reference recording behind the luna voice clone"),
    dict(kind="upstream_reference", path=os.path.join(ASSETS_DIR, "watson_reference.wav"),
         ratio=0.0015, note="human reference recording behind the watson voice clone"),
    # --- our flagged bench-7 pool clips ---
    dict(kind="ours_flagged", path=os.path.join(BASELINE_DIR, "pool_F3_t4_seed20361268.wav"),
         ratio=0.6066, note="bench 7: F3, library sentence -- listener 'yes, clearly ... "
                             "strong sharp s over-drive resulting in a sharp hissing'"),
    dict(kind="ours_flagged", path=os.path.join(BASELINE_DIR, "pool_F5_t1_seed20361347.wav"),
         ratio=0.4831, note="bench 7: F5, seashells sentence -- listener 'maybe the s-es "
                             "are a bit sharp'"),
    # --- our clean bench-7 pool clips (two share text with a flagged clip above) ---
    dict(kind="ours_clean", path=os.path.join(BASELINE_DIR, "pool_M4_t4_seed20361077.wav"),
         ratio=0.5205, note="bench 7: M4, SAME library sentence as the F3 flagged clip -- "
                             "not flagged"),
    dict(kind="ours_clean", path=os.path.join(BASELINE_DIR, "pool_F2_t1_seed20361205.wav"),
         ratio=0.3952, note="bench 7: F2, SAME seashells sentence as the F5 flagged clip -- "
                             "not flagged"),
    dict(kind="ours_clean", path=os.path.join(BASELINE_DIR, "pool_M3_t2_seed20361017.wav"),
         ratio=0.5553, note="bench 7: M3, doctor sentence -- not flagged"),
]

# Persists per-clip verdicts to the artifact's `db` capability at
# `verdicts/<clip_id>` and the page-level free-text box at `verdicts/_overall`.
# Deliberately carries NO source/kind field anywhere -- the CLIPS array below
# has only {id}, and the write body is exactly what the listener typed/picked
# plus a timestamp. The source mapping is never present in page text or
# script; it lives only in manifest.json (and, if this is ever published,
# would need to be added to the db separately, out of band, after the fact).
DB_SCRIPT = r"""(function () {
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }

  var verdictEls = $all('.verdict');
  var state = {};
  var pending = {};
  var timers = {};
  var db = null;

  verdictEls.forEach(function (v) {
    var tid = v.getAttribute('data-clip');
    state[tid] = {};
    pending[tid] = {};
  });

  function setTick(tid, field, mode) {
    var el = document.querySelector('[data-tick][data-clip="' + tid + '"][data-field="' + field + '"]');
    if (!el) return;
    el.className = 'tick' + (mode ? ' ' + mode : '');
    el.textContent = mode === 'saving' ? 'saving…' : mode === 'saved' ? 'saved ✓' : mode === 'err' ? 'not saved' : '';
  }

  function save(tid) {
    if (!db) return;
    var fields = Object.keys(pending[tid]);
    if (!fields.length) return;
    pending[tid] = {};
    var body = Object.assign({}, state[tid], { updated_at: new Date().toISOString() });
    db.doc('verdicts/' + tid).set(body).then(function () {
      fields.forEach(function (f) { setTick(tid, f, 'saved'); });
      updateProgress();
    }).catch(function () {
      fields.forEach(function (f) { setTick(tid, f, 'err'); });
    });
  }

  function onFieldChange(tid, field, value) {
    state[tid][field] = value;
    pending[tid][field] = true;
    setTick(tid, field, 'saving');
    clearTimeout(timers[tid]);
    timers[tid] = setTimeout(function () { save(tid); }, 600);
  }

  function wireVerdict(v) {
    var tid = v.getAttribute('data-clip');
    $all('.choices', v).forEach(function (group) {
      var field = group.getAttribute('data-field');
      $all('input[type=radio]', group).forEach(function (input) {
        input.addEventListener('change', function () {
          if (input.checked) onFieldChange(tid, field, input.value);
        });
      });
    });
    $all('textarea', v).forEach(function (ta) {
      var field = ta.getAttribute('data-field');
      ta.addEventListener('input', function () { onFieldChange(tid, field, ta.value); });
    });
  }
  verdictEls.forEach(wireVerdict);

  function updateProgress() {
    var el = document.getElementById('progress-line');
    if (!el) return;
    if (!db) {
      el.textContent = 'Responses cannot be saved in this view.';
      el.className = 'progress nodb';
      return;
    }
    var n = CLIPS.filter(function (c) { return state[c.id] && state[c.id].q1; }).length;
    el.textContent = n + ' of ' + CLIPS.length + ' clips answered';
    el.className = 'progress ready';
  }

  function enableInputs() {
    $all('.verdict input, .verdict textarea').forEach(function (el) { el.disabled = false; });
  }

  function hydrate() {
    var ids = CLIPS.map(function (c) { return c.id; }).concat(['_overall']);
    return Promise.all(ids.map(function (tid) {
      return db.doc('verdicts/' + tid).get().then(function (snap) {
        if (!snap.exists) return;
        var data = snap.data() || {};
        state[tid] = data;
        var v = document.querySelector('.verdict[data-clip="' + tid + '"]');
        if (!v) return;
        Object.keys(data).forEach(function (field) {
          if (field === 'updated_at') return;
          var val = data[field];
          if (val === undefined || val === null || val === '') return;
          var radio = v.querySelector('.choices[data-field="' + field + '"] input[value="' + val + '"]');
          if (radio) { radio.checked = true; setTick(tid, field, 'saved'); return; }
          var ta = v.querySelector('textarea[data-field="' + field + '"]');
          if (ta) { ta.value = val; setTick(tid, field, 'saved'); }
        });
      }).catch(function () { /* leave defaults on a read failure */ });
    }));
  }

  (async function () {
    var claude = window.claude;
    db = (claude && typeof claude.use === 'function') ? await claude.use('db') : null;
    if (!db) { updateProgress(); return; }
    await hydrate();
    enableInputs();
    updateProgress();
  })();
})();"""


def load_mono(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def main():
    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    missing = [c["path"] for c in CLIP_SPECS if not os.path.exists(c["path"])]
    if missing:
        raise FileNotFoundError(f"Missing source clips (nothing is rendered by this script): {missing}")

    loaded = []
    for spec in CLIP_SPECS:
        y, sr = load_mono(spec["path"])
        rms = float(np.sqrt((y.astype(np.float64) ** 2).mean()))
        loaded.append(dict(spec, y=y, sr=sr, src_rms=rms))

    # No single natural reference clip exists across four heterogeneous
    # sources, so level-match everything to the set's own median RMS.
    target_rms = float(np.median([c["src_rms"] for c in loaded]))

    clips = []
    for c in loaded:
        leveled, gain_db = level_match(c["y"], target_rms)
        stem = os.path.splitext(os.path.basename(c["path"]))[0]
        out_path = os.path.join(LISTEN_DIR, f"{stem}.wav")
        sf.write(out_path, leveled, c["sr"], subtype="PCM_16")
        clips.append({
            "kind": c["kind"], "source_path": c["path"], "stem": stem,
            "sibilant_band_ratio": c["ratio"], "note": c["note"],
            "src_sr": c["sr"], "src_rms": round(c["src_rms"], 6),
            "levelmatch_gain_db": round(float(gain_db), 2),
            "wav_path": out_path,
        })

    order = list(range(len(clips)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    for display_i, src_i in enumerate(order, start=1):
        clips[src_i]["id"] = f"clip{display_i:02d}"
        clips[src_i]["label"] = f"Clip {display_i:02d}"
    clips.sort(key=lambda c: c["id"])

    kind_counts = {}
    for c in clips:
        kind_counts[c["kind"]] = kind_counts.get(c["kind"], 0) + 1

    manifest = {
        "experiment": "phase2a_sibilance_sources_bench",
        "purpose": "blind listener check of whether the 4-11kHz sibilant-band ratio tracks an "
                   "audible over-driven 's', mixing upstream's own showcase renders and human "
                   "reference recordings in with our flagged and clean bench-7 pool clips",
        "source_classification_basis": "assets/README.md 'Custom Voices and Audio Samples' table "
                                        "('Reference voice' / 'Supertonic 3 output' columns) -- "
                                        "*_reference.wav is the human cloning-reference recording, "
                                        "*_supertonic3.wav is the zero-shot synthesis",
        "target_rms": target_rms,
        "shuffle_seed": SHUFFLE_SEED,
        "n_clips": len(clips),
        "kind_counts": kind_counts,
        # Source mapping lives here ONLY -- never on the page or in its script.
        "clips": [{k: v for k, v in c.items() if k != "wav_path"} for c in clips],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {LISTEN_DIR}/manifest.json")
    print(f"target_rms (median of {len(clips)} clips) = {target_rms:.6f}")
    for c in clips:
        print(f"  {c['id']}: kind={c['kind']:18s} stem={c['stem']:32s} "
              f"ratio={c['sibilant_band_ratio']} gain={c['levelmatch_gain_db']:+.2f}dB")

    build_html(clips, manifest)


def build_html(clips, manifest):
    def clip_card(c):
        return f"""      <article class="clip" data-clip="{c['id']}">
        <div class="clip-head"><span class="clip-label">{c['label']}</span></div>
        <audio controls preload="metadata" src="data:audio/wav;base64,{b64(c['wav_path'])}"></audio>
        <div class="verdict" data-clip="{c['id']}">
          <div class="q-block">
            <p class="q-label">Q1 &middot; Do you hear an over-driven or harsh &lsquo;s&rsquo; sound in this clip? <span class="tick" data-tick data-clip="{c['id']}" data-field="q1"></span></p>
            <div class="choices" data-field="q1">
              <label><input type="radio" name="{c['id']}_q1" value="yes_clear" disabled> Yes, clearly</label>
              <label><input type="radio" name="{c['id']}_q1" value="maybe" disabled> Maybe, slightly</label>
              <label><input type="radio" name="{c['id']}_q1" value="no_normal" disabled> No, the &lsquo;s&rsquo; sounds normal</label>
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">Q2 &middot; Where, and what does it sound like? <span class="tick" data-tick data-clip="{c['id']}" data-field="q2"></span></p>
            <textarea data-field="q2" data-clip="{c['id']}" disabled rows="2"
              placeholder="e.g. the s in a word near the middle, sounds hissy/sharp/normal"></textarea>
          </div>
        </div>
      </article>"""

    clip_cards = "\n".join(clip_card(c) for c in clips)
    triples_for_js = [{"id": c["id"]} for c in clips]

    HTML = """<title>Sibilance: Engine or Measure?</title>
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
  .wrap { max-width:860px; margin:0 auto; display:flex; flex-direction:column; gap:40px; }
  .eyebrow { font-family:var(--mono); font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); }
  h1 { font-family:var(--sans); font-weight:600; font-size:clamp(28px,4.4vw,40px); letter-spacing:-.02em; line-height:1.12; text-wrap:balance; margin:10px 0 14px; }
  .deck { color:var(--ink-2); max-width:64ch; margin:0; }
  h2 { font-family:var(--sans); font-weight:600; font-size:15px; margin:0 0 4px; }
  .sect-note { font-size:14.5px; color:var(--ink-3); margin:0 0 18px; max-width:64ch; }
  .task { background:var(--accent-soft); border:1px solid var(--accent); border-radius:3px; padding:18px 20px; }
  .task .tag { font-family:var(--mono); font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent); margin:0 0 9px; font-weight:500; }
  .task ol, .task ul { margin:0; padding-left:20px; color:var(--ink); }
  .task li { margin-bottom:7px; }
  .task li:last-child { margin-bottom:0; }
  .task .why { margin:12px 0 0; font-size:14px; color:var(--ink-2); }
  .correction { background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--flag); border-radius:3px; padding:18px 20px; }
  .correction .was { font-family:var(--mono); font-size:11.5px; color:var(--flag); letter-spacing:.06em; text-transform:uppercase; margin:0 0 7px; }
  .correction p { margin:0 0 8px; }
  .correction p:last-child { margin-bottom:0; }

  .clips { display:flex; flex-direction:column; gap:14px; }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:15px 18px 17px; display:flex; flex-direction:column; gap:11px; }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:14px; }
  audio { width:100%; height:36px; display:block; }

  footer { border-top:1px solid var(--rule); padding-top:18px; font-size:14px; color:var(--ink-3); max-width:68ch; }
  code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }
  audio:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

  .progress { font-family:var(--mono); font-size:12px; color:var(--ink-3); margin:0; }
  .progress.ready { color:var(--accent); }
  .progress.nodb { color:var(--flag); }

  .verdict { margin-top:0; padding-top:0; display:flex; flex-direction:column; gap:12px; }
  .q-block { display:flex; flex-direction:column; gap:6px; }
  .q-label { font-family:var(--sans); font-weight:600; font-size:12.5px; color:var(--ink); margin:0; display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .choices { display:flex; flex-direction:column; gap:5px; }
  .choices label { display:flex; align-items:flex-start; gap:7px; font-family:var(--sans); font-size:12.5px; color:var(--ink-2); cursor:pointer; }
  .choices input[type="radio"] { margin-top:2px; accent-color:var(--accent); flex-shrink:0; }
  textarea { width:100%; font-family:var(--sans); font-size:12.5px; color:var(--ink); background:var(--ground); border:1px solid var(--rule); border-radius:3px; padding:8px 10px; resize:vertical; box-sizing:border-box; }
  textarea::placeholder { color:var(--ink-3); }
  textarea:disabled, .choices input:disabled { opacity:.55; }
  .tick { font-family:var(--mono); font-size:10px; letter-spacing:.04em; color:var(--ink-3); font-weight:400; }
  .tick.saving { color:var(--ink-3); }
  .tick.saved { color:var(--accent); }
  .tick.err { color:var(--flag); }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Phase 2a &middot; Sibilance: engine or measure?</div>
    <h1>Is An Over-Driven &lsquo;S&rsquo; Something You Can Hear Here?</h1>
    <p class="deck">Below are __N_CLIPS__ short speech clips, drawn from several different
      synthesis and recording sources. Different clips have different speakers and say
      different sentences &mdash; that is unavoidable given where they come from, and is not
      itself something to judge. The only question that matters for each clip is narrower:
      does the &lsquo;s&rsquo; sound &mdash; specifically &mdash; sound over-driven, harsh, or hissy to you,
      or does it sound like an ordinary &lsquo;s&rsquo;?</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you, and what it decides</p>
      <ol>
        <li>For each clip, listen once and judge only the &lsquo;s&rsquo; / &lsquo;sh&rsquo; sounds: do they
          sound over-driven or harsh, or do they sound normal?</li>
        <li>Each clip stands alone &mdash; you are not comparing clips to each other, and there is
          nothing to reveal or look up after answering.</li>
        <li>Free text: say where in the clip (which word, roughly) and describe what you hear in
          your own words, even if your forced-choice answer is &ldquo;no.&rdquo;</li>
      </ol>
      <p class="why">This decides two things at once: whether a harsh-sounding &lsquo;s&rsquo; is a
        property of this speech engine's fricative synthesis in general (showing up across many
        different voices and sources) or something specific to a couple of presets, and separately,
        whether a spectral-ratio measure we have been using for this actually tracks what a person
        hears, or just responds to how many &lsquo;s&rsquo; sounds are in the sentence regardless of how
        they sound.</li>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Limitation</p>
      <p>These clips come from more than one source, so speakers, sentences, recording conditions
        and languages differ across them &mdash; you will hear different voices saying different
        things. That is expected and does not need to be flagged; judge each clip's &lsquo;s&rsquo; sounds
        on their own terms. Loudness has been matched across all clips so it cannot bias which one
        sounds more &ldquo;forward&rdquo; or present.</p>
    </div>
  </section>

  <section>
    <h2>The clips</h2>
    <p class="sect-note">__N_CLIPS__ clips, blind and in random order. One clip, one judgment about
      the &lsquo;s&rsquo; sounds only.</p>
    <div class="clips">
__CLIP_CARDS__
    </div>
  </section>

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One free-text box across all clips, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Any pattern across the set -- did some clips have a harsher s than others, anything that surprised you"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. Clips are drawn from several synthesis and recording sources, RMS level-matched to the
    set's median loudness before embedding. The mapping of which clip came from where is recorded
    only in this generator's own source list and in <code>manifest.json</code> alongside the leveled
    WAVs &mdash; it does not appear anywhere in this page or its script.
  </footer>
</div>

<script>
__PLAYER_PAUSE_SCRIPT__
</script>
<script>
var CLIPS = __CLIPS_JSON__;
__DB_SCRIPT__
</script>
"""

    HTML = (HTML
            .replace("__N_CLIPS__", str(len(clips)))
            .replace("__CLIP_CARDS__", clip_cards)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(triples_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
