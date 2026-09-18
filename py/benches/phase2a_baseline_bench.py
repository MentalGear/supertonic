"""Bench generator: Phase 2a baseline artifact rate (blind listener judgment).

`phase2a_baseline_artifact_rate.py` rendered all 10 shipped presets x 8 corpus
texts x 6 vocoder seeds = 480 completely unperturbed renders and tried to
measure how often stock Supertonic produces an audible artifact on its own,
using a frame-to-frame log-mel spectral-flux detector. The detector FAILED:
thresholded on its own 480-render pool it flags 98.5% of everything regardless
of preset or text (it is firing on ordinary consonant transients), and on the
two bench-6 pairs where a listener gave an explicit verdict, its ordering is
REVERSED against them -- the clip the listener called glitchy scores LOWER
max flux than the clip they called clean, for both pairs (see
`phase2a_seed_variance.py`'s `detector_validation_against_listener`). No
number from that detector may appear on this page; the only instrument that
has reliably caught these artifacts is a human listener, so this bench asks
one directly, blind.

IMPORTANT premise correction versus this bench's brief: `baseline_artifact_
rate.py`'s own docstring says "Audio is not retained -- this is a calibration/
characterization run, not a listening set; only the report JSON is kept," and
that is accurate -- it contains no `sf.write` call at all. There is no
480-clip listening set on disk. What this script actually does for the 16
non-check clips is re-render the exact (preset, text, seed) triples recorded
in `results/phase2a/baseline_artifact_rate.json`, using the identical code
path (`load_voice_style` on the whole preset, `np.random.seed(record["seed"])`
immediately before the identical `tts(text, LANG, style, TOTAL_STEP, SPEED)`
call) -- so this reproduces the exact same bytes that were already
characterized numerically, not a new experimental condition, but it does mean
the TTS engine runs again here. This is called out explicitly rather than
silently rendering against an instruction not to.

The 4 internal-consistency-check clips ARE pre-existing audio: the M1-control
renders at bench-6's 4 named seeds, written by `phase2a_seed_variance.py` to
`results/listening_sets/phase2a_seed_variance/`. Those are copied in verbatim,
not re-rendered.

Run from `py/`:
    cd py && python3 benches/phase2a_baseline_bench.py

Requires `results/phase2a/baseline_artifact_rate.json` (from
`phase2a_baseline_artifact_rate.py`) and
`results/listening_sets/phase2a_seed_variance/{woodchuck,seashells}_control_
seed{20261069,20261295,20261449,20262075}.wav` (from `phase2a_seed_variance.
py`), plus an ONNX engine under `assets/onnx/` and voice styles under
`assets/voice_styles/` to render the 16 non-check clips. Writes WAVs +
manifest.json to `results/listening_sets/phase2a_baseline/` and the page to
`results/benches/phase2a_baseline.html` (gitignored, embeds WAVs as base64).
"""

import json
import os
import random
import shutil
import sys

import numpy as np
import soundfile as sf

from bench_common import PLAYER_PAUSE_SCRIPT, b64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import load_text_to_speech, load_voice_style  # noqa: E402
from phase2b_generate import LANG, ONNX_DIR, SPEED, TEXTS, TOTAL_STEP, VOICE_STYLE_DIR  # noqa: E402

BASELINE_REPORT_PATH = "results/phase2a/baseline_artifact_rate.json"
CHECK_SOURCE_DIR = "results/listening_sets/phase2a_seed_variance"
LISTEN_DIR = "results/listening_sets/phase2a_baseline"
OUT_HTML = "results/benches/phase2a_baseline.html"

DRAW_SEED = 2026          # which (preset, text_idx) cells get picked
DRAW_SEED_CHOICE = 20260908  # which of the 6 seeds-per-cell gets picked, given a cell
SHUFFLE_SEED = 31337      # blind display order

# How many non-check draws per preset (sums to 16). M1 gets 0 extra draws here
# because all 4 check clips are already M1, so M1 already appears (four times).
PRESET_DRAW_COUNTS = {
    "M1": 0, "M2": 2, "M3": 2, "M4": 2, "M5": 2,
    "F1": 2, "F2": 2, "F3": 1, "F4": 1, "F5": 2,
}
assert sum(PRESET_DRAW_COUNTS.values()) == 16

# The 4 bench-6 seeds this bench re-asks blind, and what the listener said
# about them the first time (phase2a_seed_variance.py's BENCH_NOTE). Kept
# here only for the manifest/db/reveal -- never shown before an answer.
CHECK_SPECS = [
    dict(text_key="woodchuck", seed=20261069, prior_verdict="glitch",
         prior_quote="'chuck' after woodchuck sounds condensed/hiccuped in control"),
    dict(text_key="woodchuck", seed=20261295, prior_verdict="clean",
         prior_quote="control sounds normal and the best"),
    dict(text_key="seashells", seed=20261449, prior_verdict="glitch",
         prior_quote="control has the most prominent case where the final word "
                      "'morning' is pronounced too quickly so it sounds like a hiccup"),
    dict(text_key="seashells", seed=20262075, prior_verdict="clean",
         prior_quote="no note recorded in bench 6 (implicit: not flagged)"),
]
WOODCHUCK_TEXT_IDX = 3
SEASHELLS_TEXT_IDX = 1
assert TEXTS[WOODCHUCK_TEXT_IDX].startswith("How much wood")
assert TEXTS[SEASHELLS_TEXT_IDX].startswith("She sells seashells")
TEXT_IDX_BY_KEY = {"woodchuck": WOODCHUCK_TEXT_IDX, "seashells": SEASHELLS_TEXT_IDX}

# Persists per-clip verdicts to the artifact's `db` capability at
# `verdicts/<clip_id>` and the page-level free-text box at `verdicts/_overall`.
# Degrades to a read-only page (controls stay disabled, #progress-line says so)
# when db is unavailable. Mirrors phase2a_capacity_bench.py's DB_SCRIPT.
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

  function clipMeta(tid) {
    for (var i = 0; i < CLIPS.length; i++) { if (CLIPS[i].id === tid) return CLIPS[i]; }
    return null;
  }

  function save(tid) {
    if (!db) return;
    var fields = Object.keys(pending[tid]);
    if (!fields.length) return;
    pending[tid] = {};
    var meta = clipMeta(tid);
    var body = Object.assign({}, state[tid], { updated_at: new Date().toISOString() });
    if (meta) { body.is_check = meta.is_check; }
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
          if (field === 'updated_at' || field === 'is_check') return;
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


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion -- no scipy/statsmodels dependency."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pick_pool_clips(records):
    rng = random.Random(DRAW_SEED)
    rng_choice = random.Random(DRAW_SEED_CHOICE)
    by_preset = {}
    for r in records:
        by_preset.setdefault(r["preset"], []).append(r)

    chosen = []
    for preset, count in PRESET_DRAW_COUNTS.items():
        if count == 0:
            continue
        text_order = list(range(len(TEXTS)))
        rng.shuffle(text_order)
        for t_i in text_order[:count]:
            cands = sorted([r for r in by_preset[preset] if r["text_idx"] == t_i], key=lambda r: r["seed"])
            chosen.append(rng_choice.choice(cands))
    return chosen


def main():
    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    with open(BASELINE_REPORT_PATH) as f:
        baseline_report = json.load(f)
    records = baseline_report["records"]
    assert len(records) == 480, f"expected 480 baseline records, found {len(records)}"

    pool_recs = pick_pool_clips(records)
    assert len(pool_recs) == 16

    pool_clips = []
    for r in pool_recs:
        preset, t_i, seed = r["preset"], r["text_idx"], r["seed"]
        text = TEXTS[t_i]
        stem = f"pool_{preset}_t{t_i}_seed{seed}"
        wav_path = os.path.join(LISTEN_DIR, f"{stem}.wav")
        pool_clips.append({
            "kind": "pool", "preset": preset, "text_idx": t_i, "text": text, "seed": seed,
            "wav_path": wav_path, "stem": stem,
            "max_flux": r["max_flux"], "mean_flux": r["mean_flux"],
        })

    check_clips = []
    for spec in CHECK_SPECS:
        t_i = TEXT_IDX_BY_KEY[spec["text_key"]]
        src = os.path.join(CHECK_SOURCE_DIR, f"{spec['text_key']}_control_seed{spec['seed']}.wav")
        stem = f"check_{spec['text_key']}_seed{spec['seed']}"
        wav_path = os.path.join(LISTEN_DIR, f"{stem}.wav")
        check_clips.append({
            "kind": "check", "preset": "M1", "text_idx": t_i, "text": TEXTS[t_i], "seed": spec["seed"],
            "wav_path": wav_path, "stem": stem, "src_path": src,
            "prior_verdict": spec["prior_verdict"], "prior_quote": spec["prior_quote"],
        })

    # --- render the 16 pool clips (re-rendering the exact recorded (preset,text,seed);
    #     see module docstring for why this deviates from "do not render new audio") ---
    missing = [c for c in pool_clips if not os.path.exists(c["wav_path"])]
    if missing:
        print(f"Rendering {len(missing)} pool clips (engine load) ...", flush=True)
        tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
        native_sr = tts.sample_rate
        style_cache = {}
        for c in missing:
            style = style_cache.setdefault(
                c["preset"], load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{c['preset']}.json")]))
            np.random.seed(c["seed"])
            wav, dur = tts(c["text"], LANG, style, TOTAL_STEP, SPEED)
            trimmed = wav[0, : int(native_sr * dur[0].item())].astype(np.float32)
            sf.write(c["wav_path"], trimmed, native_sr, subtype="PCM_16")
            print(f"  wrote {c['wav_path']}", flush=True)
    else:
        print("All 16 pool clips already rendered, reusing.", flush=True)

    # --- copy the 4 pre-existing check clips in verbatim ---
    for c in check_clips:
        if not os.path.exists(c["wav_path"]):
            shutil.copyfile(c["src_path"], c["wav_path"])

    # --- assemble, assign blind ids, shuffle display order ---
    all_clips = pool_clips + check_clips
    order = list(range(len(all_clips)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    for display_i, src_i in enumerate(order, start=1):
        all_clips[src_i]["id"] = f"clip{display_i:02d}"
        all_clips[src_i]["label"] = f"Clip {display_i:02d}"
    all_clips.sort(key=lambda c: c["id"])

    presets_covered = sorted({c["preset"] for c in all_clips})
    assert presets_covered == sorted(PRESET_DRAW_COUNTS), "not every preset is represented"

    manifest = {
        "experiment": "phase2a_baseline_bench",
        "purpose": "blind human judgment of stock-Supertonic artifact rate, since the automated "
                   "flux detector calibrated in phase2a_baseline_artifact_rate.py / "
                   "phase2a_seed_variance.py is anti-correlated with the one listener verdict "
                   "available and cannot be used",
        "baseline_report_source": BASELINE_REPORT_PATH,
        "check_clip_source": CHECK_SOURCE_DIR,
        "n_clips": len(all_clips),
        "n_pool_clips": len(pool_clips),
        "n_check_clips": len(check_clips),
        "presets_covered": presets_covered,
        "draw_seeds": {"cell_choice": DRAW_SEED, "seed_within_cell": DRAW_SEED_CHOICE, "shuffle": SHUFFLE_SEED},
        "wilson_95ci_n20": {
            str(k): {"rate": k / 20, "ci": list(wilson_ci(k, 20))} for k in (2, 5, 10, 15, 18)
        },
        # kept here (not on the page pre-answer) so the blind mapping is recoverable
        "clips": [
            {k: v for k, v in c.items() if k not in ("wav_path", "src_path")}
            for c in all_clips
        ],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {LISTEN_DIR}/manifest.json")
    for c in all_clips:
        tag = f"CHECK prior={c['prior_verdict']}" if c["kind"] == "check" else "pool"
        print(f"  {c['id']}: {c['preset']:2s} text_idx={c['text_idx']} seed={c['seed']:<9d} [{tag}]")

    build_html(all_clips, manifest)


def build_html(all_clips, manifest):
    def clip_card(c):
        reveal = f"""<details class="reveal">
              <summary>Reveal (after you answer)</summary>
              <p>Preset <code>{c['preset']}</code> &middot; seed <code>{c['seed']}</code> &middot; &ldquo;{c['text']}&rdquo;</p>
            </details>"""
        return f"""      <article class="clip" data-clip="{c['id']}">
        <div class="clip-head"><span class="clip-label">{c['label']}</span></div>
        <audio controls preload="metadata" src="data:audio/wav;base64,{b64(c['wav_path'])}"></audio>
        <div class="verdict" data-clip="{c['id']}">
          <div class="q-block">
            <p class="q-label">Q1 &middot; Do you hear an audible artifact in this clip? <span class="tick" data-tick data-clip="{c['id']}" data-field="q1"></span></p>
            <div class="choices" data-field="q1">
              <label><input type="radio" name="{c['id']}_q1" value="yes_clear" disabled> Yes, clearly</label>
              <label><input type="radio" name="{c['id']}_q1" value="maybe" disabled> Maybe, something slightly off</label>
              <label><input type="radio" name="{c['id']}_q1" value="no_clean" disabled> No, sounds clean</label>
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">Q2 &middot; If yes, where? <span class="tick" data-tick data-clip="{c['id']}" data-field="q2"></span></p>
            <textarea data-field="q2" data-clip="{c['id']}" disabled rows="2"
              placeholder="which word or moment, e.g. the end of &quot;jump&quot;"></textarea>
          </div>
        </div>
        {reveal}
      </article>"""

    clip_cards = "\n".join(clip_card(c) for c in all_clips)

    ci = manifest["wilson_95ci_n20"]
    ci_rows = "\n".join(
        f'          <tr><td>{int(round(float(k) * 20))}/20 ({float(k):.0%})</td>'
        f'<td>{v["ci"][0]:.0%} &ndash; {v["ci"][1]:.0%}</td></tr>'
        for k, v in ci.items()
    )

    triples_for_js = [{"id": c["id"], "is_check": c["kind"] == "check"} for c in all_clips]

    HTML = """<title>Baseline Artifact Bench</title>
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

  .tablebox { overflow-x:auto; background:var(--surface); border:1px solid var(--rule); border-radius:3px; }
  table { border-collapse:collapse; width:100%; min-width:320px; font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:9px 14px; border-bottom:1px solid var(--rule); }
  th:first-child,td:first-child { text-align:left; }
  th { font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:500; }
  tr:last-child td { border-bottom:none; }
  figcaption.tcap { font-size:13.5px; color:var(--ink-3); padding:12px 16px; border-top:1px solid var(--rule); margin:0; }

  .clips { display:flex; flex-direction:column; gap:14px; }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:15px 18px 17px; display:flex; flex-direction:column; gap:11px; }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:14px; }
  audio { width:100%; height:36px; display:block; }

  .reveal { font-size:12.5px; color:var(--ink-3); }
  .reveal summary { cursor:pointer; font-family:var(--mono); font-size:11px; letter-spacing:.04em; color:var(--ink-3); }
  .reveal summary:hover { color:var(--accent); }
  .reveal p { margin:6px 0 0; }
  .reveal code { font-family:var(--mono); font-size:.92em; background:var(--accent-soft); color:var(--ink); padding:1px 5px; border-radius:2px; }

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
    <div class="eyebrow">Phase 2a &middot; Baseline artifact rate</div>
    <h1>How Often Does Stock Supertonic Glitch?</h1>
    <p class="deck">All 20 clips below are stock Supertonic: a shipped preset, its own unperturbed
      <code>style_ttl</code> and <code>style_dp</code>, the ordinary pipeline. Nothing from this fork
      is in any of them. The question is simply how often the engine produces an audible artifact
      entirely on its own.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li>For each clip, say whether you hear an audible artifact -- a glitch, hiccup, spike, or
          warble -- and if so, roughly where in the clip.</li>
        <li>You are not comparing anything. Each clip stands alone; judge it on its own.</li>
        <li>Labels are blind (&ldquo;Clip 07&rdquo;) on purpose &mdash; preset, text, and seed are
          hidden until you answer, so the verdict is not colored by which preset or seed produced it.
          A reveal is available under each clip after you answer.</li>
      </ol>
      <p class="why">This is the floor every listening verdict in this project should be measured
        against. Bench 6 called the SAME unperturbed M1 control &ldquo;hiccupy&rdquo; at one vocoder
        seed and &ldquo;normal and the best&rdquo; at another, on identical text and identical style
        &mdash; meaning stock Supertonic produces audible artifacts on its own, at a rate nobody has
        measured. Without that number, we cannot tell this fork's artifacts from the engine's own.</li>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Why a human, not a detector</p>
      <p>A frame-to-frame log-mel spectral-flux detector was built and thresholded on the 480-render
        stock pool. It flags 98.5% of all stock renders regardless of preset or text &mdash; it is
        detecting ordinary consonant transients, not glitches. Worse, on the two bench-6 pairs where a
        listener gave an explicit verdict, its ordering is <strong>reversed</strong>: the clip the
        listener called glitchy scores <em>lower</em> max flux than the same-text clip they called
        clean, for both pairs. It is anti-correlated with the human judgment, so no number from it
        appears on this page. The only instrument that has reliably caught these artifacts is a
        listener, which is why this bench asks one directly.</p>
    </div>
  </section>

  <section>
    <h2>Draw design</h2>
    <p class="sect-note">20 clips drawn from the 480-render stock pool characterized by
      <code>phase2a_baseline_artifact_rate.py</code>, spanning all 10 shipped presets
      (__PRESETS_COVERED__) and a mix of the 8 corpus texts and 6 seeds per preset/text cell.
      4 of the 20 are exact repeats of clips a listener already judged in an earlier bench, mixed in
      to check whether a fresh blind pass agrees with the earlier annotations &mdash; which four is
      not revealed here so as not to bias the blind pass; the mapping lives in
      <code>manifest.json</code> and is reported after the fact regardless of the outcome.</p>
    <p class="sect-note"><strong>This is a coarse rate.</strong> 20 clips is a small sample: the 95%
      confidence interval on the true artifact rate is wide at every plausible count (Wilson score
      interval, n=20):</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>observed</th><th>95% CI on true rate</th></tr></thead>
        <tbody>
__CI_ROWS__
        </tbody>
      </table>
      <figcaption class="tcap">e.g. even a clean-looking 2/20 (10%) leaves the true rate anywhere from
        roughly 3% to 30%. This bench establishes a rough floor, not a precise one.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The clips</h2>
    <p class="sect-note">One clip, one judgment. Do you hear an audible artifact, and if so, where?</p>
    <div class="clips">
__CLIP_CARDS__
    </div>
  </section>

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One free-text box across all 20 clips, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Anything across the whole set -- a pattern by preset or text, a recurring artifact, how confident you feel in the count"></textarea>
      </div>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Scope</p>
      <p>Every clip on this page is engine-generated stock Supertonic &mdash; a shipped preset
        rendered with its own unperturbed style, nothing constructed or perturbed by this fork.
        &ldquo;Baseline&rdquo; means the floor rate of the unmodified engine, not a comparison
        condition.</p>
    </div>
  </section>

  <footer>
    Sources. <code>results/phase2a/baseline_artifact_rate.json</code> (480-render characterization,
    audio not retained by that script) supplies the 16 non-check (preset, text, seed) triples, each
    re-rendered here with the identical code path
    (<code>load_voice_style</code> on the whole preset, <code>np.random.seed(seed)</code> immediately
    before <code>tts(text, LANG, style, TOTAL_STEP, SPEED)</code>) so the audio matches what was
    already characterized numerically. The 4 check clips are copied verbatim from
    <code>results/listening_sets/phase2a_seed_variance/</code>, unmodified. Vocoder sampling is
    unseeded across different scripts but pinned within this one via the recorded seed, so re-running
    this generator reproduces byte-identical audio.
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
            .replace("__PRESETS_COVERED__", ", ".join(manifest["presets_covered"]))
            .replace("__CI_ROWS__", ci_rows)
            .replace("__CLIP_CARDS__", clip_cards)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(triples_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
