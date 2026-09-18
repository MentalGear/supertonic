"""Bench generator: does a ridge-fitted style_ttl direction de-ess F3 without
turning it into a different speaker?

Source: `phase2a_deessing_direction.py`'s `results/phase2a_deessing/report.json`.
That script rendered F3's library sentence at a ladder of magnitudes along a
ridge-fitted descent direction for the sibilant-peak-ratio measure validated
in `phase2a_sibilance.py`, all at the SAME vocoder seed and the SAME
(unperturbed) style_dp, so duration/alignment is pinned across the ladder.

This bench asks the two open questions CLAUDE.md's listening-bench discipline
requires -- "do you hear an audible artifact in this clip?" and "does this
still sound like the same speaker?" -- never "is the artifact still present",
which primed a listener to re-identify a description rather than report a
fresh perception on bit-identical audio (docs/LISTENING_BENCHES.md; the
woodchuck case). Clips are blind-labeled and level-matched to the baseline's
RMS so loudness cannot leak the condition. One stock, verified-clean, different
-speaker clip (`pool_M3_t2_seed20361017.wav`, from the phase2a-baseline pool,
never flagged) is mixed in unmodified as a hidden control for over-eager
"everything sounds artifacty" bias.

Run from `py/`:
    python3 benches/phase2a_deessing_bench.py

Requires `results/phase2a_deessing/report.json` (from
`phase2a_deessing_direction.py`) with a `ladder` key -- i.e. a run that found
the ratio controllable and the ridge fit generalizing; a run that stopped
early (ratio not controllable, or fit is noise) has nothing to bench and this
script says so and exits.  Writes leveled WAVs to
`results/listening_sets/phase2a_deessing_bench/` and the page to
`results/benches/phase2a_deessing.html` (gitignored, embeds WAVs as base64).
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
from phase2b_speaker_similarity import level_match  # noqa: E402

REPORT_PATH = "results/phase2a_deessing/report.json"
CONTROL_SRC = "results/listening_sets/phase2a_baseline/pool_M3_t2_seed20361017.wav"
LISTEN_DIR = "results/listening_sets/phase2a_deessing_bench"
OUT_HTML = "results/benches/phase2a_deessing.html"

SHUFFLE_SEED = 20270401

# Persists per-clip verdicts to the artifact's `db` capability, verdicts
# collection, mirroring phase2a_baseline_bench.py's DB_SCRIPT exactly except
# for a second question (speaker identity) added per field.
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
    if (meta) { body.is_control = meta.is_control; }
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
    var n = CLIPS.filter(function (c) { return state[c.id] && state[c.id].q1 && state[c.id].q2; }).length;
    el.textContent = n + ' of ' + CLIPS.length + ' clips fully answered';
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
          if (field === 'updated_at' || field === 'is_control') return;
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


def load_wav(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def main():
    if not os.path.exists(REPORT_PATH):
        print(f"No report at {REPORT_PATH}; run phase2a_deessing_direction.py first.")
        return
    with open(REPORT_PATH) as f:
        report = json.load(f)

    if "ladder" not in report:
        print("Report has no 'ladder' key -- the experiment stopped before finding a "
              "controllable/generalizing direction. Nothing to bench.")
        print(f"Verdict recorded: {report.get('verdict')}")
        return

    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    ladder = report["ladder"]
    baseline = next(e for e in ladder if e["mag"] == 0.0)
    ref_rms = baseline["rms"]

    # ---- reference clip (openly labeled, not part of the blind set) ----
    ref_y, ref_sr = load_wav(baseline["path"])
    ref_leveled, _ = level_match(ref_y, ref_rms)
    ref_path = os.path.join(LISTEN_DIR, "reference_F3_unperturbed.wav")
    sf.write(ref_path, ref_leveled, ref_sr, subtype="PCM_16")

    # ---- blind test clips: every ladder magnitude (incl. mag=0.00 baseline) ----
    blind = []
    for e in ladder:
        y, sr = load_wav(e["path"])
        leveled, gain_db = level_match(y, ref_rms)
        stem = f"mag{e['mag']:.3f}"
        out_path = os.path.join(LISTEN_DIR, f"{stem}.wav")
        sf.write(out_path, leveled, sr, subtype="PCM_16")
        blind.append({
            "kind": "ladder", "mag": e["mag"], "wav_path": out_path,
            "ratio": e["ratio"], "sibilant_peak": e["sibilant_peak"],
            "whole_peak": e["whole_peak"], "rms_before_levelmatch": e["rms"],
            "levelmatch_gain_db": round(gain_db, 2),
            "ecapa_cosine_to_anchors_mean": e.get("ecapa_cosine_to_anchors_mean"),
            "is_control": False,
        })

    # ---- hidden clean control: stock M3, verified clean, different speaker/text ----
    ctrl_y, ctrl_sr = load_wav(CONTROL_SRC)
    ctrl_leveled, ctrl_gain_db = level_match(ctrl_y, ref_rms)
    ctrl_path = os.path.join(LISTEN_DIR, "control_M3_clean.wav")
    sf.write(ctrl_path, ctrl_leveled, ctrl_sr, subtype="PCM_16")
    blind.append({
        "kind": "control", "mag": None, "wav_path": ctrl_path,
        "ratio": None, "sibilant_peak": None, "whole_peak": None,
        "rms_before_levelmatch": None, "levelmatch_gain_db": round(ctrl_gain_db, 2),
        "ecapa_cosine_to_anchors_mean": None,
        "is_control": True,
        "note": "stock M3, different preset/text, never flagged in phase2a_baseline_bench "
                "-- checks for over-eager artifact-reporting, not the de-essing question",
    })

    order = list(range(len(blind)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    for display_i, src_i in enumerate(order, start=1):
        blind[src_i]["id"] = f"clip{display_i:02d}"
        blind[src_i]["label"] = f"Clip {display_i:02d}"
    blind.sort(key=lambda c: c["id"])

    manifest = {
        "experiment": "phase2a_deessing_bench",
        "source_report": REPORT_PATH,
        "base_preset": report["base_preset"], "text": report["text"],
        "speed": report["speed"], "total_step": report["total_step"],
        "ridge_r2_heldout": report["ridge_fit"]["r2_heldout_trainmean_baseline"],
        "best_magnitude": report.get("best_magnitude"),
        "shuffle_seed": SHUFFLE_SEED,
        "ref_rms_target": ref_rms,
        "clips": [{k: v for k, v in c.items() if k != "wav_path"} for c in blind],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {LISTEN_DIR}/manifest.json")
    for c in blind:
        tag = "CONTROL" if c["is_control"] else f"mag={c['mag']}"
        print(f"  {c['id']}: {tag}")

    build_html(blind, ref_path, manifest, report)


def build_html(blind, ref_path, manifest, report):
    def clip_card(c):
        if c["is_control"]:
            reveal_body = (f"<p>Hidden clean control &mdash; stock M3, different preset and "
                            f"sentence, never flagged in the phase2a baseline bench. Level-match "
                            f"gain {c['levelmatch_gain_db']:+.2f} dB.</p>")
        else:
            cos = c["ecapa_cosine_to_anchors_mean"]
            cos_str = f"{cos:.4f}" if cos is not None else "n/a"
            reveal_body = (f"<p>Step magnitude <code>{c['mag']:.2f}</code> along the ridge-fitted "
                            f"de-essing direction &middot; sibilant-peak ratio <code>{c['ratio']:.4f}</code> "
                            f"&middot; sibilant-frame peak <code>{c['sibilant_peak']:.4f}</code> "
                            f"&middot; whole-clip peak <code>{c['whole_peak']:.4f}</code> "
                            f"&middot; ECAPA cosine to unperturbed-F3 anchors <code>{cos_str}</code> "
                            f"&middot; level-match gain {c['levelmatch_gain_db']:+.2f} dB.</p>")
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
            <p class="q-label">Q2 &middot; Does this sound like the same speaker as the reference clip above? <span class="tick" data-tick data-clip="{c['id']}" data-field="q2"></span></p>
            <div class="choices" data-field="q2">
              <label><input type="radio" name="{c['id']}_q2" value="same" disabled> Yes, clearly the same speaker</label>
              <label><input type="radio" name="{c['id']}_q2" value="unsure" disabled> Not sure</label>
              <label><input type="radio" name="{c['id']}_q2" value="different" disabled> No, sounds like a different speaker</label>
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">Where / notes (optional) <span class="tick" data-tick data-clip="{c['id']}" data-field="notes"></span></p>
            <textarea data-field="notes" data-clip="{c['id']}" disabled rows="2"
              placeholder="which word or moment, or what changed about the voice"></textarea>
          </div>
        </div>
        <details class="reveal">
          <summary>Reveal (after you answer)</summary>
          {reveal_body}
        </details>
      </article>"""

    clip_cards = "\n".join(clip_card(c) for c in blind)
    triples_for_js = [{"id": c["id"], "is_control": c["is_control"]} for c in blind]

    HTML = """<title>De-essing Direction Bench</title>
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

  .ref-box { background:var(--surface); border:1px solid var(--accent); border-radius:3px; padding:16px 18px; display:flex; flex-direction:column; gap:8px; }
  .ref-box .clip-label { color:var(--accent); }

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
    <div class="eyebrow">Phase 2a &middot; Sibilant de-essing direction</div>
    <h1>Does This Direction Fix F3's Hiss Without Changing the Voice?</h1>
    <p class="deck"><code>phase2a_sibilance.py</code> found a validated measure of the sibilant
      over-drive artifact (sibilant-frame peak / whole-clip peak). <code>phase2a_deessing_direction.py</code>
      fitted a ridge direction in <code>style_ttl</code>'s tangent space that lowers that ratio on
      F3's worst-case sentence, and stepped along it at several magnitudes. This bench asks whether
      that actually sounds fixed, and whether the voice is still recognizably F3.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you, and what it decides</p>
      <ol>
        <li>Listen to the reference clip below once &mdash; it is openly labeled, the unperturbed
          F3 voice on this sentence, and stays the same throughout.</li>
        <li>For each blind clip: does it have an audible artifact (Q1), and does it still sound
          like the SAME SPEAKER as the reference (Q2)? Answer both from what you hear, not from
          what you expect a "corrected" clip to sound like.</li>
        <li>A step magnitude only counts as a fix if Q1 verdicts improve (fewer "yes"/"maybe")
          WITHOUT Q2 verdicts degrading (still "same speaker"). Lowering Q1 by making the clip
          sound like someone else is not a fix.</li>
      </ol>
      <p class="why">Do not ask yourself "is the artifact I was told about still here" &mdash;
        answer both questions fresh for each clip on its own terms.</p>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Why open questions, not a leading one</p>
      <p>A prior bench asked "is the artifact you flagged before still present in this clip?" of a
        clip verified bit-identical to one played earlier under an open question, and got the
        OPPOSITE verdict &mdash; "no" to the leading question, "yes, clearly" to the open one, seven
        minutes apart, same audio. This bench only asks open questions for exactly that reason.</p>
    </div>
  </section>

  <section>
    <h2>Reference</h2>
    <p class="sect-note">Unperturbed F3, &ldquo;__TEXT__&rdquo;, the same clip as this run's
      magnitude-0.00 baseline. Openly labeled &mdash; not part of the blind judgment below.</p>
    <div class="ref-box">
      <span class="clip-label">Reference &middot; F3, unperturbed</span>
      <audio controls preload="metadata" src="data:audio/wav;base64,__REF_B64__"></audio>
    </div>
  </section>

  <section>
    <h2>The clips</h2>
    <p class="sect-note">__N_CLIPS__ clips, blind order, level-matched to the reference's RMS so
      loudness cannot give away the condition. One is an unmodified stock clip from a different,
      never-flagged preset, mixed in as a hidden control on the artifact question.</p>
    <div class="clips">
__CLIP_CARDS__
    </div>
  </section>

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One free-text box across the whole set, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Which magnitude, if any, sounded like a real fix? Anything that surprised you"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. <code>results/phase2a_deessing/report.json</code> (ridge fit held-out R^2
    __R2__, best magnitude by the automatic ratio/ECAPA gate __BEST_MAG__) supplies the ladder
    clips, all rendered at the same vocoder seed and the same (unperturbed) <code>style_dp</code>
    so durations are pinned across the ladder. The clean control is
    <code>results/listening_sets/phase2a_baseline/pool_M3_t2_seed20361017.wav</code>, copied
    verbatim. Every clip on this page is level-matched (RMS gain only) to the reference's RMS
    before embedding; see <code>manifest.json</code> in
    <code>results/listening_sets/phase2a_deessing_bench/</code> for the exact per-clip gain and
    the blind-id mapping.
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
            .replace("__TEXT__", manifest["text"])
            .replace("__REF_B64__", b64(ref_path))
            .replace("__N_CLIPS__", str(len(blind)))
            .replace("__CLIP_CARDS__", clip_cards)
            .replace("__R2__", f"{manifest['ridge_r2_heldout']:.4f}")
            .replace("__BEST_MAG__", str(manifest["best_magnitude"]))
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(triples_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
