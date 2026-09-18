"""Bench generator: latent-frame attenuation vs. a plain audio-domain
de-esser, as fixes for the sibilant over-drive artifact (bench 7).

Sources:
  `results/phase2a/latent_frame_span.json`  (phase2a_latent_frame_span.py)
    -- Task 2's direct latent-domain manipulation: ONLY the latent frames
       covering each flagged clip's detected sibilants are scaled by a
       factor and the vocoder alone re-decodes (no re-running the denoising
       loop).
  `results/phase2a/deesser_baseline.json`   (phase2a_deesser_baseline.py)
    -- Task 3's boring baseline: a 4-11 kHz band-split gain applied only
       inside the same detected sibilant sample windows (5 ms crossfade),
       residual added back untouched.

Both operate on the SAME two flagged clips (F3 "library", F5 "seashells",
both English) using the SAME sibilant localisation, so the two repair
strategies are compared on identical ground truth, not on separately-tuned
detections.

Per family (library / seashells) the bench shows: the untouched flagged
clip, latent-attenuated at 0.7 and 0.6, audio-de-essed at 0.6 (picked to
match the HF-ratio reduction of the latent factor=0.6 condition -- see
manifest for the exact numbers, matching is by that one measure, not by ear,
and the bench's own listener judgment is what actually decides "matched"),
and the same-sentence CLEAN preset clip (M4 for library, F2 for seashells)
as a hidden hold-your-ear-honest control. All ten clips are level-matched
per family and shuffled into one blind, order-randomised set.

Run from `py/`:
    python3 benches/phase2a_deess_fix_bench.py

Writes leveled WAVs to `results/listening_sets/phase2a_deess_fix/` and the
page to `results/benches/phase2a_deess_fix.html` (gitignored, embeds WAVs as
base64). Does NOT publish -- see phase2a_latent_frame_span.py /
phase2a_deesser_baseline.py for the numeric findings this bench visualizes.
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

FRAME_SPAN_JSON = "results/phase2a/latent_frame_span.json"
DEESS_JSON = "results/phase2a/deesser_baseline.json"
CLEAN_DIR = "results/phase2a/frame_span_audio"
LISTEN_DIR = "results/listening_sets/phase2a_deess_fix"
OUT_HTML = "results/benches/phase2a_deess_fix.html"
FINDINGS_JSON = "results/phase2a/deess_fix.json"

SHUFFLE_SEED = 20270918

FAMILIES = [
    dict(name="library", flagged="F3_library_flagged", clean="M4_library_clean",
         text="The library closes early on Thursday, so bring your books back.",
         latent_factors=[0.7, 0.6], deess_factor=0.6),
    dict(name="seashells", flagged="F5_seashells_flagged", clean="F2_seashells_clean",
         text="She sells seashells by the sea shore every summer morning.",
         latent_factors=[0.7, 0.6], deess_factor=0.6),
]

DB_SCRIPT = r"""(function () {
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  var verdictEls = $all('.verdict');
  var state = {}; var pending = {}; var timers = {}; var db = null;
  verdictEls.forEach(function (v) { var tid = v.getAttribute('data-clip'); state[tid] = {}; pending[tid] = {}; });

  function setTick(tid, field, mode) {
    var el = document.querySelector('[data-tick][data-clip="' + tid + '"][data-field="' + field + '"]');
    if (!el) return;
    el.className = 'tick' + (mode ? ' ' + mode : '');
    el.textContent = mode === 'saving' ? 'saving…' : mode === 'saved' ? 'saved ✓' : mode === 'err' ? 'not saved' : '';
  }
  function clipMeta(tid) { for (var i = 0; i < CLIPS.length; i++) { if (CLIPS[i].id === tid) return CLIPS[i]; } return null; }
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
    }).catch(function () { fields.forEach(function (f) { setTick(tid, f, 'err'); }); });
  }
  function onFieldChange(tid, field, value) {
    state[tid][field] = value; pending[tid][field] = true; setTick(tid, field, 'saving');
    clearTimeout(timers[tid]); timers[tid] = setTimeout(function () { save(tid); }, 600);
  }
  function wireVerdict(v) {
    var tid = v.getAttribute('data-clip');
    $all('.choices', v).forEach(function (group) {
      var field = group.getAttribute('data-field');
      $all('input[type=radio]', group).forEach(function (input) {
        input.addEventListener('change', function () { if (input.checked) onFieldChange(tid, field, input.value); });
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
    if (!db) { el.textContent = 'Responses cannot be saved in this view.'; el.className = 'progress nodb'; return; }
    var n = CLIPS.filter(function (c) { return state[c.id] && state[c.id].q1 && state[c.id].q2; }).length;
    el.textContent = n + ' of ' + CLIPS.length + ' clips fully answered';
    el.className = 'progress ready';
  }
  function enableInputs() { $all('.verdict input, .verdict textarea').forEach(function (el) { el.disabled = false; }); }
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
      }).catch(function () {});
    }));
  }
  (async function () {
    var claude = window.claude;
    db = (claude && typeof claude.use === 'function') ? await claude.use('db') : null;
    if (!db) { updateProgress(); return; }
    await hydrate(); enableInputs(); updateProgress();
  })();
})();"""


def load_wav(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def main():
    if not (os.path.exists(FRAME_SPAN_JSON) and os.path.exists(DEESS_JSON)):
        print(f"Missing {FRAME_SPAN_JSON} or {DEESS_JSON}; run "
              "phase2a_latent_frame_span.py and phase2a_deesser_baseline.py first.")
        return
    with open(FRAME_SPAN_JSON) as f:
        span = json.load(f)
    with open(DEESS_JSON) as f:
        deess = json.load(f)

    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    blind = []
    findings = {"experiment": "phase2a_deess_fix_bench", "families": []}

    for fam in FAMILIES:
        flagged_name, clean_name = fam["flagged"], fam["clean"]
        orig_path = os.path.join("results/phase2a/frame_span_audio", f"{flagged_name}.wav")
        ref_y, ref_sr = load_wav(orig_path)
        ref_rms = float(np.sqrt((ref_y.astype(np.float64) ** 2).mean()))

        fam_findings = dict(name=fam["name"], text=fam["text"], ref_rms=ref_rms, clips=[])

        def add(kind, wav_path, extra):
            y, sr = load_wav(wav_path)
            leveled, gain_db = level_match(y, ref_rms)
            stem = f"{fam['name']}_{kind}"
            out_path = os.path.join(LISTEN_DIR, f"{stem}.wav")
            sf.write(out_path, leveled, sr, subtype="PCM_16")
            rec = dict(family=fam["name"], kind=kind, wav_path=out_path,
                       levelmatch_gain_db=round(gain_db, 2), is_control=(kind == "clean_control"),
                       **extra)
            blind.append(rec)
            fam_findings["clips"].append({k: v for k, v in rec.items() if k != "wav_path"})

        # original flagged (untouched)
        add("original_flagged", orig_path, dict(note="untouched flagged render"))

        # latent-attenuated, two strengths
        lat_rec = span["task2_manipulation"][flagged_name]
        for factor in fam["latent_factors"]:
            s = next(e for e in lat_rec["sweep"] if e["factor"] == factor)
            add(f"latent_atten_{factor}", s["wav_path"], dict(
                method="latent_attenuation", factor=factor,
                sibilant_peak=s["sibilant_peak"], nonsibilant_peak=s["nonsibilant_peak"],
                max_abs_diff_outside_sibilant_region=s.get("max_abs_diff_outside_sibilant_region_vs_factor1"),
            ))

        # audio de-esser, matched factor
        de_rec = deess["results"][flagged_name]
        s = next(e for e in de_rec["sweep"] if e["factor"] == fam["deess_factor"])
        add(f"deess_{fam['deess_factor']}", s["wav_path"], dict(
            method="audio_deesser", factor=fam["deess_factor"],
            sibilant_peak=s["sibilant_peak"], nonsibilant_peak=s["nonsibilant_peak"],
            max_abs_diff_outside_sibilant_region=s["max_abs_diff_outside_sibilant_region_vs_original"],
        ))

        # hidden clean-preset control
        clean_path = os.path.join(CLEAN_DIR, f"{clean_name}.wav")
        add("clean_control", clean_path, dict(note=f"stock {clean_name}, same sentence, never flagged"))

        findings["families"].append(fam_findings)

    order = list(range(len(blind)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    for display_i, src_i in enumerate(order, start=1):
        blind[src_i]["id"] = f"clip{display_i:02d}"
        blind[src_i]["label"] = f"Clip {display_i:02d}"
    blind.sort(key=lambda c: c["id"])

    manifest = {
        "experiment": "phase2a_deess_fix_bench",
        "shuffle_seed": SHUFFLE_SEED,
        "families": [fam["name"] for fam in FAMILIES],
        "clips": [{k: v for k, v in c.items() if k != "wav_path"} for c in blind],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(FINDINGS_JSON, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"Wrote {LISTEN_DIR}/manifest.json and {FINDINGS_JSON}")
    for c in blind:
        print(f"  {c['id']}: family={c['family']} kind={c['kind']}")

    build_html(blind, manifest)


def build_html(blind, manifest):
    def clip_card(c):
        bits = [f"family <code>{c['family']}</code>", f"kind <code>{c['kind']}</code>",
                f"level-match gain {c['levelmatch_gain_db']:+.2f} dB"]
        if c.get("factor") is not None:
            bits.append(f"factor <code>{c['factor']}</code>")
        if c.get("sibilant_peak") is not None:
            bits.append(f"sibilant-frame peak <code>{c['sibilant_peak']:.4f}</code>")
        if c.get("nonsibilant_peak") is not None:
            bits.append(f"non-sibilant peak <code>{c['nonsibilant_peak']:.4f}</code>")
        if c.get("max_abs_diff_outside_sibilant_region") is not None:
            bits.append(f"max |diff| outside the sibilant region vs. the untouched clip "
                        f"<code>{c['max_abs_diff_outside_sibilant_region']:.4f}</code>")
        if c.get("note"):
            bits.append(c["note"])
        reveal_body = "<p>" + " &middot; ".join(bits) + "</p>"
        return f"""      <article class="clip" data-clip="{c['id']}">
        <div class="clip-head"><span class="clip-label">{c['label']}</span></div>
        <audio controls preload="metadata" src="data:audio/wav;base64,{b64(c['wav_path'])}"></audio>
        <div class="verdict" data-clip="{c['id']}">
          <div class="q-block">
            <p class="q-label">Q1 &middot; Do you hear an over-driven or harsh 's' in this clip? <span class="tick" data-tick data-clip="{c['id']}" data-field="q1"></span></p>
            <div class="choices" data-field="q1">
              <label><input type="radio" name="{c['id']}_q1" value="yes_clear" disabled> Yes, clearly</label>
              <label><input type="radio" name="{c['id']}_q1" value="maybe" disabled> Maybe</label>
              <label><input type="radio" name="{c['id']}_q1" value="no" disabled> No</label>
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">Q2 &middot; Does anything else sound wrong or degraded? <span class="tick" data-tick data-clip="{c['id']}" data-field="q2"></span></p>
            <textarea data-field="q2" data-clip="{c['id']}" disabled rows="2"
              placeholder="free text -- a fix that fixes the s and wrecks something else is not a fix"></textarea>
          </div>
        </div>
        <details class="reveal">
          <summary>Reveal (after you answer)</summary>
          {reveal_body}
        </details>
      </article>"""

    clip_cards = "\n".join(clip_card(c) for c in blind)
    triples_for_js = [{"id": c["id"], "is_control": c["is_control"]} for c in blind]

    HTML = """<title>De-ess Fix Bench</title>
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
    <div class="eyebrow">Phase 2a &middot; sibilant fix comparison</div>
    <h1>Latent Attenuation vs. a Plain De-esser</h1>
    <p class="deck">Two candidate fixes for the sibilant over-drive artifact (bench 7), on the
      same two flagged clips (F3 "library", F5 "seashells") and the same detected sibilant
      windows: attenuating only the latent frames that cover the sibilants
      (<code>phase2a_latent_frame_span.py</code>), vs. a boring 4-11 kHz audio-domain de-esser
      applied to those same windows (<code>phase2a_deesser_baseline.py</code>). Every clip is
      English and level-matched.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you, and what it decides</p>
      <ol>
        <li>For each clip: does it have an over-driven / harsh 's' (Q1)? Answer fresh, not by
          recalling which clip you think is which.</li>
        <li>Separately: does anything ELSE sound wrong or degraded (Q2, free text)? A fix that
          quiets the 's' but damages a neighbouring word or the wrong sound is not a fix.</li>
        <li>One clip per sentence family is a hidden, unmodified clean-preset control (different
          voice, never flagged) &mdash; if it gets flagged too, treat every "yes" on this page
          with more suspicion.</li>
      </ol>
      <p class="why">This decides whether the latent-domain manipulation is a usable fix, whether
        the plain de-esser matches or beats it, or whether neither is clean enough to ship.</p>
    </div>
  </section>

  <section>
    <h2>The clips</h2>
    <p class="sect-note">10 clips (2 sentence families &times; 5 conditions each), blind order,
      level-matched per family so loudness cannot give away the condition.</p>
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
          placeholder="Which condition, if any, sounded like a real fix? Anything that surprised you"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. <code>results/phase2a/latent_frame_span.json</code> (latent-domain attenuation) and
    <code>results/phase2a/deesser_baseline.json</code> (audio-domain de-esser) supply the clips;
    both operate on the identical sibilant windows detected on the untouched flagged clip. Clean
    controls are <code>M4_library_clean</code> and <code>F2_seashells_clean</code>, stock renders
    of the same sentences on never-flagged presets. Every clip is level-matched (RMS gain only)
    to its family's untouched-flagged-clip RMS before embedding; see <code>manifest.json</code> in
    <code>results/listening_sets/phase2a_deess_fix/</code> for the exact per-clip gain and the
    blind-id mapping.
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
            .replace("__CLIP_CARDS__", clip_cards)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(triples_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
