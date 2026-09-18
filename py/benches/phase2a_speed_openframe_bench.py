"""Bench generator: Phase 2a -- speed root-cause, open-frame redo.

`phase2a_speed_rootcause_bench.py` (bench 8) asked, per clip: "Is the
artifact you flagged before still present in this clip?" For the woodchuck
sentence the listener answered "no" at every speed. Seven minutes later, in
bench 7, the SAME clip -- verified bit-identical, waveform correlation
1.00000000 -- was played blind under the open question "do you hear an
audible artifact in this clip?" and came back "yes, clearly -- chuck too
condensed". The leading question suppressed a real detection, so bench 8's
woodchuck ladder is void and the speed finding currently rests on one
sentence (seashells).

This bench redoes the woodchuck speed ladder (1.05 / 1.00 / 0.90, flagged
seed 20261069) under an open, non-leading frame, and includes the seashells
ladder (flagged seed 20261449) alongside it as an independent open-frame
confirmation of the result that is currently carrying the finding.

Design differences from bench 8, all deliberate:
  - No speed value is shown anywhere in visible page text or in the page's
    embedded JS. The speed-to-clip mapping lives ONLY in this script's
    manifest.json and in the artifact's db (keyed by opaque clip id).
  - Clip order is randomised per sentence group (fixed order seed, so the
    build is reproducible, but nothing on the page indicates order = speed).
  - Sentence text itself is never printed: both source sentences contain the
    two words this bench must not name ("woodchuck ... chuck ... chuck ..."
    and "... summer morning."), so sentences are labelled generically
    ("Sentence 1" / "Sentence 2") rather than quoted.
  - The question is the plain, open one from bench 7 ("do you hear an
    audible artifact in this clip?"), never "is the artifact you flagged
    before still present" -- see CLAUDE.md's bullet on this exact failure.
  - No text on the page states or implies which direction (faster/slower)
    is expected to help, or that this is a re-test of a prior finding.

Reuses the existing renders from `phase2a_speed_rootcause.py`
(`results/phase2a/speed_rootcause.json` + the WAVs already on disk under
`results/listening_sets/phase2a_speed_rootcause/`) -- nothing is rendered
here, only read, RMS-level-matched, and reordered.

Run from `py/`:
    python3 benches/phase2a_speed_openframe_bench.py

Writes leveled/reordered WAVs + manifest.json to
`results/listening_sets/phase2a_speed_openframe/` and the page to
`results/benches/phase2a_speed_openframe.html` (gitignored: embeds WAVs as
base64). Not published as an Artifact.
"""

import json
import os
import random

import numpy as np
import soundfile as sf

from bench_common import PLAYER_PAUSE_SCRIPT, b64

SRC_JSON = "results/phase2a/speed_rootcause.json"
SRC_DIR = "results/listening_sets/phase2a_speed_rootcause"
LISTEN_DIR = "results/listening_sets/phase2a_speed_openframe"
OUT_HTML = "results/benches/phase2a_speed_openframe.html"

FLAGGED_SEED = {"woodchuck": 20261069, "seashells": 20261449}
SPEEDS = [1.05, 1.00, 0.90]

# Fixed so the build is reproducible; the resulting order is not revealed
# anywhere on the page or in its embedded JS -- only in manifest.json / db.
ORDER_SEED = 20260918

Q1_OPTIONS = [
    ("yes_clear", "Yes, clearly"),
    ("maybe", "Maybe, something slightly off"),
    ("no", "No, sounds clean"),
]

# Persists per-clip verdicts to the artifact's `db` capability at
# `verdicts/<clip_id>` and a page-level overall free-text box at
# `verdicts/_overall`. No speed or sentence-text field is ever attached to a
# saved doc from this page's own JS -- CLIPS below carries only an opaque id
# and a generic group label, so the mapping stays out of the page entirely,
# per the brief ("Keep the speed-to-clip mapping in the manifest and db
# only" -- meaning: written into the db by us, from the manifest, never by
# the browser echoing it back from page source).
DB_SCRIPT = r"""(function () {
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }

  var verdictEls = $all('.verdict[data-clip]');
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
    if (meta) { body.group = meta.group; }
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
          if (field === 'updated_at' || field === 'group') return;
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


def level_match(w, ref_rms):
    r = float(np.sqrt((w.astype(np.float64) ** 2).mean()))
    g = ref_rms / max(r, 1e-12)
    return (w * g).astype(np.float32), 20 * np.log10(g)


def find_row(report, sentence, seed, speed):
    for r in report["speed_sweep"]:
        if r["sentence"] == sentence and r["seed"] == seed and abs(r["speed"] - speed) < 1e-6:
            return r
    raise KeyError((sentence, seed, speed))


def build_group(report, sentence, group_label, group_no, order_rng):
    seed = FLAGGED_SEED[sentence]
    rows = [find_row(report, sentence, seed, sp) for sp in SPEEDS]

    wavs = []
    sr = None
    for r in rows:
        y, sr = sf.read(os.path.join(SRC_DIR, r["trimmed_file"]))
        wavs.append(y.astype(np.float32))

    # RMS level-match the three clips in this group to each other: target is
    # their own mean RMS (not any single clip), so no clip is the silent
    # "reference" and none of the three is favoured.
    raw_rms = [float(np.sqrt((y.astype(np.float64) ** 2).mean())) for y in wavs]
    target_rms = float(np.mean(raw_rms))

    slots = []
    for i, (r, y, rrms) in enumerate(zip(rows, wavs, raw_rms)):
        leveled, gain_db = level_match(y, target_rms)
        speed = r["speed"]
        slots.append({
            "sentence": sentence,
            "seed": seed,
            "speed": speed,
            "trimmed_file": r["trimmed_file"],
            "raw_rms": rrms,
            "gain_db": gain_db,
            "leveled": leveled,
            "sr": sr,
        })

    # Randomise display order; the order itself is what hides which slot is
    # which speed -- nobody looking at the page can infer speed from
    # position, since position is shuffled independently of speed each
    # build (but deterministically, from ORDER_SEED, for reproducibility).
    order_rng.shuffle(slots)

    os.makedirs(LISTEN_DIR, exist_ok=True)
    out_slots = []
    for i, s in enumerate(slots):
        clip_id = f"g{group_no}_clip{i + 1}"
        fname = f"g{group_no}_clip{i + 1}.wav"
        out_path = os.path.join(LISTEN_DIR, fname)
        sf.write(out_path, s["leveled"], s["sr"], subtype="PCM_16")
        out_slots.append({
            "id": clip_id,
            "display_label": f"Clip {i + 1}",
            "sentence": sentence,
            "speed": s["speed"],
            "seed": seed,
            "source_trimmed_file": s["trimmed_file"],
            "raw_rms": s["raw_rms"],
            "levelmatch_gain_db": s["gain_db"],
            "leveled_wav_path": out_path,
        })

    return {
        "group_no": group_no,
        "group_id": f"group{group_no:02d}",
        "display_label": group_label,
        "sentence": sentence,
        "target_rms": target_rms,
        "slots": out_slots,
    }


def clip_card(slot):
    cid = slot["id"]
    q1_choices = "\n".join(
        f'              <label><input type="radio" name="{cid}_q1" value="{val}" disabled> {label}</label>'
        for val, label in Q1_OPTIONS)
    return f"""      <article class="clip" data-clip="{cid}">
        <div class="clip-head"><span class="clip-label">{slot['display_label']}</span></div>
        <audio controls preload="metadata" src="data:audio/wav;base64,{b64(slot['leveled_wav_path'])}"></audio>
        <div class="verdict" data-clip="{cid}">
          <div class="q-block">
            <p class="q-label">1&middot; Do you hear an audible artifact in this clip? <span class="tick" data-tick data-clip="{cid}" data-field="q1"></span></p>
            <div class="choices" data-field="q1">
{q1_choices}
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">2&middot; If yes, where and what? <span class="tick" data-tick data-clip="{cid}" data-field="q2"></span></p>
            <textarea data-field="q2" data-clip="{cid}" disabled rows="2"
              placeholder="a word or moment that stood out, if any"></textarea>
          </div>
        </div>
      </article>"""


def group_section(g):
    slots_html = "\n".join(clip_card(s) for s in g["slots"])
    return f"""  <section class="group">
    <h2>{g['display_label']}</h2>
    <p class="sect-note">Three clips of the same sentence, rendered under settings that differ in one
      respect. Order is randomised and not otherwise indicated -- judge each clip on its own.</p>
    <div class="clips">
{slots_html}
    </div>
  </section>"""


def main():
    with open(SRC_JSON) as f:
        report = json.load(f)

    order_rng = random.Random(ORDER_SEED)
    groups = [
        build_group(report, "woodchuck", "Sentence 1", 1, order_rng),
        build_group(report, "seashells", "Sentence 2", 2, order_rng),
    ]

    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    manifest = {
        "experiment": "phase2a_speed_openframe_bench",
        "purpose": "open-frame (non-leading) redo of the woodchuck speed ladder from bench 8, "
                   "whose leading question ('is the artifact you flagged before still present?') "
                   "suppressed a real detection on bit-identical audio; seashells ladder included "
                   "as an independent open-frame confirmation of the finding it currently carries",
        "source_json": SRC_JSON,
        "source_wav_dir": SRC_DIR,
        "speeds_tested": SPEEDS,
        "order_seed": ORDER_SEED,
        "level_match": "RMS level-matched within each group to the group's own mean RMS "
                       "across its 3 clips (not to any single clip)",
        "q1_options": [v for _, v in Q1_OPTIONS],
        "groups": [
            {
                "group_id": g["group_id"],
                "display_label": g["display_label"],
                "sentence": g["sentence"],
                "target_rms": g["target_rms"],
                "slots": g["slots"],
            }
            for g in groups
        ],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {LISTEN_DIR}/manifest.json")
    for g in groups:
        print(f"  {g['group_id']} ({g['sentence']}):", end="")
        for s in g["slots"]:
            print(f"  {s['id']}->speed={s['speed']:.2f}(gain{s['levelmatch_gain_db']:+.2f}dB)", end="")
        print()

    # CLIPS embedded on the page carries only opaque id + generic group --
    # no speed, no sentence text, no source filename.
    clips_for_js = [{"id": s["id"], "group": g["group_id"]} for g in groups for s in g["slots"]]

    groups_html = "\n".join(group_section(g) for g in groups)

    HTML = """<title>Speed Open-Frame Bench</title>
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

  .group { display:flex; flex-direction:column; gap:12px; padding-top:20px; border-top:1px solid var(--rule); }
  .group:first-of-type { border-top:none; padding-top:0; }
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
    <div class="eyebrow">Phase 2a &middot; Speed, open frame</div>
    <h1>Do you hear an artifact?</h1>
    <p class="deck">Below are two sentences, each rendered three times under settings that differ in one
      respect. Within each sentence, the three clips are RMS level-matched to each other and shown in a
      randomised order that is not indicated anywhere on this page.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li>For each clip, just listen and say whether you hear an audible artifact -- some
          glitch, hiccup, or moment that sounds wrong, of any kind. Not "which do you prefer",
          and not a check against anything described elsewhere.</li>
        <li>If you hear something, say where and what, in your own words.</li>
      </ol>
      <p class="why">This decides whether a compression-sounding artifact found on one sentence
        (currently the only evidence for the finding) also shows up on a second, independent
        sentence -- i.e. whether the finding holds on more than one sentence.</p>
    </div>
  </section>

__GROUPS_HTML__

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One free-text box across all clips, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Anything that stood out across the set, any pattern you noticed, how confident you feel"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. <code>results/phase2a/speed_rootcause.json</code> plus the WAVs under
    <code>results/listening_sets/phase2a_speed_rootcause/</code> supply the underlying renders; this
    page only reads, RMS-level-matches (within each group, to the group's own mean RMS), and reorders
    them. The mapping from clip to render setting is kept out of this page and lives only in
    <code>results/listening_sets/phase2a_speed_openframe/manifest.json</code> and the saved responses.
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
            .replace("__GROUPS_HTML__", groups_html)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(clips_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
