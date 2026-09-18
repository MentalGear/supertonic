"""Bench generator: Phase 2a preset-span vs random-control -- audible? and
does it read as a different VOICE or as damage?

`phase2b_generate_presetspan.py` built two conditions at matched
dimensionality and amplitude (eps=0.20, Frobenius perturbation norm target
0.98 over the 24 active rows): (A) preset-span, perturbing along the 9
tangent-projected preset-difference directions P_i - M1; (B) random control,
an isotropic random subspace of the same rank. A probe showed A is far more
RECOVERABLE from audio than B (mean per-component R^2 0.938 vs 0.759). But
bench 6 already showed recoverability and audibility are decoupled -- its
highest-scoring sample, R^2 0.991, was indistinguishable from the unperturbed
control by ear. So this bench asks the two things R^2 cannot: is a preset-span
perturbation AUDIBLE, and if so does it sound like a different voice (a
feature) or like damage (a bug)? That is the distinction Phase 3 needs: an
axis that is recoverable but ugly is not usable.

Design
------
5 of the corpus's 8 texts (0, 1, 3, 5, 6 -- picked for phonetic variety). Per
text: one preset-span sample and one random-control sample, both drawn from
the corpus's held-out ("test") split so nothing here was seen by the R^2
probe's own fit, plus a freshly rendered unperturbed M1 reference at the same
settings (speed=1.05, TOTAL_STEP=8, lang=en -- read from the corpus manifest,
NOT this fork's speed=1.0 default; see phase2b_generate_presetspan.py's
"SPEED IS PINNED TO 1.05" docstring section and CLAUDE.md). The reference is
rendered at the SAME VOCODER SEED as the group's preset-span sample (the only
one of the two seeds we can match without re-rendering the whole corpus,
since preset-span and random-control draw from one continuing idx_global and
so never share a seed with each other); the random-control sample keeps its
own corpus seed. This is noted on the page and in the manifest rather than
silently presented as a fully seed-matched triple.

The corpus only wrote 16 kHz embedding copies (`audio16k/`, no full-rate
directory exists on disk -- confirmed by listing `results/phase2b_presetspan/
{preset_span,random_control}/`, both containing only `audio16k/`). The
reference is therefore also rendered and resampled to 16 kHz so all three
clips in a group share quality; this is noted on the page.

All three clips in a group are RMS level-matched to the group's reference
(magnitude alone buys up to +4.9 dB in style_ttl perturbations -- see
CLAUDE.md). The reference is presented first and openly labelled; the other
two are blind, order randomised per group, with NO label, filename, or
surrounding text revealing which is which -- that mapping lives only in
`manifest.json` and the artifact's `db`.

Questions are open, never leading (CLAUDE.md: a leading "is the artifact you
flagged before still present" question suppressed a real detection on
bit-identical audio). Q1 is a forced choice among four descriptions of what
was heard (not asked to compare quality); Q2 is free text.

Run from `py/`:
    python3 benches/phase2a_presetspan_bench.py

Requires `results/phase2b_presetspan/{preset_span,random_control}/
{manifest.json,audio16k/*.wav}` (from `phase2b_generate_presetspan.py`), plus
an ONNX engine under `assets/onnx/` and voice styles under
`assets/voice_styles/` to render the 5 reference clips (cheap; skipped if
already rendered). Writes WAVs + manifest.json to
`results/listening_sets/phase2a_presetspan/` and the page to
`results/benches/phase2a_presetspan.html` (gitignored, embeds WAVs as
base64). Not published per task instructions.
"""

import json
import os
import random
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from bench_common import PLAYER_PAUSE_SCRIPT, b64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import Style, load_text_to_speech, load_voice_style  # noqa: E402
from phase2b_generate import EMBED_SR, LANG, ONNX_DIR, TEXTS, TOTAL_STEP, VOICE_STYLE_DIR  # noqa: E402
from phase2b_generate_presetspan import BASE_PRESET, SPEED  # noqa: E402

PRESET_SPAN_DIR = "results/phase2b_presetspan/preset_span"
RANDOM_CONTROL_DIR = "results/phase2b_presetspan/random_control"
GEOMETRY_PATH = "results/phase2b_presetspan/geometry.json"
LISTEN_DIR = "results/listening_sets/phase2a_presetspan"
OUT_HTML = "results/benches/phase2a_presetspan.html"

# 5 of the corpus's 8 texts: the pangram, a sibilant-heavy sentence, the
# woodchuck tongue-twister (this project's most-scrutinized artifact case),
# and two longer/more varied sentences.
TEXT_IDXS = [0, 1, 3, 5, 6]

SAMPLE_SEED = 20270913   # which held-out (test-split) sample represents each text, per condition
ORDER_SEED = 20270914    # blind A/B display order per group

# Bench 9 forced all ten listener verdicts into "same voice with something
# wrong in it" because that was the closest available box for an outcome the
# original 4-way option set did not offer at all: same voice, audibly
# different, nothing wrong. The listener's free-text notes on every clip
# contradicted the forced choice ("not wrong, all fine, only ... different
# strength of pronunciation"; "even better than original"; "this feels like
# the best as almost no distortion"), which is how the real verdict survived
# a defective option set. Do not simplify this back to 4 options — the
# neutral-difference choice is load-bearing, not decorative.
Q1_OPTIONS = [
    ("different_clean", "A different voice, and clean speech"),
    ("same_different", "The same voice, audibly different, nothing wrong with it"),
    ("same_wrong", "The same voice, with something wrong in it"),
    ("different_wrong", "A different voice AND something wrong"),
    ("no_difference", "No difference I can hear"),
]

# Persists per-clip verdicts to the artifact's `db` capability at
# `verdicts/<clip_id>` and the page-level free-text box at `verdicts/_overall`.
# Mirrors phase2a_baseline_bench.py / phase2a_deessing_bench.py's DB_SCRIPT,
# with a 5-way forced-choice Q1 instead of a 3-way one. `condition` is stashed
# into the saved doc for later analysis (matching those scripts' is_check /
# is_control pattern) but is never rendered into visible page text.
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
    if (meta) { body.condition = meta.condition; body.group = meta.group; }
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
    el.textContent = n + ' of ' + CLIPS.length + ' blind clips answered';
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
          if (field === 'updated_at' || field === 'condition' || field === 'group') return;
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
    """Rescale w so its RMS equals ref_rms. Returns (leveled, gain_db). Verbatim
    body shared with phase2a_capacity_bench.py / phase2a_speed_rootcause_bench.py
    / phase2b_speaker_similarity.py -- duplicated here rather than imported to
    avoid pulling phase2b_speaker_similarity's torch/ECAPA import chain for a
    3-line function."""
    r = float(np.sqrt((w.astype(np.float64) ** 2).mean()))
    g = ref_rms / max(r, 1e-12)
    return (w * g).astype(np.float32), 20 * np.log10(g)


def load_wav16(path):
    y, sr = sf.read(path, dtype="float32")
    assert sr == EMBED_SR, f"{path}: expected {EMBED_SR} Hz, found {sr}"
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def pick_groups(ps_manifest, rc_manifest):
    rng = random.Random(SAMPLE_SEED)
    groups = []
    for text_idx in TEXT_IDXS:
        ps_cands = sorted(
            [r for r in ps_manifest["records"] if r["text_idx"] == text_idx and r["split"] == "test"],
            key=lambda r: r["idx"])
        rc_cands = sorted(
            [r for r in rc_manifest["records"] if r["text_idx"] == text_idx and r["split"] == "test"],
            key=lambda r: r["idx"])
        assert ps_cands and rc_cands, f"text_idx {text_idx}: no held-out sample in one condition"
        ps_rec = dict(rng.choice(ps_cands))
        rc_rec = dict(rng.choice(rc_cands))
        groups.append({"text_idx": text_idx, "text": TEXTS[text_idx],
                        "preset_span": ps_rec, "random_control": rc_rec})
    return groups


def ensure_references(groups):
    os.makedirs(LISTEN_DIR, exist_ok=True)
    for g in groups:
        ref_seed = g["preset_span"]["seed"]  # seed-matched to this group's preset-span sample; see docstring
        g["ref_seed"] = ref_seed
        g["ref_path"] = os.path.join(LISTEN_DIR, f"ref_t{g['text_idx']}_seed{ref_seed}.wav")

    missing = [g for g in groups if not os.path.exists(g["ref_path"])]
    if not missing:
        print("All 5 reference clips already rendered, reusing.", flush=True)
        return

    print(f"Rendering {len(missing)} reference clips (unperturbed {BASE_PRESET}) ...", flush=True)
    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    base_style = load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{BASE_PRESET}.json")])
    base_ttl = base_style.ttl.astype(np.float32)
    dp_ref = base_style.dp.copy()
    for g in missing:
        np.random.seed(g["ref_seed"])  # vocoder latent RNG -- same seed as this group's preset-span sample
        wav, dur = tts(g["text"], LANG, Style(base_ttl.copy(), dp_ref.copy()), TOTAL_STEP, SPEED)
        trimmed = wav[0, : int(tts.sample_rate * dur[0].item())].astype(np.float32)
        w16 = resample_poly(trimmed, EMBED_SR, tts.sample_rate).astype(np.float32)
        sf.write(g["ref_path"], w16, EMBED_SR, subtype="PCM_16")
        print(f"  wrote {g['ref_path']}", flush=True)


def build_group_audio(groups):
    for g in groups:
        ref_wav = load_wav16(g["ref_path"])
        ref_rms = float(np.sqrt((ref_wav.astype(np.float64) ** 2).mean()))
        g["ref_rms"] = ref_rms

        for cond, src_dir in (("preset_span", PRESET_SPAN_DIR), ("random_control", RANDOM_CONTROL_DIR)):
            rec = g[cond]
            src = os.path.join(src_dir, "audio16k", rec["file"])
            wav = load_wav16(src)
            leveled, gain_db = level_match(wav, ref_rms)
            out_path = os.path.join(LISTEN_DIR, f"g{g['text_idx']}_{cond}_idx{rec['idx']:05d}.wav")
            sf.write(out_path, leveled, EMBED_SR, subtype="PCM_16")
            rec["wav_path"] = out_path
            rec["levelmatch_gain_db"] = gain_db
            rec["src_rms"] = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))


def assign_blind(groups):
    rng = random.Random(ORDER_SEED)
    clip_no = 0
    for gi, g in enumerate(groups, start=1):
        g["group_id"] = f"group{gi:02d}"
        conds = ["preset_span", "random_control"]
        rng.shuffle(conds)
        slots = []
        for slot_letter, cond in zip(("A", "B"), conds):
            clip_no += 1
            rec = g[cond]
            rec["id"] = f"clip{clip_no:02d}"
            rec["slot_label"] = f"Clip {slot_letter}"
            rec["condition"] = cond
            slots.append(rec)
        g["slots"] = slots


def main():
    with open(os.path.join(PRESET_SPAN_DIR, "manifest.json")) as f:
        ps_manifest = json.load(f)
    with open(os.path.join(RANDOM_CONTROL_DIR, "manifest.json")) as f:
        rc_manifest = json.load(f)
    with open(GEOMETRY_PATH) as f:
        geometry = json.load(f)
    assert ps_manifest["texts"] == TEXTS == rc_manifest["texts"]
    assert ps_manifest["speed"] == SPEED == rc_manifest["speed"] == 1.05
    assert ps_manifest["total_step"] == rc_manifest["total_step"] == TOTAL_STEP
    assert ps_manifest["synthesis_sample_rate"] == rc_manifest["synthesis_sample_rate"] == 44100
    assert ps_manifest["embed_sample_rate"] == rc_manifest["embed_sample_rate"] == EMBED_SR
    # confirm only 16 kHz copies exist on disk -- no full-rate audio directory
    for d in (PRESET_SPAN_DIR, RANDOM_CONTROL_DIR):
        entries = set(os.listdir(d))
        unexpected = entries - {"audio16k", "manifest.json", "subspace.npz", "wavlm_feats.npz"}
        assert not unexpected, f"{d}: unexpected entries {unexpected}, re-check for a full-rate audio dir"
        assert entries & {"audio16k"}, f"{d}: no audio16k directory found"

    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    groups = pick_groups(ps_manifest, rc_manifest)
    ensure_references(groups)
    build_group_audio(groups)
    assign_blind(groups)

    ratios = {p: geometry["perturbation_frobenius_norm_target"] / v
              for p, v in geometry["natural_scale_frobenius"].items()}

    manifest = {
        "experiment": "phase2a_presetspan_bench",
        "purpose": "blind human judgment of whether a preset-span style_ttl perturbation "
                   "(eps=0.20, matched-rank random control) is audible, and whether it reads as "
                   "a different voice or as damage",
        "preset_span_source": os.path.join(PRESET_SPAN_DIR, "manifest.json"),
        "random_control_source": os.path.join(RANDOM_CONTROL_DIR, "manifest.json"),
        "geometry_source": GEOMETRY_PATH,
        "base_preset": BASE_PRESET,
        "rank": ps_manifest["rank"],
        "eps": ps_manifest["eps"],
        "perturbation_frobenius_norm_target": geometry["perturbation_frobenius_norm_target"],
        "natural_scale_frobenius_per_preset": geometry["natural_scale_frobenius"],
        "target_over_natural_ratio_per_preset": ratios,
        "target_over_natural_ratio_mean": sum(ratios.values()) / len(ratios),
        "speed": SPEED, "total_step": TOTAL_STEP, "lang": LANG,
        "synthesis_sample_rate": 44100, "embed_sample_rate": EMBED_SR,
        "audio_note": "corpus only wrote 16 kHz embedding copies (audio16k/); no full-rate "
                       "audio exists on disk for either condition. References were rendered "
                       "and resampled to 16 kHz to match, and all clips are judged at 16 kHz.",
        "seed_note": "reference is rendered at the same vocoder seed as the group's "
                      "preset-span sample; random-control keeps its own corpus seed "
                      "(preset-span and random-control never share a seed in the corpus, "
                      "since idx_global continues across both conditions)",
        "sample_seed": SAMPLE_SEED, "order_seed": ORDER_SEED,
        "text_idxs": TEXT_IDXS,
        "q1_options": [v for _, v in Q1_OPTIONS],
        "groups": [
            {
                "group_id": g["group_id"], "text_idx": g["text_idx"], "text": g["text"],
                "reference": {"seed": g["ref_seed"], "rms": g["ref_rms"], "wav_path": g["ref_path"]},
                "slots": [dict(s) for s in g["slots"]],
            }
            for g in groups
        ],
    }
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {LISTEN_DIR}/manifest.json")
    for g in groups:
        print(f"  {g['group_id']} text_idx={g['text_idx']}: "
              f"ref seed={g['ref_seed']}", end="")
        for s in g["slots"]:
            print(f"  {s['id']}={s['slot_label']}->{s['condition']}"
                  f"(idx{s['idx']:05d} seed{s['seed']} gain{s['levelmatch_gain_db']:+.2f}dB)", end="")
        print()

    build_html(groups, manifest)


def build_html(groups, manifest):
    def q1_choices(cid):
        opts = "\n".join(
            f'              <label><input type="radio" name="{cid}_q1" value="{val}" disabled> {label}</label>'
            for val, label in Q1_OPTIONS)
        return opts

    def slot_card(s):
        return f"""      <article class="clip" data-clip="{s['id']}">
        <div class="clip-head"><span class="clip-label">{s['slot_label']}</span></div>
        <audio controls preload="metadata" src="data:audio/wav;base64,{b64(s['wav_path'])}"></audio>
        <div class="verdict" data-clip="{s['id']}">
          <div class="q-block">
            <p class="q-label">Q1 &middot; Compared with the reference, what do you hear? <span class="tick" data-tick data-clip="{s['id']}" data-field="q1"></span></p>
            <div class="choices" data-field="q1">
{q1_choices(s['id'])}
            </div>
          </div>
          <div class="q-block">
            <p class="q-label">Q2 &middot; Notes <span class="tick" data-tick data-clip="{s['id']}" data-field="q2"></span></p>
            <textarea data-field="q2" data-clip="{s['id']}" disabled rows="2"
              placeholder="a word or moment that stood out, e.g. &quot;flattens on 'seashells'&quot;"></textarea>
          </div>
        </div>
      </article>"""

    def group_section(g, gi):
        slots_html = "\n".join(slot_card(s) for s in g["slots"])
        return f"""  <section class="group">
    <h2>Group {gi} &middot; &ldquo;{g['text']}&rdquo;</h2>
    <p class="sect-note">One reference, then two blind clips in random order. Judge each blind clip
      against the reference above it.</p>
    <article class="clip anchor">
      <div class="clip-head"><span class="clip-label">Reference (unperturbed {manifest['base_preset']})</span></div>
      <audio controls preload="metadata" src="data:audio/wav;base64,{b64(g['ref_path'])}"></audio>
    </article>
    <div class="clips">
{slots_html}
    </div>
  </section>"""

    groups_html = "\n".join(group_section(g, gi) for gi, g in enumerate(groups, start=1))

    triples_for_js = []
    for g in groups:
        for s in g["slots"]:
            triples_for_js.append({"id": s["id"], "condition": s["condition"], "group": g["group_id"]})

    ratio_rows = "\n".join(
        f'          <tr><td>{p}</td><td>{v:.4f}</td><td>{manifest["target_over_natural_ratio_per_preset"][p]:.3f}&times;</td></tr>'
        for p, v in manifest["natural_scale_frobenius_per_preset"].items()
    )

    HTML = """<title>Preset-Span Voice Or Damage</title>
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

  .group { display:flex; flex-direction:column; gap:12px; padding-top:8px; border-top:1px solid var(--rule); }
  .group:first-of-type { border-top:none; padding-top:0; }
  .clips { display:flex; flex-direction:column; gap:14px; }
  .clip { background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:15px 18px 17px; display:flex; flex-direction:column; gap:11px; }
  .clip.anchor { border-color:var(--accent); border-width:1.5px; background:var(--accent-soft); }
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
    <div class="eyebrow">Phase 2a &middot; Preset-span vs random control</div>
    <h1>Voice, or Damage?</h1>
    <p class="deck">A probe showed that perturbing <code>style_ttl</code> along the span of real preset
      differences is far more recoverable from audio than perturbing along a random direction of the
      same size (mean per-component R&sup2; 0.938 vs 0.759). But bench 6 already showed recoverability
      and audibility are decoupled &mdash; its highest-scoring sample was indistinguishable from the
      control by ear. This bench asks the two things R&sup2; cannot.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li>Every clip below is the M1 preset with a small perturbation applied to
          <code>style_ttl</code> &mdash; about a third of the way from M1 toward a real shipped preset
          (target/natural-scale ratio, per preset, in the table below). <code>style_dp</code> and the
          text are held fixed, so durations and words are pinned.</li>
        <li>Each of the 5 groups below has one reference clip (unperturbed M1, openly labelled) and
          two blind clips in randomised order: one is the preset-span perturbation, one is a random
          control of the same size and dimensionality. Which is which is not shown anywhere on this
          page.</li>
        <li>For each blind clip, say what you hear <strong>compared with the reference</strong>: a
          different voice, the same voice with something wrong, both, or no difference. Then add notes
          on what stood out, if anything.</li>
      </ol>
      <p class="why">This decides whether the preset-difference span is a usable axis basis for
        Phase 3. An axis that is recoverable from audio but reads as damage, not as a voice, is not a
        feature &mdash; it would need to be reworked or abandoned regardless of its R&sup2;.</p>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Why these questions</p>
      <p>The question is never &ldquo;which sounds better&rdquo;, and never asks whether a
        previously-described artifact is still present. CLAUDE.md records that a leading question of
        that second kind (&ldquo;is the artifact you flagged before still present?&rdquo;) got &ldquo;no&rdquo;
        on a clip that, asked the open question blind, came back &ldquo;yes, clearly&rdquo; seven minutes
        later &mdash; on bit-identical audio. So Q1 here is a forced choice among four neutral
        descriptions of what changed, not a comparison and not a repeat of any prior description.</p>
    </div>
  </section>

  <section>
    <h2>Perturbation scale</h2>
    <p class="sect-note">Both conditions perturb __RANK__ dimensions (the numerical rank of the 9
      tangent-projected preset-difference directions) at eps=__EPS__, giving a Frobenius perturbation
      norm of __TARGET_NORM__ over the 24 active rows. Below, that norm against each preset's actual
      distance from M1 (<code>||P_i&nbsp;&minus;&nbsp;M1||</code>) &mdash; on average the perturbation
      covers __MEAN_RATIO__ of the distance to a real preset.</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>preset</th><th>&#x2016;P&#8722;M1&#x2016;</th><th>target &divide; natural</th></tr></thead>
        <tbody>
__RATIO_ROWS__
        </tbody>
      </table>
      <figcaption class="tcap">Random control uses an isotropic random subspace of the identical rank
        and the identical eps, so both conditions are perturbed by the same amount &mdash; only the
        direction differs.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The groups</h2>
__GROUPS_HTML__
  </section>

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One free-text box across all 5 groups, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Any pattern across groups -- did one condition read as a voice more often than the other, any recurring failure mode, how confident you feel"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. <code>results/phase2b_presetspan/{preset_span,random_control}/manifest.json</code> supply
    the corpus samples (held-out/&ldquo;test&rdquo;-split only); <code>geometry.json</code> supplies the
    perturbation-scale table. The corpus wrote only 16 kHz embedding copies (no full-rate audio exists
    on disk for either condition), so the 5 reference clips were newly rendered at the corpus's own
    settings (speed=1.05, TOTAL_STEP=8 &mdash; not this fork's speed=1.0 default) and resampled to
    16 kHz to match. Each reference shares its vocoder seed with that group's preset-span sample only;
    the random-control sample keeps its own corpus seed, since preset-span and random-control never
    share a seed with each other in the corpus. All three clips in a group are RMS level-matched to the
    reference.
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
            .replace("__RANK__", str(manifest["rank"]))
            .replace("__EPS__", f"{manifest['eps']:.2f}")
            .replace("__TARGET_NORM__", f"{manifest['perturbation_frobenius_norm_target']:.4f}")
            .replace("__MEAN_RATIO__", f"{manifest['target_over_natural_ratio_mean']:.0%}")
            .replace("__RATIO_ROWS__", ratio_rows)
            .replace("__GROUPS_HTML__", groups_html)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(triples_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


if __name__ == "__main__":
    main()
