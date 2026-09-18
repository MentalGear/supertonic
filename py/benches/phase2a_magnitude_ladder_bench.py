"""Bench generator: Phase 2a magnitude ladder -- where does identity start to
move, and does cleanliness survive the trip?

`phase2b_generate_presetspan.py` / `phase2a_presetspan_bench.py` tested
preset-span vs random-control perturbation of `style_ttl` at eps=0.20
(Frobenius norm 0.98 over the 24 active rows) and found them
indistinguishable by ear -- same voice, audibly different in emphasis,
nothing wrong. But that step is small: 0.98 against real M1-to-preset
distances of 1.85 (M2) to 3.49 (M3). We went about a third of the way to the
nearest real voice and asked whether it sounded like someone else.

On-manifoldness should matter MORE as magnitude grows, not less: a random
tangent direction leaves the manifold of voice-like styles as it scales, a
preset-difference direction does not, by construction. This bench tests that
directly, at magnitudes comparable to and beyond a real preset difference.

UNLIKE most bench generators here, this one both renders the corpus AND
builds the page (the `phase2a_capacity_bench.py` pattern), because the
corpus is small and purpose-built for this one bench: 3 texts, ONE fixed
direction per subspace (not redrawn per sample) stepped through 3
magnitudes, so the ladder is a genuine ladder -- same direction, growing
scale -- not three unrelated draws. 24 renders total.

Directions (fixed across the ladder)
-------------------------------------
Preset-span: the TOP right-singular vector (index 0) of the tangent-
projected 9-preset-difference matrix at M1 -- i.e.
`build_presetspan_basis(...)[0][0]`, the same construction
`phase2b_generate_presetspan.py` condition A uses, reduced to its single
dominant direction. On-manifold by construction.

Random control: a single isotropic tangent-space direction from
`phase2b_generate_subspace.build_basis(..., n_dims=1, seed=RANDOM_DIR_SEED)`
-- the same isotropic-Gaussian-then-QR construction condition B uses,
restricted to one direction, with a seed dedicated to this script (not
`phase2b_generate_presetspan.BASIS_SEED_B`) so it is a fresh, independent
draw, not a re-use.

Both are rows of an orthonormal basis (unit Frobenius norm over the 24x256
tangent space), so `target_norm = eps * sqrt(24)` and the same add-then-
renormalize step `phase2b_generate_subspace.sample_style` uses reproduces
its perturbation model exactly, just with a fixed direction instead of a
random coefficient draw.

Magnitudes: eps in {0.40, 0.70, 1.00} -> target Frobenius norms
{1.96, 3.43, 4.90} -- respectively about 1.06x, 1.85x, 2.65x the M1-M2
natural distance (1.85), so the ladder brackets the nearest real preset
and pushes past it.

Calibration anchor
-------------------
The real M2 preset, rendered DIRECTLY with its own style_ttl AND style_dp
(not M1's dp) -- the nearest genuinely different shipped voice
(||M2-M1||_F = 1.85) -- so the listener has a concrete referent for "a
genuinely different voice" before judging the blind clips. Every other clip
in a group (reference + the 6 perturbed clips) shares M1's style_dp, held
fixed, so only style_ttl differs among them.

Speed
-----
Rendered at speed=1.0, this fork's default (CLAUDE.md), passed explicitly --
NOT the speed=1.05 the presetspan/K-ladder scripts use for comparability
with each other. speed=1.05 divides the duration predictor's own estimate
and carries its own rate of glitch/artifact (docs/GLITCH_MITIGATION.md);
using it here would confound the cleanliness judgement this bench exists to
make. Stated openly on the page.

Seeding
-------
One vocoder seed per GROUP (per text) -- `np.random.seed(seed)` immediately
before EVERY synthesis call in that group (reference, anchor, and all 6
perturbed renders), same seed value throughout. So within a group nothing
differs but the style tensor fed to the engine. (Durations can still differ
slightly across clips since duration is itself a function of style_dp/
style_ttl -- the anchor in particular uses M2's own style_dp.)

Presentation
------------
Per group: reference first (openly labelled), then the calibration anchor
(openly labelled as a different shipped preset -- calibration, not a
leading answer), then the 6 perturbed clips BLIND and order-randomised,
labelled only "Clip A".."Clip F" with no condition or magnitude anywhere in
visible page text (condition/eps are stashed into the DB write payload for
later analysis only, exactly as `phase2a_presetspan_bench.py` already does
for its 2 blind slots -- see that script's DB_SCRIPT docstring). All clips
in a group are RMS level-matched to the group's reference.

Questions: per blind clip, the CORRECTED 5-option forced choice from
`phase2a_presetspan_bench.py` (Q1_OPTIONS, imported verbatim) plus free
text. One page-level question: did any clip sound as different from the
reference as the calibration anchor did (yes/no/not sure) plus free text.

Run from `py/`:
    python3 benches/phase2a_magnitude_ladder_bench.py [--smoke]

Writes WAVs + manifest.json to `results/listening_sets/phase2a_magnitude/`
and the page to `results/benches/phase2a_magnitude.html` (gitignored, embeds
WAVs as base64). Not published per task instructions.
"""

import argparse
import datetime
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

from bench_common import PLAYER_PAUSE_SCRIPT, b64

# helper.py and the phase2b_generate* modules live in py/, one level up from
# this benches/ script -- this generator also renders the corpus itself
# (like phase2a_capacity_bench.py), so it needs py/ on sys.path regardless
# of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import Style, load_text_to_speech, load_voice_style, timer  # noqa: E402
from phase2b_generate import (  # noqa: E402
    ACTIVE_ROWS, LANG, ONNX_DIR, PRESETS, SEED_BASE, TEXTS, TOTAL_STEP, VOICE_STYLE_DIR,
)
from phase2b_generate_presetspan import BASE_PRESET, build_presetspan_basis  # noqa: E402
from phase2b_generate_subspace import build_basis  # noqa: E402
from phase2a_presetspan_bench import Q1_OPTIONS  # noqa: E402 -- corrected 5-option set, reused verbatim

ANCHOR_PRESET = "M2"  # nearest genuinely different shipped voice to M1 (||M2-M1||_F ~= 1.85)

# This fork's default (CLAUDE.md), passed explicitly. See module docstring
# "Speed" section for why this must NOT be 1.05.
SPEED = 1.0

TEXT_IDXS = [0, 3, 6]  # pangram, woodchuck tongue-twister (most-scrutinized artifact case), longer sentence
EPS_LIST = [0.40, 0.70, 1.00]

# Dedicated to this script: a fresh random-direction draw, independent of
# phase2b_generate_presetspan.BASIS_SEED_B.
RANDOM_DIR_SEED = SEED_BASE + 9500
GROUP_SEED_BASE = SEED_BASE + 9600  # per-group shared vocoder seed = GROUP_SEED_BASE + group_index
ORDER_SEED = SEED_BASE + 9700       # blind order (which of A..F gets which condition/eps), per group

LISTEN_DIR = "results/listening_sets/phase2a_magnitude"
OUT_HTML = "results/benches/phase2a_magnitude.html"

BLIND_SLOT_LETTERS = ["A", "B", "C", "D", "E", "F"]

# Persists per-clip verdicts to the artifact's `db` capability at
# `verdicts/<clip_id>` (6 blind clips per group) and the page-level overall
# question at `verdicts/_overall`. `condition`/`eps`/`group` are stashed into
# the saved doc for later analysis, matching phase2a_presetspan_bench.py's
# pattern, but are NEVER rendered into visible page text (labels only ever
# say "Clip A".."Clip F").
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
    if (meta) { body.condition = meta.condition; body.eps = meta.eps; body.group = meta.group; }
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
          if (field === 'updated_at' || field === 'condition' || field === 'eps' || field === 'group') return;
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


def unit_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def level_match(w, ref_rms):
    """Rescale w so its RMS equals ref_rms. Returns (leveled, gain_db)."""
    r = float(np.sqrt((w.astype(np.float64) ** 2).mean()))
    g = ref_rms / max(r, 1e-12)
    return (w * g).astype(np.float32), 20 * np.log10(g)


def perturb_ttl(base_ttl, active_rows, P, direction_24x256, target_norm):
    """Move P by `direction_24x256` (unit Frobenius norm) scaled to
    `target_norm`, renormalize rows, splice into a full (1,50,256)
    style_ttl. Returns (ttl, realized_norm) -- realized_norm measured AFTER
    per-row renormalization, since that step changes it."""
    d = direction_24x256 * target_norm
    rows = unit_rows(P + d)
    ttl = base_ttl.copy()
    ttl[0, active_rows, :] = rows.astype(np.float32)

    norms = np.linalg.norm(ttl[0, active_rows, :], axis=-1)
    assert np.abs(norms - 1.0).max() < 1e-5, f"row norm violated: {np.abs(norms - 1).max()}"
    frozen_rows = [r for r in range(50) if r not in active_rows]
    assert np.array_equal(ttl[0, frozen_rows, :], base_ttl[0, frozen_rows, :]), "frozen row mutated"

    realized_norm = float(np.linalg.norm(rows - P))
    return ttl, realized_norm


def generate_corpus(smoke):
    """Renders the full 3-group (or 1-group under --smoke) corpus, level-
    matches it, writes WAVs, and returns (groups, geometry) where `groups`
    carries in-memory clip records with 'wav_path' set."""
    audio_dir = os.path.join(LISTEN_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    all_presets = {p: load_voice_style([os.path.join(VOICE_STYLE_DIR, f"{p}.json")]) for p in PRESETS}
    base_style = all_presets[BASE_PRESET]
    base_ttl = base_style.ttl.astype(np.float32)
    dp_ref = base_style.dp.copy()

    anchor_style = all_presets[ANCHOR_PRESET]
    anchor_ttl = anchor_style.ttl.astype(np.float32).copy()
    anchor_dp = anchor_style.dp.copy()

    B_a, P, rank, extras = build_presetspan_basis(base_ttl, all_presets)
    direction_ps = B_a[0].reshape(len(ACTIVE_ROWS), 256)
    assert abs(np.linalg.norm(direction_ps) - 1.0) < 1e-8, "preset-span direction not unit Frobenius norm"

    B_r, P_r = build_basis(base_ttl, n_dims=1, seed=RANDOM_DIR_SEED)
    assert np.allclose(P, P_r), "base point mismatch between preset-span and random-direction geometry"
    direction_rand = B_r[0].reshape(len(ACTIVE_ROWS), 256)
    assert abs(np.linalg.norm(direction_rand) - 1.0) < 1e-8, "random direction not unit Frobenius norm"

    cos_between = float(np.dot(direction_ps.reshape(-1), direction_rand.reshape(-1)))
    n_active = len(ACTIVE_ROWS)
    sqrt_n = float(np.sqrt(n_active))
    natural_scale = extras["natural_scale_frobenius"]

    print(f"preset-span vs random direction: cosine = {cos_between:.4f}")
    print(f"natural ||P_i - {BASE_PRESET}||_F: " +
          ", ".join(f"{p}={v:.4f}" for p, v in natural_scale.items()))
    print(f"target norms (eps * sqrt({n_active})={sqrt_n:.6f}): " +
          ", ".join(f"eps={e:.2f}->{e * sqrt_n:.4f}" for e in EPS_LIST))

    tts = load_text_to_speech(ONNX_DIR, use_gpu=False)
    sr = tts.sample_rate

    def render(text, ttl, dp, seed):
        np.random.seed(seed)  # vocoder latent RNG -- shared per group, see module docstring
        wav, dur = tts(text, LANG, Style(ttl.astype(np.float32), dp.copy()), TOTAL_STEP, SPEED)
        return wav[0, : int(sr * dur[0].item())].astype(np.float32), float(dur[0].item())

    text_idxs = TEXT_IDXS[:1] if smoke else TEXT_IDXS
    groups = []
    with timer(f"magnitude ladder ({len(text_idxs)} groups x 8 clips)"):
        for gi, text_idx in enumerate(text_idxs):
            text = TEXTS[text_idx]
            seed = GROUP_SEED_BASE + gi
            clips = []

            wav, dur = render(text, base_ttl, dp_ref, seed)
            clips.append({"kind": "reference", "condition": None, "eps": None,
                          "target_norm": None, "realized_norm": None, "wav": wav, "dur": dur})

            wav, dur = render(text, anchor_ttl, anchor_dp, seed)
            clips.append({"kind": "anchor", "condition": None, "eps": None,
                          "target_norm": None, "realized_norm": None, "wav": wav, "dur": dur})

            for cond_name, direction in (("preset_span", direction_ps), ("random_control", direction_rand)):
                for eps in EPS_LIST:
                    target_norm = eps * sqrt_n
                    ttl, realized_norm = perturb_ttl(base_ttl, ACTIVE_ROWS, P, direction, target_norm)
                    wav, dur = render(text, ttl, dp_ref, seed)
                    clips.append({"kind": "blind", "condition": cond_name, "eps": eps,
                                  "target_norm": target_norm, "realized_norm": realized_norm,
                                  "wav": wav, "dur": dur})

            groups.append({"group_id": f"group{gi + 1:02d}", "text_idx": text_idx, "text": text,
                            "seed": seed, "clips": clips})
            print(f"  group{gi + 1:02d} (text_idx={text_idx}) rendered, seed={seed}", flush=True)

    geometry = {
        "base_preset": BASE_PRESET, "anchor_preset": ANCHOR_PRESET,
        "active_rows": ACTIVE_ROWS, "rank_of_full_presetspan_basis": rank,
        "eps_list": EPS_LIST, "target_norms": {f"eps{e:.2f}": e * sqrt_n for e in EPS_LIST},
        "natural_scale_frobenius": natural_scale,
        "anchor_natural_distance_frobenius": natural_scale[ANCHOR_PRESET],
        "cosine_preset_span_vs_random_direction": cos_between,
        "direction_preset_span": "top right-singular vector (index 0) of the tangent-projected "
                                  "9-preset-difference matrix at M1 (build_presetspan_basis, same "
                                  "construction as phase2b_generate_presetspan.py condition A)",
        "direction_random_control": f"single isotropic tangent-space direction (build_basis, n_dims=1, "
                                     f"seed={RANDOM_DIR_SEED}), same construction as "
                                     f"phase2b_generate_presetspan.py condition B, restricted to one direction",
    }

    if smoke:
        return groups, geometry

    for group in groups:
        ref_wav = group["clips"][0]["wav"]
        ref_rms = float(np.sqrt((ref_wav.astype(np.float64) ** 2).mean()))
        for ci, clip in enumerate(group["clips"]):
            leveled, gain_db = level_match(clip["wav"], ref_rms)
            fname = f"{group['group_id']}_c{ci:02d}_{clip['kind']}.wav"
            path = os.path.join(audio_dir, fname)
            sf.write(path, leveled, sr, subtype="PCM_16")
            clip["wav_path"] = path
            clip["file"] = fname
            clip["levelmatch_gain_db"] = gain_db
            clip["src_rms"] = float(np.sqrt((clip["wav"].astype(np.float64) ** 2).mean()))
            clip["ref_rms"] = ref_rms
            clip["peak_after_levelmatch"] = float(np.abs(leveled).max())
            clip["finite"] = bool(np.isfinite(leveled).all())
            del clip["wav"]  # keep the manifest JSON-serializable and small

    return groups, geometry


def assign_blind(groups):
    rng = random.Random(ORDER_SEED)
    for g in groups:
        blind = [c for c in g["clips"] if c["kind"] == "blind"]
        order = list(range(len(blind)))
        rng.shuffle(order)
        for slot_i, src_i in enumerate(order):
            blind[src_i]["slot_letter"] = BLIND_SLOT_LETTERS[slot_i]
        blind.sort(key=lambda c: c["slot_letter"])
        g["blind_ordered"] = blind


def write_manifest(groups, geometry, meta_extra):
    entries = []
    for g in groups:
        entries.append({
            "group_id": g["group_id"], "text_idx": g["text_idx"], "text": g["text"], "seed": g["seed"],
            "clips": [
                {k: v for k, v in c.items() if k != "wav"}
                for c in g["clips"]
            ],
        })
    meta = {
        "experiment": "phase2a_magnitude_ladder_bench",
        "date": datetime.date.today().isoformat(),
        "purpose": "blind human judgment of whether preset-span and random-direction style_ttl "
                   "perturbations, at magnitudes comparable to and beyond a real preset difference, "
                   "are audible as identity change, and whether cleanliness survives that scale -- "
                   "with a real shipped preset (M2) as a calibration anchor for what a genuinely "
                   "different voice sounds like",
        "geometry": geometry,
        "speed": SPEED,
        "speed_note": "This fork's speed=1.0 default, passed explicitly -- NOT the speed=1.05 the "
                       "presetspan/K-ladder scripts use. See module docstring.",
        "total_step": TOTAL_STEP, "lang": LANG,
        "seed_rule": "np.random.seed(GROUP_SEED_BASE + group_index) immediately before EVERY "
                     "synthesis call within that group (reference, anchor, all 6 perturbed) -- "
                     "one shared vocoder seed per group.",
        "group_seed_base": GROUP_SEED_BASE, "order_seed": ORDER_SEED,
        "levelmatch_note": "every clip in a group RMS level-matched to that group's reference",
        "q1_options": [v for _, v in Q1_OPTIONS],
        "groups": entries,
    }
    meta.update(meta_extra)
    os.makedirs(LISTEN_DIR, exist_ok=True)
    with open(os.path.join(LISTEN_DIR, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote manifest -> {LISTEN_DIR}/manifest.json")
    return meta


def build_html(groups, geometry, manifest):
    def q1_choices(cid):
        return "\n".join(
            f'              <label><input type="radio" name="{cid}_q1" value="{val}" disabled> {label}</label>'
            for val, label in Q1_OPTIONS)

    def blind_card(c):
        cid = c["id"]
        return f"""        <article class="clip" data-clip="{cid}">
          <div class="clip-head"><span class="clip-label">Clip {c['slot_letter']}</span></div>
          <audio controls preload="metadata" src="data:audio/wav;base64,{b64(c['wav_path'])}"></audio>
          <div class="verdict" data-clip="{cid}">
            <div class="q-block">
              <p class="q-label">Q1 &middot; Compared with the reference, what do you hear? <span class="tick" data-tick data-clip="{cid}" data-field="q1"></span></p>
              <div class="choices" data-field="q1">
{q1_choices(cid)}
              </div>
            </div>
            <div class="q-block">
              <p class="q-label">Q2 &middot; Notes <span class="tick" data-tick data-clip="{cid}" data-field="q2"></span></p>
              <textarea data-field="q2" data-clip="{cid}" disabled rows="2"
                placeholder="a word or moment that stood out, e.g. &quot;flattens on 'seashells'&quot;"></textarea>
            </div>
          </div>
        </article>"""

    def group_section(g, gi):
        blind_html = "\n".join(blind_card(c) for c in g["blind_ordered"])
        return f"""  <section class="group">
    <h2>Group {gi} &middot; &ldquo;{g['text']}&rdquo;</h2>
    <p class="sect-note">Reference, then the calibration anchor, then 6 blind clips in random order.
      Judge each blind clip against the reference above it.</p>
    <article class="clip anchor">
      <div class="clip-head"><span class="clip-label">Reference (unperturbed M1)</span></div>
      <audio controls preload="metadata" src="data:audio/wav;base64,{b64(g['clips'][0]['wav_path'])}"></audio>
    </article>
    <article class="clip anchor cal">
      <div class="clip-head"><span class="clip-label">Calibration anchor &middot; preset M2, rendered directly</span></div>
      <p class="cal-note">A genuinely different shipped voice (not a perturbation of M1) &mdash; so you have a
        concrete reference for &ldquo;a different voice&rdquo; before judging the blind clips below.</p>
      <audio controls preload="metadata" src="data:audio/wav;base64,{b64(g['clips'][1]['wav_path'])}"></audio>
    </article>
    <div class="clips">
{blind_html}
    </div>
  </section>"""

    groups_html = "\n".join(group_section(g, gi) for gi, g in enumerate(groups, start=1))

    clips_for_js = []
    for g in groups:
        for c in g["blind_ordered"]:
            clips_for_js.append({"id": c["id"], "condition": c["condition"], "eps": c["eps"], "group": g["group_id"]})

    target_norms = geometry["target_norms"]
    anchor_dist = geometry["anchor_natural_distance_frobenius"]
    scale_rows = "\n".join(
        f'          <tr><td>eps={e:.2f}</td><td>{target_norms[f"eps{e:.2f}"]:.4f}</td>'
        f'<td>{target_norms[f"eps{e:.2f}"] / anchor_dist:.2f}&times;</td></tr>'
        for e in EPS_LIST
    )

    HTML = """<title>Magnitude Ladder</title>
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
  .clip.cal { border-color:var(--flag); }
  .clip-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .clip-label { font-family:var(--sans); font-weight:600; font-size:14px; }
  .cal-note { font-size:13px; color:var(--ink-2); margin:0; }
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
    <div class="eyebrow">Phase 2a &middot; Magnitude ladder</div>
    <h1>Where Does Identity Start to Move?</h1>
    <p class="deck">At eps=0.20 (Frobenius norm 0.98), preset-span and random-direction
      <code>style_ttl</code> perturbations were indistinguishable by ear &mdash; both read as the
      same voice, audibly different, nothing wrong. But that step covers less than a third of the
      distance to the nearest real preset. This bench pushes the SAME direction in each subspace out
      to magnitudes comparable to and beyond a real preset difference, to find where identity starts
      to move, and whether an on-manifold (preset-span) direction stays clean where an off-manifold
      (random) direction degrades.</p>
    <p class="progress" id="progress-line">Checking saved responses&hellip;</p>
  </header>

  <section>
    <div class="task">
      <p class="tag">What we need from you</p>
      <ol>
        <li>Each of the 3 groups below has one reference clip (unperturbed M1, openly labelled), one
          calibration anchor (the real preset M2, openly labelled as a different shipped voice, rendered
          directly with its own timing), and 6 blind clips in randomised order.</li>
        <li>The 6 blind clips are 2 directions (one on-manifold, spanning real preset differences; one a
          random tangent direction) each stepped through 3 growing magnitudes &mdash; which direction and
          which magnitude is not shown anywhere on this page.</li>
        <li>For each blind clip, say what you hear <strong>compared with the reference</strong>, using the
          5-option choice below, then add notes on anything that stood out.</li>
        <li>After all 3 groups, one overall question: did any blind clip sound as different from its
          reference as the calibration anchor (M2) did?</li>
      </ol>
      <p class="why">This decides whether the preset-difference span stays a usable, clean axis as its
        magnitude grows, or whether it degrades the same way a random direction does once you push past
        the small step already tested. That is the evidence R&sup2; alone cannot give.</p>
    </div>
  </section>

  <section>
    <div class="correction">
      <p class="was">Why these questions</p>
      <p>Q1 is a forced choice among five neutral descriptions of what changed &mdash; including &ldquo;the
        same voice, audibly different, nothing wrong&rdquo;, the option a prior bench's 4-way set omitted,
        which forced every verdict there into the wrong box until free text corrected it (see
        <code>docs/LISTENING_BENCHES.md</code>). The question is never &ldquo;which sounds better&rdquo;, and
        never asks whether a previously-flagged artifact is still present &mdash; that kind of leading
        question has suppressed real detections before (CLAUDE.md).</p>
    </div>
  </section>

  <section>
    <h2>Perturbation scale</h2>
    <p class="sect-note">Both directions are stepped through the same three magnitudes, eps in
      {0.40, 0.70, 1.00}, giving Frobenius perturbation norms over the 24 active rows shown below,
      against the M1&ndash;M2 natural distance (__ANCHOR_DIST__) &mdash; the calibration anchor's own
      scale. Speed is __SPEED__ for every clip in this bench (this fork's default, not the 1.05 the
      presetspan/K-ladder scripts use, to avoid confounding cleanliness with that speed's own artifact
      rate).</p>
    <figure class="tablebox">
      <table>
        <thead><tr><th>magnitude</th><th>target norm</th><th>&divide; M1&ndash;M2 distance</th></tr></thead>
        <tbody>
__SCALE_ROWS__
        </tbody>
      </table>
      <figcaption class="tcap">Realized norms (measured after per-row renormalization) are in
        <code>manifest.json</code> alongside the target norms above; they track the target closely but
        are not bit-identical to it, since renormalization is a nonlinear step.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The groups</h2>
__GROUPS_HTML__
  </section>

  <section>
    <h2>Overall</h2>
    <p class="sect-note">One question across all 3 groups, not per-clip.</p>
    <div class="verdict" data-clip="_overall">
      <div class="q-block">
        <p class="q-label">Did any blind clip sound as different from its reference as the calibration
          anchor (M2) did? <span class="tick" data-tick data-clip="_overall" data-field="as_different"></span></p>
        <div class="choices" data-field="as_different">
          <label><input type="radio" name="_overall_as_different" value="yes" disabled> Yes, at least one</label>
          <label><input type="radio" name="_overall_as_different" value="no" disabled> No, none came close</label>
          <label><input type="radio" name="_overall_as_different" value="not_sure" disabled> Not sure</label>
        </div>
      </div>
      <div class="q-block">
        <p class="q-label">Overall notes <span class="tick" data-tick data-clip="_overall" data-field="notes"></span></p>
        <textarea data-field="notes" data-clip="_overall" disabled rows="3"
          placeholder="Any pattern across groups -- did one kind of clip stay cleaner as it grew stronger, any recurring failure mode, how confident you feel"></textarea>
      </div>
    </div>
  </section>

  <footer>
    Sources. <code>results/listening_sets/phase2a_magnitude/manifest.json</code> has the full per-clip
    record (condition, eps, target and realized Frobenius norms, seeds, level-match gains) &mdash; none of
    it shown on this page for the blind clips. Rendered at <code>speed=__SPEED__</code>,
    <code>TOTAL_STEP=__TOTAL_STEP__</code>. All clips in a group share one vocoder seed and are RMS
    level-matched to that group's reference.
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
            .replace("__ANCHOR_DIST__", f"{anchor_dist:.4f}")
            .replace("__SPEED__", f"{SPEED:.2f}")
            .replace("__TOTAL_STEP__", str(TOTAL_STEP))
            .replace("__SCALE_ROWS__", scale_rows)
            .replace("__GROUPS_HTML__", groups_html)
            .replace("__PLAYER_PAUSE_SCRIPT__", PLAYER_PAUSE_SCRIPT)
            .replace("__CLIPS_JSON__", json.dumps(clips_for_js))
            .replace("__DB_SCRIPT__", DB_SCRIPT))

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT_HTML} {os.path.getsize(OUT_HTML)} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 text, no audio kept, no HTML written")
    args = ap.parse_args()

    groups, geometry = generate_corpus(args.smoke)
    if args.smoke:
        print("smoke: geometry + seeding checks passed, no audio/HTML written")
        return

    assign_blind(groups)
    for g in groups:
        for slot_i, c in enumerate(g["blind_ordered"]):
            c["id"] = f"{g['group_id']}_{c['slot_letter']}"

    manifest = write_manifest(groups, geometry, meta_extra={
        "random_dir_seed": RANDOM_DIR_SEED,
        "text_idxs": [g["text_idx"] for g in groups],
    })

    for g in groups:
        print(f"  {g['group_id']} text_idx={g['text_idx']} seed={g['seed']}:", end="")
        for c in g["blind_ordered"]:
            print(f" Clip{c['slot_letter']}->{c['condition']}(eps{c['eps']:.2f}"
                  f" realized{c['realized_norm']:.3f} gain{c['levelmatch_gain_db']:+.2f}dB)", end="")
        print()

    build_html(groups, geometry, manifest)


if __name__ == "__main__":
    main()
