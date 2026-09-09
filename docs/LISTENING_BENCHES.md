# Listening Benches

An index of the Artifact listening benches built for this project, in the
order they were produced. Each one exists to get a human verdict on an
audible question a metric alone could not settle — several changed a
recorded project conclusion, which is the reason to keep this index current
rather than letting the verdict live only in a chat transcript.

A published Artifact URL is not a durable record on its own. The generator
that built each bench, the `py/results/` inputs it reads, and the verdict
obtained belong in the repo — indexed here. Generated bench HTML itself
stays out of git: it embeds WAVs (and sometimes PNGs) as base64 and runs
1.8–8.3 MB per file, and the WAVs are gitignored anyway (`results` in
`.gitignore`, matched at any depth).

To rebuild any bench: `cd py && python3 benches/<generator>.py`, after
regenerating its listed inputs — the WAVs and manifests are gitignored, so
the generator only reconstructs the page from assets that must already
exist locally.

## 1. Phase 0 — Linearity gate

- **Artifact:** https://claude.ai/code/artifact/f12d5550-0804-4cc0-8ab4-c7c551aad253
- **Contains:** five renders of one sentence with `style_ttl` linearly
  interpolated between the M1 and F1 presets at weights 0.00 / 0.25 / 0.50 /
  0.75 / 1.00, plus the numeric checks (exact endpoint recovery, unit-row-norm
  preservation, no clipping) that were already settled before anyone listened.
- **Generator:** `py/benches/phase0_linearity_gate_bench.py`
- **Inputs:** `py/results/listening_sets/phase0_linearity/interp_w*.wav`
- **Verdict:** the three intermediate blends read as clean, distinct voices
  between M1 and F1 rather than smear or artifact — the interpolation gate
  passed, and the parametric approach continued as planned.

## 2. Phase 0 — Row locality

- **Artifact:** https://claude.ai/code/artifact/744225a4-dd6a-4b73-a46c-c349c851a2b4
- **Contains:** the per-row delta-share chart for the M1→F1 `style_ttl`
  difference (24 of 50 rows carry ~99.8% of it), the active/frozen row-swap
  pair, sparse top-k row subsets, and single-row swaps (strongest vs.
  deadest row).
- **Generator:** `py/benches/phase0_row_locality_bench.py`
- **Inputs:** `py/results/listening_sets/phase0_row_locality/` (WAVs plus
  `manifest.json`)
- **Verdict:** localization confirmed by ear — the 24-active-row hybrid reads
  as F1, the 26-frozen-row hybrid still reads as M1. Later axis derivation
  can target the active rows and hold the rest fixed.

## 3. Phase 2b — Prosody correction

- **Artifact:** https://claude.ai/code/artifact/d00d367b-18f0-41ba-85c8-a192c58b41e3
- **Contains:** a perturbation ladder (random-direction vs. preset-aligned
  `style_ttl` steps, level-matched) that an utterance-level ΔECAPA /
  spectral-flatness battery had reported as acoustically flat, alongside a
  table comparing that battery against emphasis-aware measures, and log-mel
  diff figures showing where the change actually lands in time.
- **Generator:** `py/benches/phase2b_prosody_correction_bench.py`
- **Inputs:** `py/results/listening_sets/phase2b_prosody_ab/` (WAVs),
  `py/results/phase2b_prosody/` (mel-diff PNGs)
- **Verdict:** the listener heard the emphasis moving word-to-word where the
  metric battery had called the ladder flat. This overturned the recorded
  "most of style space is inaudible" mechanism — the correct account is that
  ECAPA is prosody-invariant by design and could not see what random
  directions do, not that those directions do nothing.

## 4. Phase 2b — Direction collapse / WavLM

- **Artifact:** https://claude.ai/code/artifact/b9d634e6-78cd-4190-940f-830adb910346
- **Contains:** eight near-orthogonal `style_ttl` directions from one base
  voice at matched step size (job 1), and true / ECAPA-predicted /
  WavLM-predicted / fixed-average-voice recovery quadruples for three
  held-out-voice utterances (job 2), plus the pairwise emphasis-correlation
  figures behind both.
- **Generator:** `py/benches/phase2b_direction_collapse_bench.py`
- **Inputs:** `py/results/listening_sets/phase2b_direction_collapse/`,
  `py/results/listening_sets/phase2b_probe_recovery_wavlm/` (WAVs),
  `py/results/phase2b_collapse/` (figures)
- **Verdict:** the eight directions read as one speaker with one clear
  standout (dir2, brighter) rather than eight distinct voices — collapse in
  the strong sense was not heard. The WavLM reconstruction read as closer to
  the true speaker than the fixed average voice, which was the observation
  that prompted the calibrated speaker-identity re-measurement in bench 5.

## 5. Phase 2b — Speaker similarity

- **Artifact:** not yet published
- **Contains:** the calibrated ECAPA-cosine anchors (same-speaker
  different-seed, same-speaker different-sentence, different-speaker across
  45 preset pairs) against which WavLM-predicted, ECAPA-predicted, and
  fixed-average-voice style tensors are measured for 80 held-out samples
  across three identities (F4, F5, M5); per-sample and per-identity
  consistency findings; the withdrawn rms-log-mel-as-identity-measure claim;
  and 27 level-matched clips (true / WavLM prediction / fixed average voice,
  for the best/typical/worst sample of each identity).
- **Generator:** `py/benches/phase2b_speaker_similarity_bench.py`
- **Inputs:** `py/results/listening_sets/phase2b_speaker_similarity/` (WAVs
  plus `manifest.json`), `py/results/phase2b_speaker_similarity/` (the
  `speaker_similarity_report.json` and `gender_confound_report.json` reports
  the generator reads its figures from)
- **Verdict:** pending. Numerically, WavLM's mean cosine to true (0.432) sits
  inside the calibrated different-speaker range (0.225, spanning -0.012 to
  0.569), not the same-speaker one — this is evidence of a probe that beats
  a fixed-average baseline consistently (80/80 samples, and a same-gender
  impostor on 95% of samples), not evidence of voice cloning. Awaiting the
  listening verdict this bench asks for.
