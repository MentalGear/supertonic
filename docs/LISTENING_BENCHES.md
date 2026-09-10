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
- **Contains:** 8 of the 24 mutually near-orthogonal `style_ttl` directions
  the measurement used, rendered from one base voice at matched step size —
  the subset that stays A/B-able for a listener (job 1) — and true /
  ECAPA-predicted / WavLM-predicted / fixed-average-voice recovery
  quadruples for three held-out-voice utterances (job 2), plus the pairwise
  emphasis-correlation figures behind both. The direction-identifiability
  measurement itself ran on all 24 (chance rate 1/24 = 4.2%; cross-latent
  1-NN on the log-mel diff got it right 70.8% of the time at eps 0.20 and
  66.7% at eps 0.80).
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

- **Artifact:** https://claude.ai/code/artifact/1e0808cf-3de0-4968-8177-b48797202423
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
  impostor on 95% of samples), not evidence of voice cloning.
  By ear, on the three-sample set that preceded this one: the WavLM
  reconstruction reads as mostly the true voice, and the fixed average never
  does — the listener's report that prompted this whole measurement. So the
  qualitative and calibrated readings agree on direction and disagree on
  degree: a real, consistent partial recovery that does not reach identity.
  The robustness check across best/typical/worst for all three identities was
  then run and **it holds up**: the impression survives across the range. The
  one outlier is F4's worst case, where the listener heard the WavLM
  prediction as a blend of the true voice and the average one. That is what
  ridge regression does — predictions shrink toward the training mean, most
  strongly where the signal is weakest — so the weak end sounding like a
  blend is the regulariser being audible, and is independent confirmation
  that the probe degrades gracefully rather than erratically.
  Scope worth restating here, because the word "true" invites the wrong
  reading: every style in this bench is engine-generated. "True style" means
  the tensor we constructed and rendered from, not a recorded human, and the
  held-out identities are shipped presets withheld from probe training, not
  held-out people.

## 6. Phase 2a — Capacity

- **Artifact:** https://claude.ai/code/artifact/624b138d-b490-40ea-9a07-90f5776517ed
- **Contains:** what Phase 2a's R&sup2; ladder (0.912 / 0.608 / 0.232 at
  K=4/16/64, eps=0.20) sounds like: for each K, the best/typical/worst
  test-split sample by per-sample R&sup2;, each rendered as a level-matched,
  seed-matched triple — true style, probe prediction (base + predicted c @
  B[:K], `unit_rows`), and the unperturbed base preset as the "recovered
  nothing" control. Also the achieved-R&sup2;/oracle-ceiling/shuffled-control/
  loudness-control table per K, and the amplitude-matched control result
  (K=4/eps0.05 R&sup2;=0.297, K=16/eps0.10 R&sup2;=0.199, K=64/eps0.20
  R&sup2;=0.232) showing the K-decline is largely a per-direction-amplitude
  effect rather than a hard dimensional ceiling.
- **Generator:** `py/benches/phase2a_capacity_bench.py` — unlike most other
  generators here, this one also re-fits the ridge probe and renders the
  audio itself (reusing `phase2b_subspace_probe.build_xy` /
  `phase2b_probe.fit_ridge`, byte-identical to the reported R&sup2;), rather
  than only building HTML from an already-rendered listening set.
- **Inputs:** `py/results/phase2b_subspace/` (manifest, subspace basis,
  WavLM features) and `py/results/phase2a/subspace_probe*.json` (the R&sup2;
  reports), all produced by `phase2b_generate_subspace.py` +
  `phase2b_subspace_embed.py` + `phase2b_subspace_probe.py`. Outputs land in
  `py/results/listening_sets/phase2a_capacity/` (27 WAVs + manifest.json).
- **Verdict:** pending — awaiting a listener's judgment on where along the
  K=4/16/64 ladder the prediction stops being audibly distinguishable from
  the no-perturbation control. Note: an active-row style cosine was computed
  for every clip but deliberately left out of the page — it came back
  nearly flat (0.98–0.99) across the whole R&sup2; range, which is exactly
  what bench 5 already found calibrating that same metric (it does not
  separate same-/different-speaker pairs), so surfacing it here would have
  presented an uncalibrated number next to the calibrated one under test.


## 7. Phase 2a — Baseline artifact rate

- **Artifact:** not yet published.
- **Contains:** 20 clips drawn blind from stock, completely unperturbed
  Supertonic (all 10 shipped presets, a mix of the 8 corpus texts and 6
  seeds per preset/text cell), asking only whether the listener hears an
  audible artifact and where. 4 of the 20 are exact repeats of the bench-6
  control clips a listener already judged (woodchuck M1 seed 20261069
  "hiccupy" / seed 20261295 "clean"; seashells M1 seed 20261449 "hiccupy" /
  seed 20262075 clean), mixed in unlabelled as internal consistency checks.
  Labels are blind ("Clip 01"-"Clip 20"); a reveal is available per clip
  only after answering. Includes a Wilson-interval table stating plainly
  that 20 clips gives only a coarse rate. Explicitly states why an automated
  detector was not used instead (see below).
- **Generator:** `py/benches/phase2a_baseline_bench.py` -- re-renders the 16
  non-check clips from the exact (preset, text, seed) triples recorded in
  `results/phase2a/baseline_artifact_rate.json` (that script's own audio is
  not retained -- its docstring says so explicitly), using the identical
  code path so the audio matches what was already characterized
  numerically; the 4 check clips are copied verbatim from an existing
  listening set rather than re-rendered.
- **Inputs:** `py/results/phase2a/baseline_artifact_rate.json` (480-render
  characterization, no audio),
  `py/results/listening_sets/phase2a_seed_variance/` (source of the 4
  check-clip WAVs). Outputs land in
  `py/results/listening_sets/phase2a_baseline/` (20 WAVs + manifest.json).
- **Why no detector-based number appears:** a frame-to-frame log-mel
  spectral-flux detector, thresholded on the 480-render stock pool, flags
  98.5% of all stock renders regardless of preset or text -- it fires on
  ordinary consonant transients. Worse, on the two bench-6 pairs with an
  explicit listener verdict its ordering is reversed against the listener
  (see `phase2a_seed_variance.py`'s `detector_validation_against_listener`),
  so it cannot be used and no number from it is shown.
- **Verdict:** pending -- this bench has been generated and run but not yet
  listened to.
