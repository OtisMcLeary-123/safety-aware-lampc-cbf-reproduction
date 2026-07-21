# Confirmatory CS1 Feedback Effect — Design + Preregistration (FINAL)

- Date preregistered: 2026-07-19 (before any confirmatory instance is generated)
- Loop: power-analysis (paired Monte-Carlo; vendored two-proportion tool
  used only as an unpaired cross-check because it does not model paired
  designs — scope limitation declared)

## Hypothesis (one, primary)
On NEW CS1 head-on instances drawn from the plan-v1 ranges, the
**checkpointed NIM feedback protocol** (gamma 0.15 -> 0.05 at the frozen
per-instance feedback time, zero new API requests) increases
collision_free_goal_success versus soft-slack fixed gamma 0.15, both on
the soft-slack (L1, weight 1000) velocity-tube controller base.
Scope: this tests the frozen deterministic feedback protocol, not fresh
LLM variance; claim applies to the declared CS1 geometry ranges only.

## Design
Paired within-instance: every instance runs both arms. Deterministic
simulations; both arms on identical geometry/seeds = perfect concurrent
control; no assignment randomization required (order irrelevant,
processes independent).

## Primary outcome & the one planned test
collision_free_goal_success per episode; exact two-sided McNemar on the
120 pairs at alpha 0.05. Single family, single hypothesis - no
multiplicity correction. All other endpoints (clearance, solver
failures, timeouts) are EXPLORATORY.

## Minimal effect of interest & power
Baseline 0.08 (slack fixed CS1 4/50, prior data). Minimal meaningful
lift +0.10 absolute, discordant model p10=0.02, p01=0.12. Paired exact
McNemar Monte-Carlo (40k reps, seed 20260719):
n=115 -> power 0.806; **n=120 -> power 0.828** (adopted, margin for MC
error). Unpaired vendored cross-check would need ~200/group - the
paired design is the efficient choice. Observed +0.12 (p=0.070) from
the generating data is suggestive only and was NOT used to size power.

## Instances (new, hypothesis-independent)
120 NEW CS1 instances: identical plan-v1 ranges and preflight gates, new
non-overlapping seed block (documented plan-v2 instance file with its
own sha256, generated only after this preregistration is committed).
The 50 CS1 instances that generated the hypothesis are excluded.

## Determinism & measurement
Automated metrics, identical code for both arms. Known limitation: the
frozen IPOPT profile's 0.035 s CPU cap gives outcome-level (not
step-exact) reproducibility; this affects both arms symmetrically and
is noise, not bias. Profile kept identical to the generating data for
comparability. Gamma decision replayed from the existing checkpoint
(0 provider requests).

## Stopping rule
Fixed n=120 pairs (240 episodes), run to completion, one analysis at
the end. No interim looks, no early stopping, no instance replacement;
crashed episodes are recorded and reported.

## Interpretation rules (fixed in advance)
- p < 0.05 with feedback-favoring discordance -> confirmatory support
  for the CS1 feedback effect on the slack base.
- p >= 0.05 -> the +0.12 remains unconfirmed; report the interval, no
  rescue analyses.
- Effect size + CI reported regardless of significance.

## Amendment log
- 2026-07-20 (before any episode ran): the first instance draw
  (seed_base 20260720) was discarded — 41/120 geometries collided with
  the hypothesis-generating set because the seed block overlapped
  20260716-20260765 and rejection resampling is geometry-seed-keyed.
  Redrawn with seed_base 20460716 (non-overlapping with all original
  geometry and measurement blocks); overlap with the generating set
  verified 0. Frozen instances sha256: aa07f70d98857984339c187e2da3f9738118ea3fd720300b43f1208ac205af7a.

## Outcome (2026-07-20, recorded after the single preregistered analysis)
- Fixed 17/120 (Wilson [0.090, 0.215]); feedback 30/120 ([0.183, 0.337]).
- Discordant pairs: 14 feedback-only vs 1 fixed-only.
- Primary test: exact two-sided McNemar **p = 0.00098** -> per the locked
  interpretation rule, confirmatory support for the CS1 feedback effect
  on the slack base.
- Effect size: +0.108 absolute, paired bootstrap 95% CI [0.050, 0.167].
- Zero collisions in all 240 episodes; one crashed episode (CS1-E13,
  feedback arm) recorded per protocol and counted as non-success
  (conservative direction). Zero new API requests, zero fallbacks.
- Note: the realized baseline rate (0.142) exceeded the assumed 0.08;
  the observed effect (+0.108) matched the minimal effect (+0.10) the
  design was powered for.
