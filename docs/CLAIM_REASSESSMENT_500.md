# Claim reassessment after the 500-condition paired benchmark

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-13
- Verification Status: ANALYZED
- Version Label: validation_v1
- Study ID: `safe-panda-paired-500`
- Code commit used for the benchmark: `5a30eed`
- CSV SHA-256: `adcc3455bb51a78369af884c8445b82331f0eb4423efb9bf9938c305344063f9`

## Outcome

All 500 randomized conditions completed for all eight methods: 4,000 rows,
episode IDs 0--499, no duplicate method/episode pairs, and no incomplete pairs.
The benchmark is environment-sensitive rather than a formal proof or hardware
validation.

| Method | Success | Collision | Raw mean minimum clearance | Relevant mechanism |
|---|---:|---:|---:|---|
| Distance, static obstacle | 208/500 (41.6%) | 292 | 4.86 mm | Distance only |
| Fixed CBF, static, gamma 0.15 | 193/500 (38.6%) | 307 | 4.61 mm | Paper-style baseline |
| Proactive CBF, static, gamma 0.02 | 126/500 (25.2%) | 373 | 0.54 mm | Gamma ablation |
| Predictive CBF, gamma 0.02 | 363/500 (72.6%) | 0 | 69.71 mm | Velocity + tube |
| Predictive CBF + reflex, gamma 0.02 | 363/500 (72.6%) | 0 | 69.74 mm | Adds local reflex |
| Predictive optimal-decay, gamma 0.02 | 417/500 (83.4%) | 1 | 39.19 mm | Adds bounded omega |
| Robust fixed stack, gamma 0.15 | 130/500 (26.0%) | 316 | 10.26 mm | Feedback comparator |
| Robust async feedback | 118/500 (23.6%) | 316 | 10.83 mm | 168 updates applied |

Success means both collision-free and reaching the goal. A collision-free
timeout remains a failure. Safety values always use raw simulated positions;
visual smoothing is not used.

## Four-claim verdict

### 1. Full language-to-optimization: partial implementation, not end-to-end verified

The NARRATE-style TP and Optimization Designer split is implemented as a
closed JSON DSL. Unknown keys, object names, action sequences, dimensions,
units, bounds, and objective/constraint kinds are validated. CasADi compilation
uses constructors from a whitelist and never evaluates LLM-generated source.
Eight dedicated DSL tests pass.

The real DeepInfra path did not complete. The first request correctly failed
closed because the provider's xgrammar backend rejected advanced JSON Schema
features. The wire format was changed to provider-supported `json_object` mode
while retaining the strict local parser. The next single request timed out at
the then-configured eight-second deadline and again stopped before motion. The
default timeout is now 30 seconds, but the request was not automatically
retried. Therefore no real TP plan and OD specification controlled an episode
in this validation run.

Verdict: **PARTIAL / NOT VERIFIED end-to-end**. The safe compiler boundary is
implemented; the paper-level language-to-executed-optimization path is not yet
reproduced.

### 2. Proactive CBF advantage: not supported as a general gamma claim

The isolated static-obstacle gamma comparison contradicts a general proactive
advantage:

- gamma 0.02 minus gamma 0.15 success: **-13.4 percentage points**, paired
  bootstrap 95% CI **[-17.8, -9.0]**, Holm-adjusted exact McNemar
  `p=2.64e-8`;
- raw clearance difference: **-4.07 mm**, 95% CI **[-5.33, -2.82] mm**;
- median lateral avoidance onset is earlier (0.48 versus 0.52 seconds), but it
  does not translate to safer or more successful behavior.

There is strong effect heterogeneity. Across obstacle-speed quartiles, the
success differences are `[-54.4, -19.2, +19.2, +0.8]` percentage points.
Thus the aggregate result hides a regime change: smaller gamma hurts at low
speeds and can help in part of the high-speed range.

Velocity prediction plus the uncertainty tube produces a much stronger stack
result (72.6% success and 0/500 collisions), but that comparison changes the
obstacle model and robustness margin; it cannot be attributed to gamma alone.

The local reflex also does not improve success over predictive MPC alone:
paired difference exactly 0/500, with 39 interventions, all using the backup
path. Its mean clearance increase is only 0.0269 mm (paired bootstrap 95% CI
[0.0023, 0.0693] mm), too small to be practically meaningful here.

Verdict: **NOT SUPPORTED as a general proactive-gamma claim**. Prediction and
uncertainty handling are supported empirically; proactive gamma requires
speed-stratified hypotheses and a better feasibility model.

### 3. Online-feedback improvement: reproduced mechanically, improvement contradicted

The feedback decision was a validated cached HF result: gamma 0.02, measured
latency 2.142 seconds, no fallback. The async queue applied 168/500 updates and
rejected zero payloads. This reproduces non-blocking, versioned, atomic online
parameter updates.

Against the otherwise identical robust fixed stack:

- success changes from 26.0% to 23.6%: **-2.4 percentage points**, paired
  bootstrap 95% CI **[-3.8, -1.2]**, Holm-adjusted McNemar `p=0.00098`;
- collisions are unchanged at 316/500;
- raw mean minimum clearance improves by **0.567 mm**, 95% CI
  **[0.400, 0.743] mm**;
- feedback loses 12 goal completions and gains none;
- among the 168 episodes where an update arrived, success is 102 versus 114
  for fixed gamma. This subset is post-treatment selected and is diagnostic,
  not a separate causal estimate.

Verdict: **MECHANISM REPRODUCED; IMPROVEMENT CLAIM CONTRADICTED in this scene**.
The language update trades a very small clearance increase for fewer completed
tasks and does not change collision count.

### 4. Formal safety under noise and latency: not confirmed

Predictive CBF with the uncertainty tube has strong simulation evidence:
0/500 collisions, 72.6% goal success, and 69.7 mm mean raw clearance. However,
zero observed collisions only implies an approximate 95% binomial upper bound
near 0.6% for this sampled distribution; it is not a mathematical guarantee.

The deterministic tube assumes a three-sigma measurement bound even though the
simulated Gaussian noise is unbounded. The observer assumes constant obstacle
velocity. The reflex models only a spherical end effector, not Panda links,
joints, self-collision, torque limits, or actuator delay. Solver timing also
violates a hard 40 ms interpretation: predictive CBF has 100/500 episodes with
at least one solve over 40 ms and a maximum recorded solve of 2.300 seconds;
the robust feedback stack has 103/500 such episodes.

Optimal decay improves completion over predictive fixed decay by **+10.8
percentage points**, paired 95% CI **[+8.2, +13.6]**, but reduces raw clearance
by **30.52 mm**, 95% CI **[-31.78, -29.27] mm**, and records one collision.
Its bounded omega reaches the configured lower bound 0.1, so feasibility is
bought by relaxing contraction, not by establishing a stronger guarantee.

Verdict: **NOT FORMALLY VERIFIED**. Evidence supports robust simulation
performance for velocity prediction plus a tube, subject to explicit bounded
error/model/deadline assumptions.

## Statistical validation

### Primary findings

| Finding | Test | Estimate | Confidence |
|---|---|---:|---|
| Proactive static gamma effect | paired bootstrap + exact McNemar + Holm | -13.4 pp, CI [-17.8, -9.0], adjusted p=2.64e-8 | SOLID within simulator, heterogeneous by speed |
| Predictive tube vs static fixed CBF | paired bootstrap + exact McNemar + Holm | +34.0 pp, CI [+26.6, +41.4], adjusted p=7.91e-17 | CAUTION: multi-component contrast |
| Reflex vs predictive MPC | paired outcomes | 0.0 pp exactly | SOLID no success benefit in sampled scene |
| Optimal decay vs predictive fixed decay | paired bootstrap | +10.8 pp, CI [+8.2, +13.6] | CAUTION: clearance/safety trade-off |
| Async feedback vs robust fixed | paired bootstrap + exact McNemar + Holm | -2.4 pp, CI [-3.8, -1.2], adjusted p=0.00098 | SOLID within simulator; practically small |

### Fallacy scan

Coverage: **11/11 checked**.

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | CAUTION | Speed quartiles change the sign of proactive and predictive effects; aggregate claims must not be generalized across regimes. |
| Ecological fallacy | NOTE | Unit of analysis and inference is the paired simulated episode; no individual-from-group inference is made. |
| Berkson's paradox | NOTE | All 500 seeded conditions are retained; there is no success-selected sample. |
| Collider bias | CAUTION | Main comparisons do not condition on outcomes. The 168-update subset conditions on surviving until feedback arrival and is labeled diagnostic only. |
| Base-rate neglect | NOTE | Success, collision, and timeout counts are all reported with denominators. |
| Regression to the mean | NOTE | Conditions are randomized, not chosen for extreme prior performance. |
| Survivorship bias | NOTE | No episodes are dropped; all 4,000 method/episode rows are analyzed and timeouts count as failures. |
| Look-elsewhere effect | CAUTION | Eight methods and many metrics are inspected. Holm correction covers McNemar tests, but continuous secondary metrics are exploratory. |
| Garden of forking paths | CAUTION | The 500-run protocol was coded before execution, but it was not externally preregistered and follows earlier pilot/50-run work. |
| Correlation versus causation | CAUTION | Paired randomized simulation supports causal method contrasts inside this simulator only, not on a physical Panda or the paper's unpublished implementation. |
| Reverse causality | NOTE | Method settings precede outcomes. Feedback-arrival subset analysis is not used as a causal claim. |

Overall confidence: **CAUTION**. Paired effects are internally strong, but
regime heterogeneity, multiple component changes, simulator-only geometry, and
deadline violations limit external and formal claims.

## Reproducibility

- Method: one completed environment-sensitive run plus a separate two-condition
  pipeline pilot; no second 500-condition re-run.
- Verdict: **CANNOT VERIFY full-run reproducibility** without repeating all
  4,000 trials. Structural integrity and deterministic seed/config provenance
  are verified; wall-clock solver timing should not be expected to match.
- Unit/regression status after implementation: 119 passed, 2 skipped before the
  final provider compatibility patch; the eight DSL tests pass after that patch.

## Evidence files

- `artifacts/paired_benchmark_500/episodes.csv`: all 4,000 raw rows.
- `artifacts/paired_benchmark_500/benchmark_summary.json`: protocol, HF feedback
  metadata, bootstrap intervals, McNemar tests, and method summaries.
- `artifacts/paired_benchmark_500/paired_success_and_clearance.png`: success and
  raw-clearance comparison.

Generated artifacts remain outside the implementation commit according to the
repository ownership rules.
