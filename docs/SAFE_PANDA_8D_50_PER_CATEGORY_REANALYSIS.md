# 8-D 50-Scenario Benchmark — Per-Category Reanalysis (no reruns)

- Date: 2026-07-19
- Source data: the frozen, committed
  `artifacts/safe_panda_8d_double_integrator_50_benchmark/episodes.csv`
  (100 rows, 50 paired episodes). No simulation was rerun; this document
  only brings the frozen result up to the current reporting standard
  (per-category Wilson intervals, exact McNemar per category, Holm across
  the four categories — the no-pooling rule).
- Caveats carried over from the original setup: point obstacle
  (`obstacle_radius_m = 0.0`, combined radius 0.035 m), hand-authored
  scenarios without formal preflight encounter gates, discrete feedback
  gamma vocabulary. Numbers are NOT directly comparable to the core-150
  benchmark (different geometry scale).

## Per-category results (fixed gamma 0.15 vs async LLM feedback)

| Category | n | Fixed success (Wilson 95%) | Feedback success | Collisions (both arms) | Discordant | McNemar p | Holm |
|---|---|---|---|---|---|---|---|
| Head-on Collision | 15 | 0/15 [0.000, 0.204] | 0/15 | 15 | 0 / 0 | 1.0 | 1.0 |
| Orthogonal Crossing | 15 | 6/15 [0.198, 0.643] | 6/15 | 9 | 0 / 0 | 1.0 | 1.0 |
| Boundary & Kinematic Limits | 10 | 5/10 [0.237, 0.763] | 5/10 | 5 | 0 / 0 | 1.0 | 1.0 |
| High Unbounded Noise | 10 | 2/10 [0.057, 0.510] | 2/10 | 8 | 0 / 0 | 1.0 | 1.0 |

## Findings

1. **Feedback inertness is total at every category level**: zero
   discordant pairs in all four categories — not a single episode
   changed outcome anywhere. The aggregate p=1.0 was not an averaging
   artifact.
2. **The head-on wall was already visible here**: 0/15 with 15/15
   collisions exactly prefigures the core-150 CS1 result (0/50 static
   baseline). Hidden inside the old aggregate 13/50, the family
   structure of the failure was present two benchmarks ago.
3. **Crossing was already the most tractable geometry** (6/15 = 40%),
   matching CS2 being the family with the most headroom in core-150.
4. Cross-benchmark consistency: two independently designed scenario
   suites (hand-authored point-obstacle vs LHC spherical-obstacle) agree
   on the ordering head-on < noise/grazing < crossing and on complete
   feedback inertness under static prediction + fail-closed fallback —
   the third independent observation of the signature that the core-150
   NIM + soft-slack arm causally explained.

Registry/journal cross-references: mechanism closure in
`docs/SAFE_PANDA_CORE_SCENARIOS_150_RESULT.md`; original frozen aggregate
in the committed benchmark summary (13/50 both arms, failure steps 63 vs
240).
