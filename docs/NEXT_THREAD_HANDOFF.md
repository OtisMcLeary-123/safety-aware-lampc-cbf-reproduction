# Next Thread Handoff

## Current checkpoint

The repository is now centered on the Safe Panda 8-state
`double_integrator` profile. The final replay uses one fixed condition and one
contextual-feedback condition for each of 50 deterministic scenarios.

- Setup: `configs/safe_panda_8d_double_integrator_50_setup.json`
- Runner: `scripts/run_safe_panda_8d_benchmark.py`
- Aggregate result: `artifacts/safe_panda_8d_double_integrator_50_benchmark/`
- Master plan: `docs/MASTER_PLAN_SETUP.md`
- Provider checkpoint: local only; do not commit raw provider records

## Final result

- Fixed `gamma=0.15`: 13/50 goals, 37/50 collisions, 63 solver-failure steps.
- Contextual feedback: 13/50 goals, 37/50 collisions, 240 solver-failure steps.
- Paired success difference: `0.0`; exact McNemar `p=1.0`.
- Maximum model/action transition error: `0.000682 m`.
- Full tests: `278 passed, 2 skipped`.

## Next technical gate

Do not spend more provider requests before addressing solver feasibility. Compare
soft horizon CBF slack, exact-penalty tuning, hard one-step D-GCBF, and braking
fallback as separate profiles. Preserve the current baseline unchanged while
testing each remedy. Revisit goal speed/hold semantics and obstacle geometry
only in a new versioned setup manifest.

## New opt-in 3-D avoidance demo

An isolated visualization/demo profile was added for the next work session.
It does not change the 50-scenario paired benchmark defaults (`direct_target`,
legacy scenario route, isotropic CBF).

- CLI: `scripts/run_3d_avoidance_demo.py`
- Core route/config: `src/lampc_cbf/smooth_dynamic_demo.py`
- Axis-weighted MPC objective: `src/lampc_cbf/controller.py`
- Documentation: `docs/3D_AVOIDANCE_DEMO.md`
- Unit coverage: `tests/test_controller.py` and `tests/test_smooth_dynamic_demo.py`

The profile uses `reference_mode="behind_spline"`,
`reference_route_profile="3d_waypoints"`, two intermediate waypoint offsets
with nonzero `x` and `z`, a nonzero-height/vertical-moving obstacle, optional
tangential safety-reflex subgoals, and `position_q_weights`. The CBF remains an
isotropic Euclidean barrier; axis weights only shape tracking of the supplied
3-D reference and must not be used to weaken the collision barrier.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_3d_avoidance_demo.py \
  --reference-mode behind_spline \
  --goal-offset 0.00 0.30 0.00 \
  --obstacle-offset 0.00 0.15 0.06 \
  --obstacle-velocity 0.05 0.00 -0.015 \
  --route-offset 0.14 0.08 0.10 \
  --route-offset 0.14 0.23 0.10 \
  --position-q-weights 1.0 1.4 1.2 \
  --tangential-subgoal \
  --save-animation \
  --output-dir artifacts/3d_avoidance_demo
```

Expected outputs are `metrics.json`, `raw_smoothed_and_safety.png`,
`trajectory_3d_comparison.png`, and `robot_motion.gif`. The Figure-5-style
comparison contains top (`x/y`), diagonal 3-D (`x/y/z`), and clearance plots.

Latest validation of the profile reached `outcome="goal"` with
`collision=false`; raw trajectory ranges were approximately `x=-0.0608..0.1300`
m and `z=0.1780..0.3131` m. The run still reported solver fallback activity
(17 solver-failure steps), so the next session should decide whether to tune
the profile or document those fallbacks explicitly.

Targeted tests after this change: `47 passed, 1 skipped`. A full-suite rerun
has not been performed after the 3-D extension.

## Completed 50-case 3-D provider benchmark

The user explicitly authorized provider spending after the earlier feasibility
warning. A separate versioned workflow was created and completed; it must not
be mixed with the original Safe Panda 8-D result.

- Setup: `configs/safe_panda_3d_avoidance_50_setup.json`
- Provider collector: `scripts/collect_safe_panda_3d_feedback.py`
- Paired runner: `scripts/run_safe_panda_3d_benchmark.py`
- Provider checkpoint: `artifacts/safe_panda_3d_avoidance_50_provider/feedback_decisions_checkpoint.json`
- Results: `artifacts/safe_panda_3d_avoidance_50_benchmark/`
- Integration tests: `tests/test_safe_panda_3d_workflow.py`

The provider collector uses a declared `nominal_3d_spline_proxy` hazard
context because the exact intervention-time controller state is unavailable
before replay. It called NVIDIA NIM once per scenario. One request failed at
episode 35; the collector retained 34 valid records and resumed successfully.
The final checkpoint contains 50 uncached, non-fallback decisions:

- Gamma `0.03`: 42 decisions
- Gamma `0.07`: 8 decisions
- Mean / median / maximum latency: approximately `0.365 / 0.339 / 0.711 s`

Final paired result:

| Method | Success | Collision | Solver failures/rejections | Mean minimum clearance |
|---|---:|---:|---:|---:|
| Fixed `gamma=0.15` | 19/50 | 31/50 | 0 / 0 | 0.00354 m |
| Async provider feedback | 23/50 | 27/50 | 0 / 0 | 0.00697 m |

- Paired success difference: `+0.08`
- Exact McNemar p-value: `0.125` (not significant at `0.05`)
- Feedback-only successes: episodes `1, 7, 17, 19`
- Fixed-only successes: none
- Paired mean clearance improvement: `+0.00343 m`
- All 50 feedback gamma updates were applied
- No deadline misses or emergency fallbacks in either method

The completed artifacts are resumable through the provider JSON prefix and
the benchmark `run_checkpoint.json`. They are generated/local evidence and
remain excluded from normal commits by the repository artifact policy.

A successful async-feedback visualization was replayed from episode 1 using
`scripts/render_safe_panda_3d_feedback_episode.py`. Outputs are under
`artifacts/safe_panda_3d_feedback_episode_01/`: `robot_motion.gif`,
`trajectory_3d_comparison.png`, `raw_smoothed_and_safety.png`, `metrics.json`,
and `episode_summary.json`. The replay reached `goal`, remained collision-free,
used provider gamma `0.03` after `0.548 s`, and recorded a minimum true
clearance of approximately `0.02287 m`.

## Next work items for the 3-D profile

1. Inspect `trajectory_3d_comparison.png` and `metrics.json` from a full saved
   run; verify that both `x` and `z` deviations are visible over the complete
   episode, not only in the reference path.
2. Investigate the 17 solver fallback steps under `--tangential-subgoal` and
   compare against the same route without that flag.
3. Add a dedicated CLI smoke test if the profile becomes a maintained entrypoint.
4. Keep all 50-case benchmark claims and artifacts separate from this custom
   3-D visualization profile.
5. If publishing the 3-D result, add confidence intervals beyond McNemar and
   explicitly disclose the nominal spline proxy used for provider context.
