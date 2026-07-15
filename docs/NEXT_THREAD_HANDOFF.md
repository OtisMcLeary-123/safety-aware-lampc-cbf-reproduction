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
