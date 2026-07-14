# Collision-Cone Liveness Protocol

## Motivation

The preceding 20-episode safety-reflex ablation completed without collision,
but the selected collision-cone controller stalled in 4/20 scenes and invoked
the reflex 1,291 times. This protocol treats that result as a liveness and
nominal/reflex-alignment problem. It does not increase the 140-step timeout.

The implementation adapts three public design patterns without copying their
controller formulation:

- [Policy Library CBF](https://github.com/tkkim-robot/plcbf): hard-screen a
  finite set of closed-loop fallback candidates before scoring task progress.
- [Circulation-embedded CBF](https://github.com/sh-keyumarsi/Circulation-embedded-CBF):
  retain a signed circulation side to avoid undesirable equilibria.
- [Multi-object CLF-CBF navigation](https://github.com/UMich-BipedLab/multi_object_avoidance_via_clf_cbf):
  feed a temporary liveness subgoal to the nominal planner instead of asking
  the fast safety filter to fight an unchanged reference indefinitely.

These references are architectural precedents. They do not establish a formal
safety or liveness proof for the Safe Panda model in this repository.

## Frozen paired variants

All variants use the same seed, obstacle conditions, velocity uncertainty tube,
OSCBF limits, solver policy, and raw-trajectory safety evaluator.

1. `collision_cone_baseline`: existing memoryless cone projection.
2. `side_latch`: deterministic circulation side, 15-step dwell and hysteresis.
3. `policy_library`: hard-screened cone, tangent, circulation, obstacle-match,
   and stop candidates; score only candidates with non-negative rollout
   clearance and barrier residual.
4. `policy_library_tangential_subgoal`: additionally send a 60 mm, goal-biased,
   selected-side waypoint to the next MPC horizon.

The policy score combines predicted goal progress, deviation from nominal MPC,
and a side-switch penalty. Safety constraints are not softened by the score.

If a sensor update expands the uncertainty tube around the current state so
far that no bounded command can restore robust clearance within one 40 ms
step, the event is explicitly recorded as `policy_library_recovery`. Recovery
candidates must remain outside the physical collision set and satisfy its
collision-cone residual; among those, the controller selects same-side motion
within 5 mm of the best robust-clearance recovery, then minimizes the nominal
task score. Such a step is physical-safety filtered but is not reported as
robust-feasible.

## Gate before development-100 or timeout changes

- collision rate <= 5%;
- controller-stall rate <= 10%;
- reflex interventions reduced by at least 50% versus the paired baseline;
- mean side switches <= 2 per episode;
- solver rejection and deadline-miss rates <= 1%;
- maximum episode p99 solve time <= 40 ms.

Failure of an efficacy gate is retained as a negative result. The next stage is
diagnosis or a DPCBF ablation, not a longer timeout.

## Ablation result: 20 paired episodes

The frozen 140-step ablation passed all promotion gates on 2026-07-14.

| Metric | Collision-cone baseline | Policy library + subgoal |
|---|---:|---:|
| Collisions | 0/20 | 0/20 |
| Controller stalls | 4/20 | 0/20 |
| Goal successes | 0/20 | 3/20 |
| Mean goal progress | 77.3 mm | 198.8 mm |
| Reflex interventions | 1,290 | 587 |
| Mean side switches | n/a | 0.0 |
| Mean minimum raw clearance | 70.7 mm | 56.5 mm |
| Maximum episode p99 solve time | 32.6 ms | 20.3 ms |

The target reduced intervention count by 54.5% and stall rate by 20 percentage
points while retaining zero observed collisions. It entered explicitly labeled
robust-recovery 383 times. Consequently, these runs support improved empirical
liveness and physical collision filtering, but not a continuous robust-safety
claim under sensor-tube discontinuities. Seventeen target episodes still ended
at the frozen 140-step budget; timeout should remain unchanged until a separate
physics-derived completion budget is preregistered.

## Physics-derived development budget

Before starting development-100, the completion budget is derived as

```math
T_{max} = \frac{d_{goal}}{v_{ref}}
        + \frac{(\pi-2)r_{detour}}{v_{ref}}
        + T_{sensor} + T_{feedback} + T_{recovery}.
```

The detour term replaces the diameter crossing an obstacle with a semicircle.
Frozen inputs are `d_goal=0.30 m`, `v_ref=0.08 m/s`,
`r_detour=0.10+0.035+0.08=0.215 m`, `T_sensor=0.67 s`,
`T_feedback=0.4971896839560941 s`, `T_recovery=0.80 s`, and
`dt=0.04 s`. Therefore:

- direct path: 3.750 s;
- extra semicircular detour: 3.068 s;
- sensing/feedback/recovery reserve: 1.967 s;
- total: 8.785 s;
- development budget: `ceil(8.785/0.04) = 220` control steps.

The runner refuses to start unless the preceding 20-episode summary exists,
uses the expected protocol, contains the selected target, and has a passed
gate. Development uses only the selected C3BF policy-library/subgoal stack; it
does not rerun discarded variants or DPCBF.

## Development-100 result

The selected C3BF stack completed the frozen 100-episode deterministic grid
with the 220-step budget and passed every development gate.

| Metric | Development result |
|---|---:|
| Goal success | 82/100 (82.0%) |
| Success Wilson 95% interval | [73.3%, 88.3%] |
| Safety timeout | 18/100 |
| Collision | 0/100; Wilson upper bound 3.70% |
| Controller stall | 0/100; Wilson upper bound 3.70% |
| Mean goal progress | 240.9 mm |
| Mean / worst raw clearance | 56.6 / 27.5 mm |
| Reflex interventions | 2,974 / 17,698 steps (16.8%) |
| Robust-recovery steps | 1,992 / 17,698 (11.3%) |
| Mean side switches | 0.0 per episode |
| Solver rejections | 15 / 17,698 (0.0848%) |
| Deadline misses | 3 / 17,698 (0.0170%) |
| Maximum episode p99 solve time | 28.0 ms |

All ten scenes at the slowest obstacle speed (`0.025 m/s`) timed out. The
other eight timeouts occurred between `0.044` and `0.103 m/s`; no scene at or
above `0.122 m/s` timed out. Twelve of eighteen timeouts used negative lateral
offsets. Three timeout episodes stopped within 4.1 mm beyond the 50 mm goal
threshold. These are completion-budget effects with positive progress, not the
controller stalls observed before the liveness remediation.

The run still contains 1,992 explicitly labeled robust-recovery steps, so the
result supports empirical physical collision avoidance and improved task
completion but does not establish continuous robust feasibility. Development
success is not confirmatory-500 evidence.

## Run

```bash
PYTHONPATH=src python scripts/run_collision_cone_liveness_ablation.py \
  --episodes 20 --workers 4 --max-steps 140
```

Outputs are written to `artifacts/collision_cone_liveness_ablation_20/` and are
not committed by default.

Run the gated development stage with:

```bash
PYTHONPATH=src python scripts/run_liveness_development.py \
  --episodes 100 --workers 4
```
