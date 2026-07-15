# 3-D Avoidance Demo

This is an opt-in visualization profile for producing an explicit two-axis
(`x`/`z`) avoidance trajectory while the robot progresses toward the goal on
`y`. It does not alter the paired 50-scenario benchmark or its paper-fidelity
defaults.

The profile combines:

- a 3-D `behind_spline` reference with explicit intermediate waypoint offsets;
- nonzero obstacle height and vertical obstacle velocity;
- optional tangential safety-reflex subgoals;
- per-axis position tracking weights in the MPC objective;
- the unchanged isotropic Euclidean CBF for the safety guarantee.

Run it from the repository root:

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

The output directory contains `metrics.json`, `raw_smoothed_and_safety.png`,
the Figure-5-style `trajectory_3d_comparison.png`, and, with
`--save-animation`, `robot_motion.gif`. `metrics.json` records the
3-D route points, interpolated reference path, raw end-effector positions,
obstacle positions, controls, clearances, and solver diagnostics.

The CBF remains spherical/Euclidean. Axis weights shape tracking of the
explicit route; they do not replace the collision barrier with an unsafe
axis-weighted barrier.
