## Material Passport

- Material ID: `dynamic-obstacle-smoothness-ablation-seed-11`
- Type: deterministic controller ablation
- Verification status: `VERIFIED`
- Generated: 2026-07-12 (Asia/Taipei)
- Controller: do-mpc 5.1.1 + CasADi 3.7.2 + IPOPT
- Simulator: Safe Panda Gym compatibility commit `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c`
- Safety evidence: raw end-effector positions only

## Command

```bash
PYTHONPATH=src .venv/bin/python scripts/run_smoothness_ablation.py
```

## Primary result

All six evaluated variants reached the goal without collision and solved below
the 40 ms control period. Among the requested `Δu` weights, `spline_du_5` had
the lowest raw end-effector jerk RMS and was selected.

| Metric | Waypoint baseline | Selected B-spline `Δu=5` | Change |
|---|---:|---:|---:|
| Path length | 0.614504 m | 0.728150 m | +18.49% |
| Mean curvature | 25.735760 1/m | 5.562920 1/m | -78.38% |
| Maximum curvature | 134.808471 1/m | 7.163154 1/m | -94.69% |
| Acceleration RMS | 1.760006 m/s² | 1.401658 m/s² | -20.36% |
| Jerk RMS | 46.049082 m/s³ | 5.952746 m/s³ | -87.07% |
| Maximum jerk | 213.622133 m/s³ | 12.729572 m/s³ | -94.04% |
| Minimum true clearance | 0.000865 m | 0.036635 m | safer |
| Final goal distance | 0.048340 m | 0.032824 m | smaller |
| Mean solve time | 8.285 ms | 6.369 ms | -23.13% |
| Maximum solve time | 16.717 ms | 8.927 ms | below 40 ms |

The smoother trajectory is longer because it follows a broad continuous arc
behind the crossing obstacle rather than the baseline's piecewise waypoint
route. This is a safety/smoothness versus path-length trade-off.

## `Δu` ablation

| Variant | Jerk RMS [m/s³] | Accel RMS [m/s²] | Mean curvature [1/m] | Raw min clearance [m] |
|---|---:|---:|---:|---:|
| B-spline `Δu=0.5` | 6.724808 | 1.531400 | 5.967365 | 0.023999 |
| B-spline `Δu=1` | 6.670205 | 1.505351 | 5.886072 | 0.026603 |
| B-spline `Δu=2` | 6.626815 | 1.482680 | 5.790090 | 0.030266 |
| B-spline `Δu=5` | **5.952746** | **1.401658** | 5.562920 | 0.036635 |

## Jerk-cost finding

The augmented 12-state controller stores the preceding input increment and
penalizes `Δ²u[k] = (u[k] - u[k-1]) - Δu[k-1]`.

At `Δu=5, Δ²u=5`, command jerk RMS fell another 8.46% and command maximum jerk
fell 30.87%. However, raw end-effector jerk RMS increased 3.96%, from 5.952746
to 6.188593 m/s³. The formulation is retained for future tuning, but it is not
selected as the best physical-trajectory result.

## Ruckig decision

Ruckig was not integrated. The predeclared gate required safe real-time MPC to
fail to reduce raw jerk RMS by at least 25%, or fail to reduce maximum raw jerk.
MPC-only reduced these values by 87.07% and 94.04%, respectively, while
remaining collision-free and real-time. A post-controller trajectory modifier
is therefore not justified by this experiment.

## Visualization boundary

Each variant saves the raw path, a visualization-only smoothing spline, and the
continuous B-spline MPC reference. The visualization spline is never used for
clearance, collision, goal success, or the smoothness metrics in this report.

## Reproducibility verification

The selected `spline_du_5` episode was replayed into an independent output
directory with seed 11. Raw positions, controls, true/measured obstacle traces,
clearances, reference path, sensor-update indices, goal distances, and terminal
result matched exactly. Wall-clock IPOPT timings were excluded from the exact
comparison.

## Artifacts

- `ablation_summary.json`: complete comparison and Ruckig decision.
- `ablation_metrics.png`: smoothness and raw-safety bar charts.
- `raw_trajectory_comparison.png`: waypoint, selected MPC, and jerk trajectories.
- `selected_spline_du_5/robot_motion.gif`: rendered selected controller motion.
- `selected_spline_du_5/raw_smoothed_and_safety.png`: raw/smoothed/reference paths and raw clearance.
- Per-variant folders: raw trajectories, references, controls, timings, and plots.
