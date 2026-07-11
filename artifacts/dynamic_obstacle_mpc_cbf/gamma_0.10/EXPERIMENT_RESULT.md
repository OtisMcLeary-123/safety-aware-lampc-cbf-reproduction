## Material Passport

- Material ID: `safe-panda-dynamic-obstacle-cbf-gamma-010-seed-11`
- Type: deterministic moving-obstacle simulation result
- Verification status: `VERIFIED`
- Generated: 2026-07-11 (Asia/Taipei)
- Controller: do-mpc 5.1.1 + CasADi 3.7.2 + IPOPT
- Simulator: Safe Panda Gym compatibility commit `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c`

## Experiment Result

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dynamic_obstacle_mpc_cbf.py
```

| Metric | Result |
|---|---:|
| Goal reached | yes |
| Collision | no |
| Control steps | 67 |
| Simulated duration | 2.68 s |
| Sensor updates | 4 |
| Minimum true clearance | 0.000865 m |
| Minimum measured CBF clearance | 0.001780 m |
| Final goal distance | 0.048340 m |
| Mean MPC solve time | 0.008285 s |
| Maximum MPC solve time | 0.016717 s |
| Control period | 0.040 s |

The maximum measured solve time remained below the 40 ms control period.

## Paper alignment

- The red spherical obstacle has constant Cartesian velocity of 0.06 m/s.
- Robot control and MPC updates run every 0.04 s.
- Obstacle sensing runs every 0.67 s with Gaussian noise sigma = 0.005 m.
- Each sensor value is zero-order held and treated as static throughout the
  current 15-step MPC prediction horizon.
- The discrete CBF enforces
  `h(x[k+1]) - (1-gamma) h(x[k]) >= 0` with gamma = 0.10.

The deterministic task-planner layer selects two waypoints behind the moving
obstacle. The MPC-CBF is the online low-level controller and safety constraint;
the waypoint policy is an explicit reconstruction assumption because the
paper's task-planner implementation is not public.

## Reproducibility verification

The episode was rerun from seed 11 into a separate directory. Positions,
controls, true/measured obstacle traces, clearances, goal distances, subtask
indices, sensor update indices, and terminal results matched exactly. Only
wall-clock solver timings were excluded from exact comparison.

## Artifacts

- `robot_motion.gif`: animated robot and moving red sphere.
- `robot_motion_montage.png`: start/middle/final rendered frames.
- `trajectory_and_clearance.png`: robot/obstacle paths and true/measured safety margins.
- `metrics.json`: configuration and complete numeric trace.
