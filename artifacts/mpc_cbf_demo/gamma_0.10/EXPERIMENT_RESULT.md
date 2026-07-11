## Material Passport

- Material ID: `safe-panda-mpc-cbf-gamma-010-seed-7`
- Type: deterministic simulation result
- Verification status: `VERIFIED`
- Generated: 2026-07-11 (Asia/Taipei)
- Controller: do-mpc 5.1.1 + CasADi 3.7.2 + IPOPT
- Simulator: Safe Panda Gym compatibility commit `97a7bbf6c619e5e20ba3bde3c5f423b435a3062a`

## Experiment Result

Command:

```bash
python scripts/run_safe_panda_mpc_cbf.py \
  --gamma 0.10 \
  --max-steps 300 \
  --render-stride 2 \
  --output-dir artifacts/mpc_cbf_demo/gamma_0.10
```

| Metric | Result |
|---|---:|
| Goal reached | yes |
| Collision | no |
| Control steps | 53 |
| Simulated duration | 2.12 s |
| Minimum CBF clearance | 0.001755 m |
| Final goal distance | 0.048915 m |
| Mean MPC solve time | 0.007003 s |
| Maximum MPC solve time | 0.013473 s |
| Control period | 0.040 s |

The maximum measured solve time remained below the 40 ms control period.

## Reproducibility verification

The episode was rerun from the same seed into a separate output directory. Positions, controls, clearances, goal distances, active subtask indices, step count, collision status, and terminal result matched exactly. Wall-clock solver timings were excluded from exact comparison.

## Interpretation boundary

This verifies the reconstructed fixed-gamma MPC-CBF controller, deterministic waypoint layer, Safe Panda API integration, rendering, and result pipeline. It does not reproduce the paper's unpublished LLM prompts, moving-obstacle scene, or Figure 5/6 numeric data.

## Artifacts

- `robot_motion.gif`: animated robot motion.
- `robot_motion_montage.png`: start/middle/final rendered frames.
- `trajectory_and_clearance.png`: top-view path and safety/convergence curves.
- `metrics.json`: configuration, full trajectory, controls, clearances, targets, and solve times.
