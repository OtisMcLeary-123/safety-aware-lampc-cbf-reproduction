## Material Passport

- Material ID: `safe-panda-build-l-mpc-cbf-gamma-015-seed-7`
- Type: deterministic sequential manipulation simulation
- Verification status: `VERIFIED`
- Environment: `PandaBuildL-v3`
- Controller: do-mpc 5.1.1 + CasADi 3.7.2 + IPOPT
- Safe Panda commit: `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c`

## Experiment Result

| Metric | Result |
|---|---:|
| Task success | yes |
| Cubes placed | 4 / 4 |
| Total simulator steps | 909 |
| Minimum measured CBF clearance | 0.030492 m |
| Maximum cube placement error | 0.031542 m |
| Placement tolerance | 0.040000 m |
| Mean MPC solve time | 0.006581 s |
| Maximum MPC solve time | 0.010637 s |
| Control period | 0.040000 s |

All four final cube errors are below the Build-L placement tolerance. The maximum measured MPC solve time remained below the control period.

## State machine

For every cube:

`open → approach → descend → close → attach → lift → transport → place → release → retreat`

Free-space motion phases use closed-loop MPC-CBF. The gripper open/close phases are discrete actions.

## Grasp abstraction

The legacy Build-L environment does not expose a stable grasp primitive. After the end-effector reaches the grasp pose and the gripper closes, the demo creates a fixed PyBullet constraint between the gripper link and cube. The constraint is removed during release. This validates task planning, constrained motion, placement, and rendering; it is not a contact-dynamics grasp benchmark.

## Reproducibility verification

The complete episode was rerun with the same seed into a separate directory. Initial objects, targets, final objects, placement errors, stage events, end-effector trajectory, clearances, task result, and step count matched exactly. Wall-clock solve timings were excluded from exact comparison.

## Command

```bash
python scripts/run_build_l_mpc_cbf.py \
  --gamma 0.15 \
  --seed 7 \
  --output-dir artifacts/build_l_mpc_cbf
```

## Artifacts

- `build_l_motion.gif`: full sequential manipulation animation.
- `build_l_motion_montage.png`: start/middle/final frames.
- `trajectory_and_placement.png`: end-effector path and per-cube placement errors.
- `metrics.json`: full configuration, stage trace, trajectory, clearances, placements, and solver timings.
