# Cartesian DPCBF Ablation

## Scope

This experiment adapts the public
[`KinematicBicycle2D_DPCBF`](https://github.com/tkkim-robot/safe_control/blob/main/dynamic_env/kinematic_bicycle2D_dpcbf.py)
implementation from `safe_control`. The source method rotates relative velocity
into a line-of-sight frame and replaces a collision cone with a distance- and
velocity-adaptive parabolic boundary.

The reproduction retains the line-of-sight longitudinal term and generalizes
the single planar lateral velocity to the squared norm of the 3D component
perpendicular to line of sight. Default source parameters are retained:
`s=1.05`, `k_lambda=0.10`, `k_mu=0.50`, `eps_d=0.10 m`, and
`eps_v=0.05 m/s`.

This is not a proof-preserving port. The original model is a nonholonomic
kinematic bicycle, whereas this project filters a Cartesian Panda end-effector
velocity. DPCBF is therefore labeled experimental and is not substituted for
the confirmed collision-cone baseline.

## Deterministic gate

Both conditions use the same side latch, policy library, goal-biased subgoal,
uncertainty tube, 140-step budget, and four deterministic scenes.

| Metric | C3BF | Cartesian DPCBF |
|---|---:|---:|
| Collisions | 0/4 | 0/4 |
| Controller stalls | 0/4 | 1/4 |
| Mean goal progress | 111.2 mm | 64.0 mm |
| Mean minimum raw clearance | 60.7 mm | 60.0 mm |
| Reflex interventions | 154 | 133 |
| Robust-recovery steps | 96 | 73 |
| Maximum episode p99 solve time | 36.6 ms | 10.2 ms |

DPCBF reduced intervention count but lost 47.1 mm of mean progress and stalled
in the high-speed `0.20 m/s`, `-45 mm` offset scene. That scene also produced
one circulation-side switch and a lower minimum physical clearance of 28.3 mm.
The deterministic liveness gate therefore failed, and the paired 20-episode
DPCBF run was deliberately not started.

This negative result is consistent with a less restrictive barrier intervening
later in the difficult crossing, but four scenes are insufficient to establish
that explanation statistically. Parameter tuning or a proof for the Cartesian
model must be preregistered before reconsidering promotion.

## Reproduce the blocked grid

```bash
PYTHONPATH=src python scripts/archive/run_dpcbf_ablation.py \
  --episodes 4 --workers 4 --max-steps 140 \
  --output-dir artifacts/dpcbf_grid_4
```

The command exits non-zero when a gate fails. Generated rows and summaries are
kept local by default.
