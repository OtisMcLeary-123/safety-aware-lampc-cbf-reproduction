# Language-guided blue-on-red pick-and-place

## Material Passport

- Experiment ID: `language-guided-blue-on-red-seed-7`
- Status: completed successfully
- Environment: `PandaBuildL-v3`, Tiny renderer, third-person oblique camera
- User instruction: “Safely pick up the blue cube and put it on the red cube.
  Keep a generous distance from the moving obstacle.”
- LLM: `Qwen/Qwen3-235B-A22B-Instruct-2507` through DeepInfra
- TP/OD result: one validated four-step macro and two accepted optimization
  specs, `OD fallbacks=0`
- OD parameters for both moves: `q=1`, `delta_u=1`, `gamma=0.05`,
  0.1 m/s linear speed limit, 0.05 m collision clearance, and workspace
  `[-0.30,-0.30,0.00]` to `[0.25,0.30,0.50]` m
- Recorded live inference latency: TP 9.624 s, OD total 25.246 s
- Final artifact execution source: deterministic simulator replay of the
  recorded and locally revalidated accepted TP/OD responses; no API call
  occurs during replay
- Controller: do-mpc + CasADi + IPOPT, 40 ms control period, 15-step horizon
- Scene: four cubes (blue, green, orange, red) and one moving white sphere
- Obstacle: radius 0.055 m, velocity `(0, -0.01, 0)` m/s
- Gripper collision approximation: radius 0.012 m
- Physical gripper-obstacle boundary radius: 0.067 m
- Active OD-CBF boundary radius: 0.105 m
- Sensing: 0.67 s zero-order hold with 0.005 m Gaussian noise
- Seed: 7
- Grasp abstraction: fixed PyBullet constraint after physical approach and
  gripper closure; released during placement

## Result

| Metric | Value |
|---|---:|
| Goal success | true |
| Collision-free | true |
| Cubes placed | 1/1 |
| Blue-on-red placement error | 7.11 mm |
| Minimum raw physical dynamic-obstacle clearance | 39.10 mm |
| Minimum raw margin outside active OD-CBF boundary | 1.10 mm |
| Minimum other-cube clearance | 41.73 mm |
| Mean MPC solve time | 8.08 ms |
| Maximum MPC solve time | 16.31 ms |
| Total simulator steps | 121 |

The closest raw center-to-center separation was approximately 106.10 mm.
Relative to the 67 mm physical gripper-obstacle boundary this is 39.10 mm of
clearance. Relative to the active 105 mm OD-CBF boundary it is only 1.10 mm of
margin. Both values use raw simulator poses; neither is computed from a
smoothed plot. This is one successful demonstration, not a statistical
estimate of TP/OD reliability or controller success.

## Control interpretation

The LLM does not generate Cartesian trajectory samples or executable CasADi
source. TP produces the validated sequence `move(blue) -> close ->
move_above(red) -> open`. OD produces bounded JSON parameters for the MPC
objective, workspace/input limits, collision clearance, and CBF gamma. A
trusted executor expands the macro into approach, descend, attach, lift,
avoidance transport, place, detach, and retreat stages. The avoidance waypoint
is advisory rather than part of the paper-level success definition; its
completion remains visible in `stage_events`. MPC-CBF optimizes every movement
stage, including placement and retreat, using the active OD specification.

The paper-style figure uses the following evidence-preserving legend:

- Orange: raw executed approach trajectory.
- Red dotted: nominal straight transport route, provided only as a comparison.
- Black: raw executed MPC-CBF lift/avoidance/transport trajectory.
- Yellow dashed: projected active OD-CBF boundary `h=0` at radius 0.105 m; it
  is not a robot trajectory.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_language_guided_pick_place.py
```

To reproduce the final simulator artifacts without another external model
call, revalidate and replay the accepted responses stored in `metrics.json`:

```bash
PYTHONPATH=src python scripts/run_language_guided_pick_place.py \
  --replay-metrics artifacts/language_guided_pick_place/metrics.json
```

Outputs are written to `artifacts/language_guided_pick_place/`:

- `language_guided_mpc_cbf.png`: annotated paper-style camera figure.
- `build_l_motion.gif`: complete simulated motion.
- `build_l_motion_montage.png`: start, middle, and final camera frames.
- `trajectory_and_placement.png`: top-down trajectory and placement error.
- `metrics.json`: raw trajectory, obstacle measurements, stage labels,
  physical clearances, active CBF margins/radii, solve times, raw accepted OD
  responses, replay provenance, LLM metadata, and configuration.
