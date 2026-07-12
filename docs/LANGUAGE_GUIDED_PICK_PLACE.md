# Language-guided blue-on-red pick-and-place

## Material Passport

- Experiment ID: `language-guided-blue-on-red-seed-7`
- Status: completed successfully
- Environment: `PandaBuildL-v3`, Tiny renderer, third-person oblique camera
- User instruction: “Safely pick up the blue cube and put it on the red cube.
  Keep a generous distance from the moving obstacle.”
- LLM: `Qwen/Qwen3-235B-A22B-Instruct-2507` through DeepInfra
- LLM output: safety level 1/5, `gamma=0.02`, no fallback
- Recorded LLM latency: 2.689 s
- Controller: do-mpc + CasADi + IPOPT, 40 ms control period, 15-step horizon
- Scene: four cubes (blue, green, orange, red) and one moving white sphere
- Obstacle: radius 0.055 m, velocity `(0, -0.01, 0)` m/s
- Gripper collision approximation: radius 0.012 m
- CBF safety-boundary radius: 0.067 m
- Sensing: 0.67 s zero-order hold with 0.005 m Gaussian noise
- Seed: 7
- Grasp abstraction: fixed PyBullet constraint after physical approach and
  gripper closure; released during placement

## Result

| Metric | Value |
|---|---:|
| Goal success | true |
| Cubes placed | 1/1 |
| Blue-on-red placement error | 13.23 mm |
| Minimum raw dynamic-obstacle clearance | 26.24 mm |
| Minimum other-cube clearance | 42.65 mm |
| Mean MPC solve time | 8.72 ms |
| Maximum MPC solve time | 19.56 ms |
| Total simulator steps | 222 |

Clearance is measured outside the combined 0.067 m safety boundary. A positive
26.24 mm therefore means the closest center-to-center separation was about
93.24 mm. This is one deterministic demonstration, not a statistical estimate
of controller success.

## Control interpretation

The LLM does not generate Cartesian trajectory samples. It maps the language
safety intent to `gamma=0.02` and supplies task primitives. The Task Planner
uses an avoidance waypoint for the conservative head-on encounter. MPC-CBF
then optimizes and executes every free-space segment. Dynamic CBF is active for
approach, lift, avoidance, and transport; the post-transport vertical placement
and retreat use nominal MPC after the safe `move_above(red_cube)` primitive is
complete. Raw obstacle clearance is still recorded over all phases.

The paper-style figure uses the following evidence-preserving legend:

- Orange: raw executed approach trajectory.
- Red dotted: nominal straight transport route, provided only as a comparison.
- Black: raw executed MPC-CBF lift/avoidance/transport trajectory.
- Yellow dashed: projected CBF boundary `h=0`; it is not a robot trajectory.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_language_guided_pick_place.py
```

Outputs are written to `artifacts/language_guided_pick_place/`:

- `language_guided_mpc_cbf.png`: annotated paper-style camera figure.
- `build_l_motion.gif`: complete simulated motion.
- `build_l_motion_montage.png`: start, middle, and final camera frames.
- `trajectory_and_placement.png`: top-down trajectory and placement error.
- `metrics.json`: raw trajectory, obstacle measurements, stage labels,
  clearances, solve times, LLM metadata, and configuration.

