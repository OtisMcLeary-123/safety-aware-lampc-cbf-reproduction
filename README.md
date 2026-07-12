# Safety-Aware LaMPC-CBF Reproduction

Reproduction scaffold for *Safety-Aware Optimal Control With Language-Guided Online Parameter Adjustment via Large Language Models* (IEEE Access, 2026, DOI: 10.1109/ACCESS.2026.3664145).

The implementation follows the paper's reported stack:

- Python
- do-mpc
- CasADi
- IPOPT
- Safe Panda Gym

## Status

The mathematical components, simulator adapter, static and dynamic obstacle
experiments, and sequential Build-L demonstration are implemented and covered
by unit tests. This remains a clean-room reconstruction because the authors'
controller source is not public.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,simulation]'
python -m pytest -q
```

The simulation extra pins do-mpc and CasADi and installs a tested compatibility fork of the paper-cited Safe Panda Gym repository at a fixed commit. The compatibility patch restores the safe environment exports and migrates their runtime API to Gymnasium on Python 3.12.

## Implemented components

- Paper-aligned 8-state, 4-input discrete dynamics and MPC configuration.
- CasADi expressions for the obstacle barrier, discrete CBF residual, and orientation/velocity regularizer.
- IPOPT options, diagnostics, and fail-closed control acceptance policy.
- Safe Panda Gym adapter with deterministic seeding, 0.67 s zero-order-held obstacle sensing, and 0.005 m Gaussian measurement noise.
- Typed configuration and a dry-run CLI that never sends prompts or credentials externally.
- Deterministic Safe Panda MPC-CBF demo with rendered frames, trajectory/clearance plot, GIF, and JSON metrics.
- Paper-aligned moving spherical obstacle with constant velocity, noisy 0.67 s
  zero-order-held sensing, and an online do-mpc TVP CBF constraint.

## Render the MPC-CBF demo

```bash
source .venv/bin/activate
python scripts/run_safe_panda_mpc_cbf.py --gamma 0.10
```

The default scene places a spherical unsafe region on the straight line from the initial end-effector pose to the goal. A deterministic Task-Planner-style route supplies two safe low-level waypoints around the obstacle; fixed-gamma MPC-CBF solves every control step and enforces the exclusion radius. Results are written under `artifacts/mpc_cbf_demo/`.

## Run the dynamic-obstacle CBF experiment

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_dynamic_obstacle_mpc_cbf.py
```

The red sphere crosses the robot's route at constant Cartesian velocity. Its
position is measured every 0.67 s with 0.005 m Gaussian noise. Between sensor
updates, the measured center is held as a do-mpc TVP and replicated across the
15-step prediction horizon, matching the paper's stated static-within-horizon
assumption. The result folder contains an animation, rendered montage, safety
plot, and full JSON trace.

## Run trajectory-smoothness ablation

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_smoothness_ablation.py
```

This compares the waypoint baseline with a continuous B-spline reference at
`Δu` weights 0.5, 1, 2, and 5, then evaluates an augmented-state `Δ²u` jerk
penalty. It reports path length, curvature, acceleration RMS, jerk RMS/max,
raw safety clearance, and solve time. Plot smoothing is explicitly marked as
visualization-only; collision and clearance always use raw simulated poses.
Ruckig is gated on the MPC-only result rather than installed unconditionally.

## Configure MPC-CBF from language with Hugging Face

```bash
source .venv/bin/activate
python -m pip install -e '.[llm,simulation]'
PYTHONPATH=src python scripts/run_hf_configured_mpc_cbf.py
```

The default mapper uses structured output from
`Qwen/Qwen3-235B-A22B-Instruct-2507`, validates one of five experimental gamma
levels locally, and fails closed to `gamma=0.05`. The token remains in the
ignored `hftoken.txt` file and the LLM is never called inside the 40 ms control
loop. See [docs/HF_LLM_INTEGRATION.md](docs/HF_LLM_INTEGRATION.md).

## Run the paper-style hard scene

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_hard_scene_study.py
PYTHONPATH=src python scripts/render_hard_scene_examples.py
```

This removes the hand-designed avoidance waypoint, sends a randomized obstacle
head-on toward a straight reference, and runs 50 paired episodes for a distance
constraint, fixed CBF, LLM-selected initial gamma, and measured-latency online
feedback. The committed run is a negative aggregate result: 34% success for
the distance baseline versus 32% for each CBF/LLM condition, while the 1.612 s
feedback response arrived too late for every episode. A paired rendered case
still shows `gamma=0.02` avoiding a collision that the distance constraint does
not. See [docs/HARD_SCENE_STUDY.md](docs/HARD_SCENE_STUDY.md).

## Render the Build-L scene

```bash
source .venv/bin/activate
python scripts/render_build_l_scene.py
```

This recreates the repository's four colored cubes and translucent L-shaped target layout using the restored `PandaBuildL-v3` Gymnasium environment.

## Run sequential Build-L manipulation

```bash
source .venv/bin/activate
python scripts/run_build_l_mpc_cbf.py --gamma 0.15
```

The state machine opens the gripper, approaches a cube, descends, closes and attaches it, lifts, transports, places, releases, and retreats. Free-space motion is solved by MPC-CBF; the deterministic grasp abstraction uses a fixed PyBullet constraint because the legacy Build-L environment does not provide a stable grasp primitive.


## Reproducibility boundary

The repository reproduces the mathematical controller and simulation setup from the published description. The authors' public project repository currently contains no implementation beyond its README, so this is a clean-room reproduction rather than a bit-for-bit rerun. Prompt text, unpublished assets, environment schema, object geometry, and other details not supplied by the paper are documented as explicit assumptions.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the evidence boundary and remaining validation work.
See [docs/CLEAN_ROOM_STRATEGY.md](docs/CLEAN_ROOM_STRATEGY.md) for the reconstruction strategy required while the authors' controller and prompt code remain unavailable.
