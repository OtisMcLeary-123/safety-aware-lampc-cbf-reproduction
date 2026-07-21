# Safe Panda Scenario Authoring and Three-Core-Scenario Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-16
- Verification Status: UNVERIFIED
- Version Label: safe_panda_core_scenarios_plan_v1
- Machine-readable plan: `configs/safe_panda_core_scenarios_150_plan.json`

## Objective

Create a reproducible, configuration-driven benchmark with three core Safe
Panda scenario families and 50 perturbed episodes per family. The mandatory
benchmark contains 150 simulator episodes for one frozen controller profile.
Any later controller comparison must replay the exact same 150 resolved
conditions as paired data.

This plan is not an execution result. The scenario generator, validator, and
runner named below are planned implementation artifacts and do not exist yet.

## Authoritative Scenario-Authoring Sources

The source review on 2026-07-16 found no maintained GUI scenario editor or
separate deployed documentation site specifically for Safe Panda Gym. The
Safe Panda repository contains documentation source, while the maintained
public documentation is the upstream Panda-Gym site.

| Source | Verified revision or page | Finding | Adoption decision |
|---|---|---|---|
| [Safe Panda Gym](https://github.com/tohsin/Safe-panda-gym) | default branch `safe-rl-panda-gym-v2` | Safety-aware Panda-Gym fork built on PyBullet; includes safe tasks and custom-task documentation source. | Simulator lineage only. |
| [Pinned compatibility fork](https://github.com/OtisMcLeary-123/Safe-panda-gym/tree/c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c) | `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c` | Current runtime dependency with Gymnasium/Python 3.12 compatibility and restored safe environments. | Required runtime. |
| [Panda-Gym custom task](https://panda-gym.readthedocs.io/en/latest/custom/custom_task.html) | live documentation | A scenario is implemented by subclassing `Task` and defining reset, observation, achieved goal, success, and reward behavior. | Primary authoring API. |
| [Panda-Gym custom environment](https://panda-gym.readthedocs.io/en/latest/custom/custom_env.html) | live documentation | A Gym environment composes `PyBullet`, a robot, and a task through `RobotTaskEnv`. | Primary composition pattern. |
| [Panda-Gym PyBullet API](https://panda-gym.readthedocs.io/en/latest/base_class/pybullet.html) | live documentation | Scene primitives and URDF assets are created and positioned through the simulation wrapper. | Primary scene-construction API. |
| [Panda-Gym GitHub](https://github.com/qgallouedec/panda-gym/tree/10c4d8adaab07d8a29b2f8dc0aa85f8edd8532ae) | `10c4d8adaab07d8a29b2f8dc0aa85f8edd8532ae` | Upstream implementation of `Task`, `RobotTaskEnv`, Panda robot, and PyBullet wrapper. | Conceptual/API reference; not the installed fork. |
| [Gymnasium custom environment guide](https://gymnasium.farama.org/introduction/create_custom_env/) | live documentation; repository revision `d3f3c7fd3b1519b9edff0db0d51b1d30c9245fb1` | Defines current reset/step contracts, deterministic seeding, registration, and environment checking. | Registration and API compliance. |
| [Bullet Physics](https://github.com/bulletphysics/bullet3/tree/63c4d67e337017f9d8b298c900e9aabdb69296e7) | `63c4d67e337017f9d8b298c900e9aabdb69296e7` | Physics engine beneath Panda-Gym; supports primitive shapes and URDF/SDF/MJCF assets. | Low-level physics reference. |

The pinned wrapper exposes `create_box`, `create_sphere`, `create_cylinder`,
`create_plane`, `create_table`, `loadURDF`, `set_base_pose`, friction setters,
and camera placement. Safe tasks add `compute_cost()` to the normal goal-task
contract.

For moving obstacles, `Task.reset()` is insufficient because it only defines
episode initialization. A per-step scene updater or a controlled override of
`RobotTaskEnv.step()` must update obstacle state before the observation and
safety cost are computed. The existing smooth-dynamic runner already provides
this behavior for one obstacle, so version 1 of the core benchmark remains a
single-obstacle benchmark.

### Non-drop-in alternatives

| Framework | Verified revision | Strength | Boundary |
|---|---|---|---|
| [robosuite](https://github.com/ARISE-Initiative/robosuite/tree/5ce6643f3092639d08f7b0f90ed1c6a84f50552c) | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | Modular MuJoCo Arena, Object, Robot, Task, and placement samplers with Panda support. | Requires simulator and adapter migration. |
| [ManiSkill](https://github.com/mani-skill/ManiSkill/tree/42b68244c1497cef889b04c4f4a78aa01c927f4e) | `42b68244c1497cef889b04c4f4a78aa01c927f4e` | Reusable scene builders, Panda agent, and GPU-parallel task execution. | Requires SAPIEN/ManiSkill migration. |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/b4c321024792976150ca55fddb26fa34480d974e) | `b4c321024792976150ca55fddb26fa34480d974e` | Isaac Sim GUI/USD authoring and `InteractiveSceneCfg` configuration. | Heavy migration; results are not directly comparable with PyBullet. |

The selected approach is therefore a JSON-driven scenario builder on top of
the pinned Safe Panda Gym/PyBullet runtime, not a simulator migration.

## Research Question and Provisional Hypotheses

**Research question:** How robust is the frozen Safe Panda MPC-CBF profile to
perturbations in goal geometry, obstacle motion, obstacle size, and measured
obstacle position across three representative encounter families?

The hypotheses are predeclared but remain provisional until the setup and
resolved instance file are reviewed:

- H1: collision-free goal success differs across encounter families.
- H2: the grazing near-limit family produces more solver failures and lower
  minimum clearance than the head-on family.
- H3: increased observation noise widens the difference between true and
  observed minimum clearance.

No hypothesis claims that contextual feedback is superior. Provider-based
comparison is outside the mandatory feasibility phase.

## Fixed Experiment Contract

The following settings remain identical across all 150 mandatory episodes:

| Item | Frozen value |
|---|---|
| Environment | `PandaReachSafe-v3`, Tiny renderer |
| Task semantics | End-effector reach only; no pick/place claim |
| Controller period | `0.04 s` |
| Horizon | `15` steps |
| Model | 8-state discrete double integrator |
| Reference | `direct_target` |
| CBF profile | fixed `gamma=0.15` |
| EE collision radius | `0.035 m` |
| Sensing | `0.67 s` zero-order hold |
| Maximum episode length | `260` steps |
| Whole-arm certificate | not provided |

Only the predeclared scenario parameters and measurement-noise realizations may
change between episodes.

## Core Scenario 1: Head-On Closing Obstacle

Purpose: test direct-path safety and solver feasibility when an obstacle moves
toward the robot along the goal-forward axis.

Main perturbations:

- Goal offset: `x=-0.015..0.015 m`, `y=0.28..0.32 m`,
  `z=-0.010..0.015 m`.
- Obstacle start: `x=-0.020..0.020 m`, `y=0.38..0.48 m`,
  `z=-0.015..0.020 m`.
- Obstacle velocity: `vy=-0.20..-0.05 m/s`, with small lateral and vertical
  components.
- Obstacle radius: `0.075..0.110 m`.
- Measurement sigma: `0.003..0.008 m`.

## Core Scenario 2: Orthogonal 3-D Crossing

Purpose: test lateral prediction and clearance when an obstacle crosses the
goal path with nonzero height and vertical motion.

Main perturbations:

- Exactly 25 left-to-right and 25 right-to-left episodes.
- Initial lateral magnitude: `0.18..0.32 m`.
- Crossing speed: `0.05..0.20 m/s` toward `x=0`.
- Forward crossing location: `y=0.10..0.22 m`.
- Height: `z=0.02..0.10 m`; vertical velocity `-0.03..0.02 m/s`.
- Obstacle radius: `0.070..0.105 m`.
- Measurement sigma: `0.003..0.008 m`.

The preflight validator rejects conditions whose predicted closest approach is
too large to create a meaningful encounter.

## Core Scenario 3: Grazing Near-Limit Encounter

Purpose: test near-boundary feasibility and the difference between true and
measured clearance under stronger observation noise.

Main perturbations:

- Exactly 25 episodes per lateral side.
- Goal lateral magnitude: `0.06..0.12 m`.
- Obstacle lateral position is derived from the combined collision radius plus
  a grazing margin of `-0.012..0.025 m`.
- Obstacle speed: `vy=-0.20..-0.12 m/s`.
- Obstacle radius: `0.080..0.105 m`.
- Measurement sigma: `0.008..0.015 m`.

Negative grazing margins intentionally create a subset of collision-course
conditions. They are permitted only when initial true clearance still exceeds
the preflight threshold.

## Sampling and Noise Policy

Each family uses a deterministic 50-point Latin hypercube. This provides
coverage of each parameter range without treating a simple random draw as a
representative distribution.

- Base seed: `20260716`.
- Each family receives a non-overlapping seed block of `10000`.
- Geometry sampling and measurement-noise realization use separate seeds.
- Signed left/right variables are exactly balanced rather than randomly
  imbalanced.
- All 150 resolved conditions are written once to
  `configs/safe_panda_core_scenarios_150_instances.json` and hashed before any
  simulator run.
- Rejection sampling is limited to 1000 attempts per episode. Exhaustion is a
  preflight failure, not permission to widen ranges silently.

Measurement noise is Gaussian and applied only at sensing updates; the held
measurement remains constant between updates. Since Gaussian noise is
unbounded, no formal bounded-error safety claim may be derived from this phase.

## Preflight Geometry Gates

Every resolved episode must satisfy:

1. All values are finite and inside the declared relative workspace.
2. Initial true clearance is at least `0.03 m`.
3. The goal is outside the obstacle inflated by EE radius and goal tolerance.
4. The obstacle trajectory produces a meaningful predicted encounter; the
   closest approach to the nominal path is at most `0.06 m` beyond the combined
   collision radius.
5. Scenario ID, episode ID, geometry seed, measurement seed, and setup hash are
   unique and recorded.

## Execution Stages

### Stage 0: Implement and test the authoring layer

Planned files:

- `src/lampc_cbf/core_scenarios.py`: typed schema, Latin-hypercube generator,
  derived parameters, validation, and hashing.
- `scripts/prepare_safe_panda_core_scenarios.py`: freeze the 150 conditions and
  write preflight evidence.
- `scripts/run_safe_panda_core_scenarios.py`: resumable execution and summary.
- `tests/test_core_scenarios.py`: deterministic generation, range, balance,
  rejection, hashing, resume, and row-count tests.

Version 1 should use the existing one-obstacle smooth-dynamic interface. It
must not modify do-mpc, CasADi, IPOPT, or Safe Panda modules merely to make a
scenario pass.

### Stage 1: Smoke gate

Run one median-difficulty condition from each family. Require:

- three completed rows;
- valid observation and action shapes;
- no initial collision;
- finite clearance and solver diagnostics;
- no artifact outside the declared output directory.

### Stage 2: Pilot gate

Run five frozen conditions per family, 15 total, with no provider calls. Review
outcome diversity, runtime, solver rejection, and whether the encounter filter
creates neither trivial nor universally impossible conditions. Any range
change creates plan version 2 and a new resolved-instance hash.

### Stage 3: Mandatory benchmark

Run 50 conditions per family with fixed `gamma=0.15`, producing exactly 150
rows. Write the checkpoint after every row. A crashed or invalid episode is
recorded and reviewed; it is not silently retried or replaced.

Planned commands after implementation:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_safe_panda_core_scenarios.py
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_core_scenarios.py
```

### Stage 4: Optional paired controller replay

Only after the mandatory feasibility and integrity gates pass, replay the same
150 conditions for another controller profile. Two methods produce 300 rows.
Provider requests require separate authorization and a scenario-specific local
checkpoint; old provider records must not be reused.

## Metrics and Statistical Plan

Primary endpoint:

```text
collision_free_goal_success = reached_goal AND NOT collision
```

Secondary endpoints include collision count, true and observed minimum
clearance, true CBF residual, steps to goal, final goal distance, solver
failures/rejections, deadline misses, emergency fallback, path length, control
effort, and model/action transition error.

For each scenario family report:

- success and collision proportions with Wilson 95% intervals;
- median and mean minimum true clearance with bootstrap 95% intervals;
- solver-failure and fallback episode counts;
- distribution plots against the main perturbed variables.

Across the three families report an equal-weight macro average and the
worst-family result. Do not pool 150 rows without retaining family labels.

For a later paired two-method comparison:

- use exact McNemar tests for paired binary success within each family;
- use paired bootstrap intervals for clearance differences;
- apply Holm correction across the three family-level hypothesis tests;
- report effect sizes and confidence intervals even when `p < 0.05`.

Fifty episodes per family support benchmark estimation but do not guarantee
power for small method differences. Claims must therefore emphasize interval
width and observed effect size.

## Output Contract and Monitoring

Expected local outputs:

| Output | Purpose |
|---|---|
| `preflight_summary.json` | counts, ranges, hashes, rejection diagnostics |
| `resolved_instances.json` | exact frozen runner inputs and seeds |
| `run_checkpoint.json` | resumable row prefix and setup hash |
| `episodes.csv` | one row per completed episode and method |
| `benchmark_summary.json` | family-level and macro-level summaries |

All generated results remain under
`artifacts/safe_panda_core_scenarios_150/` and are ignored unless a compact
aggregate is explicitly selected for publication. Monitor checkpoint row count,
last-update time, process liveness, and per-episode timeout. Do not automatically
retry or widen scenario bounds after a failure.

## Promotion Gates

The benchmark may be labeled complete only when:

- the resolved instance file contains exactly 150 unique rows;
- each family contains exactly 50 rows and balanced side counts where required;
- setup and instance hashes match the checkpoint and summary;
- no provider call occurs in the mandatory phase;
- all invalid geometry and initial-collision checks are zero;
- every missing/crashed episode has an explicit diagnostic record;
- the full test suite passes;
- limitations retain the EE-only, simulator-only, and non-whole-arm wording.

## Ownership and Integration Boundaries

- Python integration owns the scenario schema, generator, CLI, manifests,
  checkpointing, and tests.
- Safe Panda ownership is involved only if a new environment/task class or
  observation mapping is required.
- do-mpc, CasADi, and IPOPT modules remain unchanged during scenario generation.
  Any controller or solver remedy is a separate versioned profile.

This separation prevents scenario tuning from silently changing the controller
being evaluated.
