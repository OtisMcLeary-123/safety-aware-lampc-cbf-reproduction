# Reproducibility notes

## Primary sources

- Paper: *Safety-Aware Optimal Control With Language-Guided Online Parameter Adjustment via Large Language Models*, IEEE Access 14 (2026), DOI `10.1109/ACCESS.2026.3664145`.
- Author project repository: `https://github.com/sohonsong/safety-aware-llm-control`.
- Paper-cited simulator fork: `https://github.com/tohsin/Safe-panda-gym`.

At the time this scaffold was created, the author project repository exposed only a README and MIT license. It did not expose the controller, prompts, experiment scripts, or numeric result files. The implementation here is therefore reconstructed from equations (1)-(19), Algorithm 1, and the experimental setup reported in the paper.

## Paper-aligned constants

| Quantity | Value |
|---|---:|
| Control period | `0.04 s` |
| Prediction horizon | `15` |
| Prediction duration | `0.6 s` |
| Sensor period | `0.67 s` |
| Obstacle measurement noise | `0.005 m` standard deviation |
| Experimental gamma interval | `(0, 0.15]` |
| Theoretical gamma interval | `(0, 1]` |
| Linear input increment weight | `0.5` |
| Yaw input increment weight | `1e-5` |
| Velocity regularization weight | `0.1` |
| Rotation regularization weight | `5e-5` |

## Locked software

- CasADi `3.7.2`
- do-mpc `5.1.1`
- Safe Panda Gym compatibility fork commit `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c`
- Upstream compatibility pull request: `https://github.com/tohsin/Safe-panda-gym/pull/1`

These versions document this reproduction environment; the paper itself does not report exact package versions.

## Verified environment status

- Installation with CasADi 3.7.2, do-mpc 5.1.1, Gymnasium 1.3.0, PyBullet 3.2.7, and the locked Safe Panda compatibility commit succeeds on Python 3.12.
- The full project suite passes with the real CasADi/do-mpc/IPOPT stack.
- All 20 safety-aware environment variants (five tasks, sparse/dense rewards, and end-effector/joint control) reset and execute one headless Gymnasium step successfully.
- The compatibility fork removes stale `gym`/`gym_robotics` runtime imports, restores five safe environment exports, fixes the accidental `StackSafeStack3` registration, and preserves the safety cost in the Gymnasium `info` mapping.
- These smoke tests verify API execution, not equivalence to the unpublished scene, obstacle dynamics, prompts, or controller used for the paper figures.
- `PandaBuildL-v3` restores the four-cube, translucent-target scene shown in the fork's `build-l.png`; reset, step, and 720x480 headless rendering are verified on Gymnasium.

## Explicit assumptions

- State order is `[x, y, z, psi, dx, dy, dz, dpsi]` and control order is `[ux, uy, uz, upsi]`.
- The Safe Panda fork does not provide a paper-specific environment ID or a stable observation schema for this experiment. The adapter therefore accepts an environment ID and extraction/mapping callbacks.
- Obstacle radius, gripper collision radius, initial poses, targets, prompt templates, and obstacle trajectory must be supplied by an experiment configuration.
- The solver acceptance layer is fail-closed: missing diagnostics, infeasibility, excessive constraint residual, invalid action shape, or non-finite action values prevent application of the candidate control.
- The paper input bound `[-0.2, 0.2]` is mapped linearly to Safe Panda's normalized Cartesian action interval `[-1, 1]`; the paper does not publish its simulator action adapter.

## Current validation boundary

- The Safe Panda adapter, controller, symbolic CBF, solver policy, and closed-loop
  benchmark runners are implemented and covered by the project test suite.
- The 8-D double-integrator benchmark and the separate 3-D extension have saved
  aggregate results under their versioned setup manifests.
- The gamma sweep is a single deterministic episode used as an illustration;
  it is not a population-level ranking of decay values.
- The 3-D provider benchmark uses NVIDIA NIM with
  `meta/llama-3.1-8b-instruct` as an explicitly disclosed model substitution.
  It does not validate the paper's GPT-4o/OpenAI claim.
- The implementation remains an engineering reproduction because the paper does
  not publish its exact scene geometry, prompts, reference generator, action
  adapter, or raw controller traces.

Run `python -m pytest -q` after installation to verify the current software
contract. Provider collection and generated animations are optional and are not
required for the local test suite.
