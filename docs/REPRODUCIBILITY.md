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
- Safe Panda Gym commit `f6cf3031f489f2f11c9e9274f3a17dda5670a649`

These versions document this reproduction environment; the paper itself does not report exact package versions.

## Verified environment status

- Editable installation with CasADi 3.7.2, do-mpc 5.1.1, Gymnasium 1.3.0, PyBullet 3.2.7, and the locked Safe Panda commit succeeds on Python 3.12.
- The full project suite passes with the real CasADi/do-mpc/IPOPT stack.
- A headless reset of `PandaReachSafe-v3` and `PandaPickAndPlaceSafe-v3` fails inside the cited fork before this project's adapter is entered. The fork's `panda_gym/envs/__init__.py` imports `PandaStack3Env`, but the merged `panda_tasks.py` does not define that class. Its safe core also retains imports of legacy `gym` and `gym_robotics` despite the package declaring Gymnasium.
- Revision `6ba05d59b65bbf1dafc2cbec5060eb77fa2ad852`, before the upstream Gymnasium merge, contains the safe environment classes but declares Gym 0.22-0.23 and Python 3.7-3.9 era dependencies. It is not directly compatible with the current Python 3.12 environment.

The simulator smoke gate therefore remains open. A faithful next step is either a Python 3.9 legacy container using the pre-merge revision or a narrowly documented compatibility fork. The installed third-party package is not patched in place because such a change would be irreproducible.

## Explicit assumptions

- State order is `[x, y, z, psi, dx, dy, dz, dpsi]` and control order is `[ux, uy, uz, upsi]`.
- The Safe Panda fork does not provide a paper-specific environment ID or a stable observation schema for this experiment. The adapter therefore accepts an environment ID and extraction/mapping callbacks.
- Obstacle radius, gripper collision radius, initial poses, targets, prompt templates, and obstacle trajectory must be supplied by an experiment configuration.
- The solver acceptance layer is fail-closed: missing diagnostics, infeasibility, excessive constraint residual, invalid action shape, or non-finite action values prevent application of the candidate control.

## Remaining verification gates

1. Resolve the cited Safe Panda fork's legacy/merged API incompatibility in a pinned Python 3.9 container or a documented compatibility fork.
2. Select and validate the concrete Safe Panda task/environment schema.
3. Connect the CBF constraint builder to the do-mpc controller for the selected observation schema.
4. Run a headless deterministic collision-avoidance smoke episode.
5. Recreate the gamma sweep and compare minimum obstacle clearance qualitatively with Figure 5.
6. Run 50-episode feedback/no-feedback ablations only after intervention timing and LLM latency policies are specified.

Until gates 1-4 pass, the project is an implementation scaffold, not a validated reproduction of the paper's results.
