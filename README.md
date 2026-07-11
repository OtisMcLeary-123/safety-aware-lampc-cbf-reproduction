# Safety-Aware LaMPC-CBF Reproduction

Reproduction scaffold for *Safety-Aware Optimal Control With Language-Guided Online Parameter Adjustment via Large Language Models* (IEEE Access, 2026, DOI: 10.1109/ACCESS.2026.3664145).

The implementation follows the paper's reported stack:

- Python
- do-mpc
- CasADi
- IPOPT
- Safe Panda Gym

## Status

Work in progress. The mathematical components and simulator adapter are implemented and covered by unit tests; a paper-scale closed-loop experiment is not yet validated.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,simulation]'
python -m pytest -q
```

The simulation extra pins do-mpc and CasADi and installs the paper-cited Safe Panda Gym fork at a fixed commit. The controller/solver stack runs on Python 3.12, but the cited fork's safe environments currently have an upstream compatibility defect described in `docs/REPRODUCIBILITY.md`.

## Implemented components

- Paper-aligned 8-state, 4-input discrete dynamics and MPC configuration.
- CasADi expressions for the obstacle barrier, discrete CBF residual, and orientation/velocity regularizer.
- IPOPT options, diagnostics, and fail-closed control acceptance policy.
- Safe Panda Gym adapter with deterministic seeding, 0.67 s zero-order-held obstacle sensing, and 0.005 m Gaussian measurement noise.
- Typed configuration and a dry-run CLI that never sends prompts or credentials externally.

## Reproducibility boundary

The repository reproduces the mathematical controller and simulation setup from the published description. The authors' public project repository currently contains no implementation beyond its README, so this is a clean-room reproduction rather than a bit-for-bit rerun. Prompt text, unpublished assets, environment schema, object geometry, and other details not supplied by the paper are documented as explicit assumptions.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the evidence boundary and remaining validation work.
