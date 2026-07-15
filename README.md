# Safety-Aware LaMPC-CBF Reproduction

Clean-room simulator reproduction and stress test of language-guided MPC-CBF
manipulation using Safe Panda Gym, do-mpc, CasADi, and IPOPT.

## Final Recorded Profile

The current committed evidence is the Safe Panda 8-state double-integrator
extension:

| Item | Setting |
|---|---|
| Environment | `PandaReachSafe-v3` with PyBullet/Tiny renderer |
| State | `[x,y,z,psi,dx,dy,dz,dpsi]` |
| Inputs | `[u_x,u_y,u_z,u_psi]`; Cartesian plant action is 3-D |
| Dynamics | `p_next=p+dt*v+0.5*dt^2*u`, `v_next=v+dt*u` |
| MPC | do-mpc, `dt=0.04 s`, horizon `15` |
| CBF | spherical EE barrier with discrete decay residual |
| Solver | CasADi IPOPT, fail-closed diagnostics |
| Sensing | `0.67 s` hold, Gaussian position noise `sigma=0.005 m` |
| Benchmark | 50 paired fixed-vs-feedback conditions |

Full setup and master-plan table: [docs/MASTER_PLAN_SETUP.md](docs/MASTER_PLAN_SETUP.md).

## Final Result

| Method | Successes | Collisions | Solver-failure steps |
|---|---:|---:|---:|
| Fixed `gamma=0.15` | 13/50 | 37/50 | 63 |
| Contextual feedback | 13/50 | 37/50 | 240 |

Paired success difference: `0.0`; exact McNemar `p=1.0`. The feedback branch
does not improve this custom simulator benchmark and increases solver fallback
activity. This is not an exact Table-4 reproduction, a physical-robot result,
or a whole-arm safety certificate.

## Result Files

- [benchmark summary](artifacts/safe_panda_8d_double_integrator_50_benchmark/benchmark_summary.json)
- [paired episode table](artifacts/safe_panda_8d_double_integrator_50_benchmark/episodes.csv)
- [setup manifest](configs/safe_panda_8d_double_integrator_50_setup.json)
- [master plan and setup matrix](docs/MASTER_PLAN_SETUP.md)

The provider checkpoint contains raw request metadata and is kept local rather
than committed. Never commit `nvidiatoken.txt`, virtual environments, or other
credentials.

## Install and Test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,simulation]'
PYTHONPATH=src pytest -q
```

The final validation run passed `278` tests with `2` optional skips.

## Replay Without Provider Calls

```bash
PYTHONPATH=src python scripts/run_safe_panda_8d_benchmark.py \
  --feedback-checkpoint \
  artifacts/safe_panda_8d_double_integrator_50_contextual_nim_llama31/feedback_decisions_checkpoint.json
```

Provider collection is a separate, explicitly authorized operation:

```bash
PYTHONPATH=src python scripts/collect_safe_panda_8d_feedback.py
```
