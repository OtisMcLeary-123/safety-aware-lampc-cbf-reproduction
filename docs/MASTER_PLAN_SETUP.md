# MPC-CBF Master Plan Setup

## Material Passport

- Origin: `academic-research-suite / experiment-agent`
- Verification status: `VERIFIED` for the recorded Safe Panda 8-state replay
- Final profile: `safe_panda_8d_double_integrator_scenario_extension`
- Evidence: 50 paired scenarios, 100 simulator rows
- Scope: simulator-only EE reach; not a physical Panda certificate or exact Table-4 reproduction

## Current Setup Matrix

| Layer | Final test setting | Implementation / source | Verification status and gap |
|---|---|---|---|
| Environment | `PandaReachSafe-v3`, PyBullet, Tiny renderer | `smooth_dynamic_demo.py`, `safe_panda_8d_double_integrator_50_setup.json` | Smoke and 100 replay rows pass; whole-arm collision is not certified |
| Observation | 8-state `[x,y,z,psi,dx,dy,dz,dpsi]` | `controller.py`, measured-state path in `smooth_dynamic_demo.py` | `psi` and `dpsi` are held at zero by the Cartesian Safe Panda adapter |
| Input | 4 inputs `[u_x,u_y,u_z,u_psi]`; plant exposes 3-D action | `paper_control_to_safe_panda_action()` | Yaw input is not actuated by this environment adapter |
| Discrete model | `p_next=p+dt*v+0.5*dt^2*u`; `v_next=v+dt*u` | `double_integrator_dynamics_matrices()` and `discrete_state_transition()` | Canonical A/B shared by MPC, CBF, and stored plant velocity |
| MPC | do-mpc, `dt=0.04 s`, horizon `15`, receding horizon | `build_mpc_controller()` | Full test suite passes; feasibility is scenario-dependent |
| CasADi CBF | `h=||p-p_obs||^2-r^2`; `h_next >= (1-gamma)h_current` | `ReferenceObstacleTVP.configure()` | Numeric cross-module successor test passes |
| IPOPT | CasADi IPOPT plugin, `tol=1e-8`, acceptable/constraint `1e-6`, CPU limit `0.035 s` | `solver.py`, `IpoptConfig` | Solver failures are counted and fail-closed; paper options are unpublished |
| Sensing | `0.67 s` zero-order hold, Gaussian noise | `SmoothDynamicConfig`, `UncertaintyTubeConfig` | Matches reconstructed baseline, not a published full scene specification |
| Obstacle | Point obstacle extension, radius `0`, EE collision radius `0.035 m` | setup JSON and scenario suite | Custom scenario geometry; not an exact paper obstacle reconstruction |
| Action adapter | average velocity `v+0.5*dt*u`, normalized by `0.05 m` displacement scale | `demo.py`, `smooth_dynamic_demo.py` | Max model/action error `0.000682 m` in the final replay |
| LLM feedback | NVIDIA NIM `meta/llama-3.1-8b-instruct`, one uncached request/episode | provider checkpoint kept local | Model substitution for GPT-4o; mean latency `0.393 s`, range `0.281-0.898 s` |
| Replay | fixed `gamma=0.15` vs contextual feedback (`0.03` or `0.07`) | `run_safe_panda_8d_benchmark.py` | 50 paired conditions; no provider call during replay |

## Final Replay Statistics

| Method | Goal successes | Collisions | Solver-failure steps | Episodes with solver failures | True CBF violation steps |
|---|---:|---:|---:|---:|---:|
| Fixed `gamma=0.15` | 13/50 | 37/50 | 63 | 9/50 | 460 |
| Contextual feedback | 13/50 | 37/50 | 240 | 28/50 | 1,109 |

Paired success difference is `0.0`; exact McNemar `p=1.0`. Feedback did not
improve the outcome and increased solver rejection/fallback activity. The
minimum true CBF residual was `-0.000797` (fixed) versus `-0.003841`
(feedback). These are descriptive simulator results, not a causal claim about
LLMs in general.

## Master Plan

1. **Freeze the contract.** Keep the 8-state order, A/B successor, goal tolerance, obstacle geometry, sensor timing, and action scale in one versioned setup JSON.
2. **Separate feasibility from efficacy.** Run a deterministic no-provider feasibility grid first; report IPOPT status, rejection count, deadline misses, transition error, and CBF residual before any LLM spend.
3. **Resolve solver failures.** Compare soft horizon CBF slack with exact penalty, hard one-step D-GCBF, and braking fallback as separate profiles. Do not silently change the frozen baseline.
4. **Validate geometry and goal semantics.** Add explicit goal speed/hold requirements and replace the point obstacle with a documented geometry only when the environment adapter is verified.
5. **Run paired feedback experiments.** Collect uncached provider records into a local checkpoint, validate model/provider/hash/latency integrity, then replay fixed and feedback conditions with identical seeds.
6. **Apply promotion gates.** Require zero unexplained fallback, bounded transition error, recorded solver diagnostics, and a predeclared efficacy metric before treating a profile as a candidate paper result.
7. **Claim audit.** Keep custom Safe Panda evidence separate from paper claims; any GPT-4o/OpenAI, red-cube pick/place, or whole-arm safety statement remains unverified until its missing setup is published or independently reconstructed.

## Reproduction Commands

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_8d_benchmark.py \
  --feedback-checkpoint \
  artifacts/safe_panda_8d_double_integrator_50_contextual_nim_llama31/feedback_decisions_checkpoint.json
```

The provider checkpoint is intentionally not committed. The committed result
files are the aggregate summary and paired CSV under
`artifacts/safe_panda_8d_double_integrator_50_benchmark/`.
