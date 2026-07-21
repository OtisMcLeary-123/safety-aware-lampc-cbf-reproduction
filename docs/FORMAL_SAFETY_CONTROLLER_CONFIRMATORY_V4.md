# Formal-safety/controller review and confirmatory protocol v4

## Outcome

Protocol v4 now tests the paper controller as the primary confirmatory contrast
and audits the discrete CBF condition on the raw true simulator trajectory. The
12-condition smoke run did not pass the development go/no-go criterion, so the
100- and 500-condition stages were deliberately not started.

## Primary repositories reviewed

| Repository | Reusable idea | Boundary relevant to this project |
|---|---|---|
| [HybridRobotics/MPC-CBF](https://github.com/HybridRobotics/MPC-CBF) | Finite-horizon discrete CBF constraint `h[k+1]-(1-gamma)h[k]>=0`; explicit infeasible return; gamma/horizon ablations | The reference code uses the optimization model and a static obstacle. It does not make PyBullet model mismatch or unbounded sensor noise disappear. |
| [tkkim-robot/safe_control](https://github.com/tkkim-robot/safe_control) | MPC-CBF, optimal-decay MPC-CBF, dynamic-obstacle C3BF/DPCBF, and gatekeeper/backup safety filters | Optimal decay addresses feasibility. Infinite-horizon gatekeeper safety additionally depends on a valid backup policy and recoverable/invariant set. The current Panda reflex does not yet implement that contract. |
| [learnsyslab/safe-control-gym](https://github.com/learnsyslab/safe-control-gym) | Separate symbolic constraints, disturbances, controller performance, and safety-filter evaluation | A disturbance benchmark is empirical unless the disturbance/error set is deterministically bounded and appears in the certificate. |
| [CBFKit](https://github.com/bardhh/cbfkit) | Robust CBF-QP construction and dynamic obstacles represented in augmented state-time space | Useful for a future certified local filter, but its certificates are not automatically inherited by the current do-mpc/Panda implementation. |

The implementation was inspected at repository revisions
`3a4ea54` (MPC-CBF), `ac950ba` (safe_control), `6b5391d`
(safe-control-gym), and `4e92cb3` (CBFKit).

## Changes made

1. The confirmatory primary contrast is now
   `paper_async_feedback_static` versus `fixed_cbf_static_g015`.
   Both use the paper profile: direct target, `Delta-u` weight 0.5, static
   obstacle within the horizon, and no robust reflex or optimal-decay extension.
2. The paper online controller uses the randomized elapsed-time intervention
   plus measured LLM latency. TTC-triggered provisional feedback remains only in
   the labeled robust extension.
3. `robust_stack_async_feedback` versus `robust_stack_fixed_g015` is retained as
   a secondary analysis and cannot satisfy the paper-claim efficacy gate.
4. Every CBF rollout now records the raw true barrier and applied residual
   `h_true[k+1] - omega[k] * (1-gamma[k]) * h_true[k]`.
5. Stepwise formal eligibility requires initial safety, no raw residual
   violation, exact or deterministically bounded observation, unchanged
   accepted MPC input, and verified model/action matching.
6. Recursive formal eligibility additionally requires a certified terminal
   safe set or invariant backup controller. This is explicitly false for the
   current Panda stack.
7. The runner can consume a frozen, previously validated `GammaDecision` JSON,
   separating controller evaluation from provider availability.

## Smoke benchmark result

Run configuration: 12 common-random-number conditions, 140-step smoke budget,
11 methods, obstacle speed 0.025--0.20 m/s, Gaussian measurement noise 0.005 m,
and sensor ZOH 0.67 s. The frozen feedback decision was `gamma=0.02` from
DeepInfra `Qwen/Qwen3-235B-A22B-Instruct-2507`, with recorded latency 2.142 s.

| Method | Goal | Collision | Timeout/stall | Raw CBF violation steps | Stepwise / recursive eligible |
|---|---:|---:|---:|---:|---:|
| Fixed paper CBF, `gamma=0.15` | 0/12 | 12/12 | 0 | 96 | 0 / 0 |
| Proactive paper CBF, `gamma=0.02` | 0/12 | 12/12 | 0 | 786 | 0 / 0 |
| Paper asynchronous feedback | 0/12 | 12/12 | 0 | 175 | 0 / 0 |
| Predictive uncertainty-tube CBF | 0/12 | 0/12 | 12 | 155 | 0 / 0 |
| Predictive optimal decay | 0/12 | 0/12 | 12 | 89 | 0 / 0 |
| Robust fixed stack | 0/12 | 0/12 | 12 | **0** | 0 / 0 |
| Robust asynchronous stack | 0/12 | 0/12 | 12 | 7 | 0 / 0 |

The paper primary paired difference was 0.0 with both methods at 0% joint
success. Only 3/12 paper-feedback episodes received the delayed update before
termination. This is a causal timing failure in the present scene, not evidence
that language feedback is intrinsically ineffective. The robust primary pair
from protocol v3 also had a 0.0 paired difference at the 140-step smoke budget.

The formal audit prevents a second misleading conclusion. The robust fixed
stack had no sampled true CBF residual violation, but all episodes remained
formally ineligible because Gaussian observation error is unbounded, the
executed input was frequently modified by the reflex, model/action equivalence
was not verified to certificate tolerance, and no terminal invariant/backup set
exists.

The smoke sample is a pipeline/go-no-go test, not an inferential experiment.
Confidence intervals and p-values from 12 all-failure pairs are not used to
support an efficacy claim. The confirmatory gate is therefore reported as not
evaluated rather than failed statistically.

## Decision

Do not run development 100 or confirmatory 500 with this frozen protocol yet.
First, add a deterministic nominal certificate grid with exact state and exact
plant transition, then implement and test a terminal invariant set or certified
backup controller. Separately, the language intervention must be scheduled
early enough that `intervention time + measured latency` precedes the collision
cone entry; this timing rule must be frozen before development data are viewed.

## Material Passport

- Material type: controller instrumentation, benchmark protocol, unit tests,
  protocol documentation, and generated smoke artifacts.
- Software environment: Python 3.12 virtual environment; PandaReachSafe-v3,
  PyBullet Tiny renderer, do-mpc, CasADi, and IPOPT.
- Inputs: protocol-v4 seeded scene generator and a previously validated frozen
  `GammaDecision`; no source repository code was copied into the project.
- Generated evidence:
  `artifacts/paired_benchmark_protocol_v4_smoke_12/benchmark_summary.json`,
  `episodes.csv`, and `paired_success_and_clearance.png`.
- Verification: 235 unit/integration tests passed and 2 optional tests skipped;
  the 12-condition smoke run completed without worker retry.
- Evidence scope: simulation and sampled raw-trajectory audit only; no hardware,
  whole-body collision certificate, deterministic noise bound, terminal
  invariant set, or recursive safety proof.
- Provider incident: a live NVIDIA NIM call returned output rejected by the JSON
  validator (`ValueError`), before any episode ran. It was not retried; the
  completed smoke run used the frozen DeepInfra decision listed above.
