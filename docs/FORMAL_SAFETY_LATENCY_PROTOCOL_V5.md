# Formal-safety and asynchronous-latency protocol v5

## Outcome

Protocol v5 removes the single frozen provider latency from the benchmark. It
collects one validated, uncached language decision for every paired condition,
stores the full latency trace, and replays that condition's latency across its
controllers. A delayed language response never bypasses local safety: the user
request activates a deterministic provisional profile immediately, while the
returned gamma is accepted only before a TTL measured from request time.

## Paper one-shot interpretation

The paper's feedback ablation is treated as one language request per
feedback-enabled episode, not one provider call shared by the full experiment.
On page 25560, the authors report 50 independent episodes, state that the
feedback-enabled variant receives the singular message `Watch out! I think it's
going to crash soon`, and report an average OpenAI round-trip latency of 2.4 s
over 50 episodes. Together, these statements support 50 episode-specific
request/latency records for the feedback condition.

This is an ablation-specific inference, not a universal restriction on the
method. Algorithm 1 checks `user_intervened` inside every control cycle, so a
later control cycle could process another intervention. The paper does not
publish the intervention thread, request launch timing, retry policy, or raw
latency trace.

The reproduction therefore locks `feedback_requests_per_episode=1`, requires
50 uncached decision records with 50 distinct request timestamps, and rejects a
single frozen record repeated across episodes. It currently pre-collects each
uncached request and replays its measured latency at the episode's intervention
time. This preserves a per-episode latency trace but does not reproduce a live
request launched at the exact intervention wall-clock time.

The protocol also adds a labeled `formal_extension`; it does not silently
upgrade the paper reproduction. This profile closes four specific assumptions
for the spherical end-effector task:

| Previous gap | Protocol-v5 contract |
|---|---|
| Gaussian sensor error has no deterministic support | Sample noise from a 3-D bounded ball and use the same declared radius in MPC and the final safety filter. |
| The operational-space reflex modifies MPC output | Treat MPC/reflex as nominal; certify the final Cartesian command immediately before action mapping. |
| PyBullet and the MPC transition are not equivalent | Declare an additive one-step Cartesian transition-error bound and fail the formal gate when measured action tracking exceeds it. |
| No recursive/terminal certificate | Require the next state to remain in the recoverable set of a maximum-speed radial escape backup; fail closed when backup authority is negative. |

If the final filter cannot certify either the one-step robust residual or the
terminal backup condition, the simulator runner raises before calling
`env.step`; no uncertified Cartesian action is sent.

## Mathematical contract

Let the measured obstacle center be `q_hat`, with
`||q-q_hat|| <= epsilon`, and let the Cartesian transition satisfy
`p[k+1] = p[k] + dt*u[k] + w[k]`, `||w[k]|| <= delta`. Obstacle velocity and
acceleration errors are also bounded. The final filter accepts a command only
if the worst-case discrete residual is nonnegative:

```text
h_robust[k+1] - (1-gamma) h_robust[k] >= 0.
```

The current sphere is enlarged by `epsilon`; the next sphere is enlarged by
`epsilon + delta + dt*e_v + 0.5*dt^2*a_max`. The terminal backup policy is

```text
u_backup = u_max * (p-q_hat) / ||p-q_hat||.
```

Its declared robust authority margin is
`u_max - v_obstacle,max - e_v - delta/dt`. The sampled terminal contract is
eligible only when both obstacle clearance and this authority margin are
nonnegative at every applied step. This construction is deliberately scoped to
one spherical end-effector and one spherical obstacle. It does not certify
Panda links, joints, self-collision, contacts, or the truth of a chosen bound.

## Sources used

- [Robust CBFs for Sampled-Data Systems](https://arxiv.org/abs/2309.08050)
  motivates bounded disturbance/measurement sets and piecewise-constant input.
- [CBFs for Sampled-Data Systems with Input Delays](https://arxiv.org/abs/2005.06418)
  motivates explicit ZOH, state-uncertainty, and input-delay margins.
- [Linear Model Predictive Safety Certification](https://arxiv.org/abs/1803.08552)
  motivates a final minimally modifying filter with bounded additive model
  error and a safe terminal target.
- [tkkim-robot/safe_control](https://github.com/tkkim-robot/safe_control)
  provides concrete gatekeeper, MPS, Backup-CBF, optimal-decay, and dynamic-CBF
  reference implementations.
- [learnsyslab/safe-control-gym](https://github.com/learnsyslab/safe-control-gym)
  demonstrates MPSC, terminal safe sets, tube MPC, and repeatable disturbance
  injection in a PyBullet/CasADi benchmark.
- [HybridRobotics/MPC-CBF](https://github.com/HybridRobotics/MPC-CBF)
  remains the reference for finite-horizon discrete MPC-CBF formulation.
- [Backup-Based Safety Filters review](https://arxiv.org/abs/2604.02401)
  unifies Backup CBF, MPS, and gatekeeper around a backup policy and terminal
  controlled-invariant set.

## Implemented gate and current diagnostic

The confirmatory summary now exposes a separate `formal_contract_gate` for
`formal_stack_fixed_g015` and `formal_stack_async_feedback`. It requires zero
collision, zero uncertified final-filter steps, zero uncertified terminal steps,
and recursive eligibility in every episode. Passing is conditional on the
declared bounds and is never labeled a whole-body Panda proof.

A single 80-step integration diagnostic completed with zero collision, zero
sampled true-CBF violation, zero uncertified final-filter steps, and zero
uncertified terminal steps. Minimum true clearance was 0.12784 m, minimum robust
filter residual was 0.001421, maximum action-tracking error was 0.003406 m under
the declared 0.008 m bound, and the backup-authority margin was exactly 0.0 m/s.
The episode stalled and moved away from its goal, so this is safety-contract
evidence only, not an efficacy success. The zero authority slack is a reason to
keep the 100/500 gates closed until a deterministic scenario grid confirms the
bound with positive engineering margin.

A separate two-condition live-provider smoke collected two uncached Hugging
Face responses at 2.418 s and 2.434 s. Both formal controllers had zero
collision and every one of each controller's 80 applied steps was final-input and
terminal-backup eligible; their minimum robust residuals were 0.003365 and
0.004131. Because the deliberately short episode budget was 1.6 s, neither LLM
response arrived before termination. The immediate local provisional profile
therefore had 2/2 causal opportunities, while semantic LLM gamma had 0/2. All
four formal episodes stalled and goal success was 0/4. This confirms the
architecture and exposes the remaining latency/efficacy problem; it is not a
reason to spend 500 episodes.

## Staged decision rule

1. Run unit tests and deterministic boundary/near-boundary grids.
2. Collect one live, uncached latency record per feedback episode; do not
   substitute a single provider measurement for all conditions.
3. Run smoke and ablation only. Inspect causal local/LLM opportunities and the
   formal contract separately from goal completion.
4. Run development 100 only after formal, feasibility, timing, and efficacy
   prerequisites pass.
5. Run confirmatory 500 only after the protocol and bounds are frozen from the
   development result. Timeout remains a joint-success failure.

## Material Passport

- Material type: controller source, benchmark protocol, unit tests, literature
  mapping, and one non-persisted integration diagnostic.
- Software: Python 3.12 project virtual environment, PandaReachSafe-v3,
  PyBullet Tiny, do-mpc, CasADi, and IPOPT.
- Inputs: seeded simulated scene; no source code copied from reviewed projects.
- Deterministic assumptions: 0.005 m measurement-error ball, 0.008 m additive
  action-transition error, 0.20 m/s obstacle-speed bound, 0.40 m/s Cartesian
  command bound, and 0.04 s control period.
- Verification at implementation time: 241 tests passed, 2 optional tests
  skipped; one 80-step formal-profile diagnostic completed without retry.
- Evidence boundary: simulation, spherical end-effector geometry, and declared
  bounded errors only; no hardware or whole-body forward-invariance claim.
