# Proactive CBF and online-feedback protocol v4

Version 4 aligns the primary efficacy contrast with the paper controller and
adds an explicit formal-scope audit. It uses a new artifact namespace and
refuses to resume older checkpoints. The 220-step confirmatory completion
budget remains frozen.

This protocol addresses two contradictions observed in the first 500-condition
benchmark: smaller static gamma did not improve safety in aggregate, and delayed
online feedback reduced rather than increased success.

## Runtime safety contract

Every do-mpc solve now records IPOPT termination, solver success, iteration
count, wall time, and the maximum nonlinear-constraint bound violation computed
from the returned `opt_g_num`. A candidate reaches the robot only when its
termination and measured feasibility pass the configured policy. Invalid or
infeasible candidates are replaced by a zero nominal Cartesian command and
routed through the local operational-space gatekeeper. Deadline misses are
reported separately; `reject_deadline_miss` controls whether they also reject
the candidate.

The episode record uses `goal`, `collision`, `timeout`,
`environment_truncated`, or `emergency_fallback` as mutually exclusive outcomes,
while retaining process counters for causal analysis. If an episode
ends without another terminal event after a rejected MPC solve, its outcome is
`emergency_fallback`. Counts for solver failures, solver rejections, deadline
misses, and fallbacks remain available independently.

Model-to-simulator calibration records two one-step Cartesian residuals:

- model transition: observed position versus `p[k] + dt * v[k]`;
- action tracking: observed position versus `p[k] + dt * u[k]`.

These expose the velocity/displacement mismatch instead of attributing it to
gamma.

## Fair experiment profiles

`paper_fidelity` uses the published-like direct target reference, delta-u weight
0.5, static obstacle within the horizon, and no reflex or optimal-decay
extension. It contains the distance baseline and fixed gamma 0.15 versus
proactive gamma 0.02 comparison. It now also contains
`paper_async_feedback_static`, which is identical to fixed gamma 0.15 until the
randomly scheduled language update becomes available after measured LLM
latency. This elapsed-time update mirrors the paper experiment; it does not use
the TTC trigger or provisional local profile.

`robust_extension` is labeled separately and uses the straight moving reference,
delta-u weight 2.0, and explicit extensions. The prediction ablation is:

1. static obstacle center;
2. constant-velocity moving center;
3. constant-velocity center plus uncertainty tube;
4. uncertainty tube plus local reflex;
5. uncertainty tube plus optimal-decay CBF;
6. complete fixed and online-feedback stacks.

No result from the robust profile should be described as a strict reproduction
of the paper controller.

## Context scheduling and feedback

The local scheduler maps predicted TTC and bounded language safety level to a
profile containing gamma, clearance margin, and reference-speed scale. TTC has
priority over language preference. Solver infeasibility never requests a
smaller gamma; it switches to the gatekeeper.

The paper-fidelity online method always uses the preregistered random elapsed-time
intervention. The robust-extension online method defaults to a TTC trigger. At
threshold crossing it applies a deterministic provisional cautious profile
immediately. The async gamma result is applied after its measured latency and
clears the provisional profile. Each row records trigger time, availability
time, whether the update had causal opportunity
(`TTC > latency + reaction margin`), and whether an update was applied.

The old elapsed-time schedule remains available for backward-compatible
ablation through `feedback_schedule_mode="elapsed_time"`.

## Staged execution

Use the project virtual environment and keep stages in separate output folders:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_paired_benchmark.py --stage smoke --workers 1
PYTHONPATH=src .venv/bin/python scripts/run_paired_benchmark.py --stage development --workers 4
PYTHONPATH=src .venv/bin/python scripts/run_paired_benchmark.py --stage confirmatory --workers 4
```

The preregistered sizes are 12, 100, and 500 common-random-number conditions.
Do not tune thresholds on the confirmatory stage. The runner deliberately does
not retry a failed worker automatically; its CSV checkpoint supports explicit
resume.

## Confirmatory efficacy gate

The primary endpoint follows the paper's joint definition:

```text
joint_success = outcome == goal AND reached_goal AND NOT collision
```

Every other terminal outcome is a failure, including `timeout`,
`safety_timeout`, `controller_stall`, `solver_failure`, `emergency_fallback`,
and `environment_truncated`. The preregistered primary contrast is
`paper_async_feedback_static` against `fixed_cbf_static_g015` on identical
episode IDs, seeds, obstacle speeds, offsets, and intervention conditions. This
is the only contrast used by the confirmatory efficacy gate. The robust-stack
online-versus-fixed contrast remains a labeled secondary analysis and cannot
substitute for the paper claim.

The confirmatory efficacy gate passes only when all three checks hold:

1. the paired joint-success difference is greater than zero;
2. the lower bound of its paired bootstrap 95% interval is greater than zero;
3. the two-sided exact McNemar p-value is at most 0.05.

The paper's reported `+34` percentage-point effect is recorded as a reference
effect size, not silently substituted for the superiority margin. Smoke and
development stages report the same diagnostics but do not apply this gate.
Confirmatory runs use the frozen physics-derived 220-step budget by default and
exit with status 2 if the efficacy gate fails. An explicit `--max-steps`
override is recorded in the run configuration and must not be used after
confirmatory data have been inspected.

Safety metrics always use raw simulated trajectories. Visual smoothing remains
presentation-only.

## Formal-scope audit

Following the reference MPC-CBF implementations, every CBF episode now records
the true-state barrier and the applied one-step residual

```text
h_true[k+1] - omega[k] * (1 - gamma[k]) * h_true[k]
```

where `omega=1` for fixed decay. A nonnegative sampled residual is necessary
evidence for the discrete CBF condition, but it is not by itself a formal proof.
The report marks a stepwise certificate eligible only if all of these checks
hold for the full episode:

1. the initial true state is in the safe set;
2. every raw true-state CBF residual passes tolerance;
3. observation error is zero or deterministically bounded;
4. the executed input is the accepted MPC input, without reflex or fallback;
5. model transition and action tracking match the certified model tolerance.

Recursive-certificate eligibility additionally requires a certified terminal
safe set or an invariant backup controller. The current Panda controller has
neither, so protocol v4 always reports recursive eligibility as false. The
Gaussian sensor noise is unbounded, and the operational-space reflex does not
certify whole-body Panda collision avoidance. Consequently, the 500-episode
benchmark remains empirical evidence even when it has zero collisions.

The implementation choices were checked against the primary repositories
[HybridRobotics/MPC-CBF](https://github.com/HybridRobotics/MPC-CBF),
[tkkim-robot/safe_control](https://github.com/tkkim-robot/safe_control),
[learnsyslab/safe-control-gym](https://github.com/learnsyslab/safe-control-gym),
and [CBFKit](https://github.com/bardhh/cbfkit). Their controller code motivates
the separation between a finite-horizon CBF constraint, feasibility-enhancing
optimal decay, disturbance-aware filters, and terminal/backup invariance; none
of those properties is inherited merely by reusing a class name or equation.
