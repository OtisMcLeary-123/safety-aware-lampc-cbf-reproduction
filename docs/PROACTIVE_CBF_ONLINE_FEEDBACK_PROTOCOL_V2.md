# Proactive CBF and online-feedback protocol v2

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
proactive gamma 0.02 comparison.

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

The online method defaults to a TTC trigger rather than a random elapsed-time
trigger. At threshold crossing it applies a deterministic provisional cautious
profile immediately. The async gamma result is applied after its measured
latency and clears the provisional profile. Each row records trigger time,
availability time, whether the update had causal opportunity
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

Safety metrics always use raw simulated trajectories. Visual smoothing remains
presentation-only. Monte Carlo results do not establish formal whole-body
Panda safety under unbounded Gaussian noise.
