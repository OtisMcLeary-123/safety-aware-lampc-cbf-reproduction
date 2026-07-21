# Safe Panda Core Scenarios — 150-Episode Mandatory Benchmark Result

- Date: 2026-07-17
- Plan: `configs/safe_panda_core_scenarios_150_plan.json` (v1, unchanged;
  sha256 `3f95325ef7a1b49c87c53d8f3928a199b11bae1bb8f7ed3f701ff620ea6d45f0`)
- Frozen instances: `configs/safe_panda_core_scenarios_150_instances.json`
  (sha256 `e0f154604660f79d6dfb26d83ed5b60955b8977f991365a228e739ff8fededdf`)
- Method: `safe_panda_core_fixed_g015` — frozen profile (gamma 0.15,
  `double_integrator`, `direct_target`, static horizon prediction, 0.67 s
  zero-order-hold sensing, EE radius 0.035 m, max 260 steps), zero provider
  requests.
- Raw outputs: `artifacts/safe_panda_core_scenarios_150/` (ignored;
  `episodes.csv`, `benchmark_summary.json`, per-row `run_checkpoint.json`).

**Scope limits:** simulator-only, end-effector-reach-only, no whole-arm
certificate, Gaussian (unbounded) measurement noise — no bounded-error
safety claim. Descriptive results for this repository's frozen profile
only; no original-paper claim is verified by this benchmark.

## Stage gates

- Stage 0: module/scripts/tests implemented; full suite 308 passed.
- Preflight: 150/150 instances passed all gates; max rejection attempts
  132/1000; min initial true clearance 0.109 m; exact 25/25 side balance
  in CS2 and CS3.
- Stage 1 smoke (CS1-E30, CS2-E19, CS3-E19): 3 completed rows, finite
  diagnostics, no crash. All three collided.
- Stage 2 pilot (15 episodes): 15 completed rows, no crash, no deadline
  miss; 3 goals / 12 collisions — conditions neither trivial nor
  universally impossible; no range change, plan stays v1.
- Stage 3: exactly 150 rows, checkpoint hash matches, 0 crashed episodes,
  0 deadline misses.

## Primary endpoint: collision-free goal success

| Family | Success | Wilson 95% | Collisions | Wilson 95% |
|---|---|---|---|---|
| CS1 head-on closing | 0/50 | [0.000, 0.071] | 50/50 | [0.929, 1.000] |
| CS2 orthogonal 3-D crossing | 10/50 | [0.112, 0.330] | 40/50 | [0.670, 0.888] |
| CS3 grazing near-limit | 14/50 | [0.175, 0.417] | 36/50 | [0.583, 0.825] |

Equal-weight macro success 0.160; worst family 0.000 (CS1).

## Secondary endpoints

| Family | Min true clearance mean [bootstrap 95%] | Solver-failure steps (episodes) | Deadline misses |
|---|---|---|---|
| CS1 | −0.0038 m [−0.0046, −0.0030] | 16 (11) | 0 |
| CS2 | +0.0015 m [−0.0007, +0.0042] | 79 (10) | 0 |
| CS3 | +0.0025 m [+0.0001, +0.0052] | 9 (4) | 0 |

All solver failures produced fail-closed fallbacks (equal counts); no
emergency-path anomalies, no crashed rows.

## Reading (descriptive only)

- H1 (family differences): supported descriptively — success spans 0% to
  28% across families with non-overlapping Wilson intervals between CS1
  and CS3.
- H2 (grazing hardest on the solver): **not supported** — CS2, not CS3,
  concentrates solver failures (79 steps in 10 episodes); CS3 has the
  fewest (9). CS1's difficulty is not solver infeasibility but sensing:
  head-on closings at 0.05–0.2 m/s cross the 0.67 s sensing dead time
  faster than the static-horizon prediction can react, collapsing success
  to 0/50 with only 16 failure steps.
- Consistency: macro 16% is in line with the frozen 8-D baseline (13/50
  goals, 37/50 collisions on its own 50-scenario set).
- The dominant failure mode is inter-sample collision under zero-order-hold
  sensing with static horizon prediction — the same mechanism isolated in
  the Stage 7/8 paper-fidelity smokes. Remedies (velocity-tube prediction,
  soft-slack CBF) are separate versioned profiles for the optional paired
  phase (Stage 4); they must replay these exact 150 instances.

## Paired phase: scripted-feedback arm (2026-07-17, same day)

Profile `configs/safe_panda_core_scripted_feedback_v1.json` — deterministic
hard-coded feedback ("Move cautiously." → gamma 0.07 at the frozen
`feedback_intervention_time_s`; "Watch out!…" → gamma 0.03 at +1.7 s;
latency 0; zero provider requests) replayed on the identical 150 frozen
instances. Outputs under
`artifacts/safe_panda_core_scenarios_150_scripted_feedback/`.

Result: **outcome-identical to the fixed arm in all 150 episodes** — zero
discordant pairs in every family, McNemar exact p = 1.0 (Holm-adjusted
1.0) across CS1/CS2/CS3; success difference exactly 0.00. The gamma
updates demonstrably applied: solver-failure steps rose from 104 (fixed)
to 669 (scripted; CS1 16→288, CS2 79→235, CS3 9→146) and minimum-clearance
distributions shifted slightly — tightening gamma to 0.03 sharply degrades
feasibility without changing a single episode outcome.

Reading (descriptive): under static horizon prediction + 0.67 s
zero-order-hold sensing, collisions are decided by the sensing dead time,
not by the CBF decay rate; language-triggered gamma tightening therefore
cannot rescue these episodes, only pay more infeasibility. This replicates
the frozen 8-D result signature (no paired difference, p = 1.0; failure
steps 63 → 240) on an independent, three-family, 150-episode set.

## Paired phase: dead-time margin arm (ZOCBF-style, 2026-07-17)

Profile `configs/safe_panda_core_deadtime_margin_v1.json` — static
prediction with the CBF radius inflated by `0.21 m/s x (measurement age +
stage offset)` (arXiv:2005.06418, 2411.17079 rationale). Identical 150
frozen instances; outputs under
`artifacts/safe_panda_core_scenarios_150_deadtime_margin/`.

| Family | Success | Collisions | Min true clearance (mean) | Solver-failure steps |
|---|---|---|---|---|
| CS1 | 0/50 (base 0) | **19 (base 50)** | +0.051 m (base −0.004) | 1624 (base 16) |
| CS2 | 4/50 (base 10) | **1 (base 40)** | +0.074 m (base +0.002) | 2091 (base 79) |
| CS3 | 1/50 (base 14) | **0 (base 36)** | +0.066 m (base +0.003) | 933 (base 9) |

Outcome mix: 116 safety_timeout, 20 collision, 9 controller_stall, 5 goal.
Paired primary endpoint: CS1 diff 0.00 (p=1.0), CS2 −0.12 (p=0.146, Holm
0.292), CS3 **−0.26 (p=2.4e-4, Holm 7.3e-4 — significantly worse)**;
macro −0.127.

Reading: the margin does exactly what the sampled-data theory promises —
collisions collapse 126 → 20 and clearance turns robustly positive — but
the worst-case ball (up to ~0.27 m inflation at the horizon end) blocks
the 0.3 m-scale task almost permanently: failures convert from collision
to timeout/stall and solver-failure steps grow 45x. On the primary
endpoint this arm is *worse* than baseline (significantly in CS3).
Safety-liveness trade quantified; the ranked next candidate is the
velocity-tube arm, whose velocity-informed tube is far tighter than the
worst-case ball.

## Paired phase: velocity-tube arm (2026-07-17)

Profile `configs/safe_panda_core_velocity_tube_v1.json` — constant-velocity
obstacle propagation through the horizon with the uncertainty tube
(`prediction_mode="velocity_tube"`, all tube parameters at defaults,
velocity estimated from the ZOH measurements). Identical 150 frozen
instances; outputs under
`artifacts/safe_panda_core_scenarios_150_velocity_tube/`.

| Family | Success | vs base | McNemar p | Holm p | Collisions |
|---|---|---|---|---|---|
| CS1 | 1/50 | +0.02 (0→1) | 1.0 | 1.0 | 20 (base 50) |
| CS2 | 40/50 | **+0.60 (10→40)** | 1.9e-9 | **5.6e-9** | 3 (base 40) |
| CS3 | 28/50 | **+0.28 (14→28)** | 4.3e-3 | **8.7e-3** | 1 (base 36) |

Macro success 0.460 (base 0.160, diff +0.300); outcome mix 69 goal /
55 safety_timeout / 24 collision / 2 stall; solver-failure steps 3740
(base 104); zero crashes. Discordant pairs are one-sided in CS2 (30
method-only wins, 0 baseline-only) and strongly favorable in CS3 (18 vs
4).

Four-arm summary on the identical instances: fixed 24/150 (126
collisions, 104 failure steps); scripted feedback 24/150 (126, 669);
dead-time margin 5/150 (20, 4648); velocity tube **69/150 (24, 3740)**.
Velocity-informed horizon propagation is the only arm that improves the
primary endpoint, and the improvement is statistically significant in
CS2/CS3 after Holm correction. CS1 head-on remains essentially unsolved
by prediction alone (1/50; obstacle rides the goal axis, so the safe set
along the direct route stays empty for most of the episode) — a
liveness-aware reference (lateral evasion) is the open lead there.
Descriptive simulator results for this repository's profiles only.

## Paired phase: soft-slack arm (2026-07-17)

Profile `configs/safe_panda_core_soft_slack_v1.json` — velocity-tube base
plus per-stage CBF slack input with L1 exact penalty (weight 1000, slack
in [0, 10]; mandated remedy profile 1). The barrier itself is untouched.
Identical 150 frozen instances; outputs under
`artifacts/safe_panda_core_scenarios_150_soft_slack/`.

| Family | Success | Collisions | Solver-failure steps | vs static base (Holm) |
|---|---|---|---|---|
| CS1 | 4/50 | **0** | 395 | +0.08 (p=0.250, ns) |
| CS2 | 38/50 | **0** | **0** | +0.56 (p=2.2e-8, sig) |
| CS3 | 21/50 | **0** | 196 | +0.14 (p=0.250, ns) |

Macro success 0.420; outcome mix 63 goal / 83 safety_timeout / 4 stall /
**0 collision** — the first arm with zero collisions in all 150 episodes,
and the first with any CS1 successes. Solver-failure steps 591 (velocity
tube alone: 3740; CS2 became fully feasible).

Exploratory slack-vs-tube pairing (not predeclared, labeled as such):
CS2 38 vs 40 (p=0.50), CS1 4 vs 1 (p=0.25), CS3 21 vs 28 — slack loses 7
CS3 episodes to timeouts (p=0.016, Holm 0.047). Net: the slack valve
converts all residual collisions and most infeasibility into
conservative timeouts, at a modest CS3 liveness cost. Safety-first
profile of choice; tube remains the success-first profile.

## Paired phase: NIM LLM feedback arm (2026-07-17, provider phase)

Predeclared manifest `configs/safe_panda_core_nim_feedback_v1.json`:
velocity-tube base; at each episode's frozen feedback time the utterance
"Watch out! I think it's going to crash soon." goes to NVIDIA NIM
`meta/llama-3.1-8b-instruct` (GPT-4o substitute) with the Table-1
calibrated continuous prompt. The model returned **gamma 0.05** (Table-2
label 1), latency 0.276 s; temperature 0 + identical prompts → exactly
**1 real API request**, 149 checkpoint cache hits, 0 fallbacks.

| Family | Success | Collisions | Solver-failure steps | vs tube (Holm) |
|---|---|---|---|---|
| CS1 | 0/50 | 42 | 1755 | −0.02 (p=1.0) |
| CS2 | 28/50 | 11 | 1676 | **−0.24 (raw p=4.9e-4, Holm 1.5e-3, sig worse)** |
| CS3 | 20/50 | 24 | 1898 | −0.16 (p=0.231) |

Macro 0.320; outcomes 48 goal / 77 collision / 24 timeout / 1 stall.

**Key finding:** LLM-triggered gamma tightening (0.15 → 0.05) on the
hard-constraint tube base is **harmful**: infeasible steps grow 3740 →
5329, each rejected solve fails closed to a zero command, and freezing
at close range drives collisions 24 → 77. Episode-level attribution
(claim-verify pass, 2026-07-19): failure share is 68.6% of steps in
collision episodes vs 15.7% in goal episodes, and **all 77 collision
episodes contain substantial rejected-solve streaks (0/77 with ≤2
failure steps)** — no collision occurs without freezing. Of the 56
collisions new relative to the tube arm, 34 were tube timeouts, **21
were previously successful tube episodes**, and 1 a stall — the freeze
does not merely convert timeouts; it also destroys former successes.
The intended safety effect inverts through the infeasibility-freeze
mechanism. (Against the static fixed baseline the
arm still helps: 48 vs 24 successes — the tube base, not the feedback,
carries that gain.) The natural follow-up arm is NIM feedback on the
**soft-slack** base, where the L1 valve removes the freeze mechanism.
Descriptive; single deterministic gamma decision (temperature 0), not
per-episode adaptation.

## Paired phase: NIM + soft-slack arm (2026-07-17, causal closure)

Predeclared manifest `configs/safe_panda_core_nim_soft_slack_feedback_v1.json`
— identical NIM feedback protocol (gamma 0.05 replayed from checkpoint,
**zero new API requests**), on the soft-slack base. Hypothesis
(predeclared in the manifest before the run): the harm in the NIM/tube
arm is caused by the infeasibility-freeze mechanism; the slack valve
removes it, so the same tightening should be neutral-or-better here.

| Family | Success | Collisions | Solver-failure steps | vs slack fixed (Holm) |
|---|---|---|---|---|
| CS1 | 10/50 | 0 | 1 | +0.12 (p=0.070, Holm 0.211; 7 wins vs 1) |
| CS2 | 36/50 | 0 | 0 | −0.04 (p=0.500) |
| CS3 | 20/50 | 0 | 0 | −0.02 (p=1.0) |

Macro 0.440; outcomes 66 goal / 84 safety_timeout / **0 collision /
0 stall**; solver-failure steps **1** (slack fixed: 591); best minimum
clearances of any arm (+0.074..+0.096 m).

**Hypothesis confirmed.** The identical feedback signal on identical
instances: hard base → −21 successes vs its fixed counterpart and
collisions 24 → 77; slack base → +3 successes vs its fixed counterpart
and collisions 0 → 0 (with CS1, the hardest family, doubling its best
success count to 10/50). The sign of the language-feedback effect is
decided by the feasibility-handling policy, not by the feedback itself.
This is the condition under which the paper's positive feedback claim is
consistent with our data: an implementation that never freezes on
infeasibility. Statistical caveat: family-level differences vs slack
fixed are not individually significant after Holm (n=50/family); the
causal statement rests on the collision-count reversal (77 vs 0) and the
solver-failure collapse, both far outside sampling noise.

## Paired phase: channel-2 prediction-feedback arm (2026-07-20)

Isolates the language-feedback CHANNEL: same base as the fixed baseline
and the scripted-gamma arm (static prediction, hard constraint, gamma
fixed at 0.15 throughout — **never touched by this arm**), same frozen
per-instance feedback time. The only difference from `scripted_feedback`
is *what* the scripted signal changes: `prediction_mode` (static ->
velocity_tube) instead of gamma. Manifest
`configs/safe_panda_core_prediction_feedback_v1.json`.

| Family | Success | Collisions | vs static baseline (Holm) |
|---|---|---|---|
| CS1 | 1/50 | 20 | +0.02 (p=1.0, ns) |
| CS2 | 29/50 | 14 | **+0.38 (p=3.8e-6, Holm 1.1e-5)** |
| CS3 | 28/50 | 1 | **+0.28 (p=4.3e-3, Holm 8.7e-3)** |

Macro 0.387 (58/150); outcomes 58 goal / 55 timeout / 35 collision / 2
stall; 0 crashed.

**Direct channel comparison** (identical base, identical timing —
gamma-channel vs prediction-channel, same 150 instances):

| Family | scripted gamma (inert) | prediction switch | Holm |
|---|---|---|---|
| CS1 | 0/50, 50 collisions | 1/50, 20 collisions | 1.0 |
| CS2 | 10/50, 40 collisions | **29/50, 14 collisions** | **1.0e-5** |
| CS3 | 14/50, 36 collisions | **28/50, 1 collision** | **8.7e-3** |

Routing the identical scripted signal into the prediction channel
instead of gamma converts a **provably inert** intervention (150/150
concordant, Sec. above) into one that **doubles-to-triples success** in
two of three families. This confirms the channel-2 hypothesis: language
feedback is a controller-property question, and prediction is the
higher-leverage channel — consistent with the InstructMPC line [19] and
with this benchmark's own headline finding that prediction dominates
parameter tuning (Sec. "prediction is the dominant lever").

Gap to the always-on tube arm (69/150): CS2 29 vs 40, CS3 28 vs 28 (tied),
CS1 1 vs 1 (tied) — the mid-episode switch pays a cost in the early
window before feedback arrives (the episode runs on stale static
predictions until the scripted switch fires), most visible in CS2. This
is expected and informative, not a flaw: it quantifies the price of
*reactive* vs *always-on* prediction.

## Seven-arm summary (identical 150 frozen instances)

| Arm | Success | Collisions | Solver-failure steps |
|---|---|---|---|
| fixed static (baseline) | 24/150 | 126 | 104 |
| scripted feedback (static) | 24/150 | 126 | 669 |
| dead-time margin (static) | 5/150 | 20 | 4648 |
| velocity tube | **69/150** | 24 | 3740 |
| soft slack (tube base) | 63/150 | **0** | 591 |
| NIM feedback (tube base) | 48/150 | 77 | 5329 |
| NIM + soft slack | 66/150 | **0** | **1** |

## Preregistered confirmatory run (2026-07-20, separate instance set)

Prereg `configs/cs1_confirmatory_prereg_v1.md` (committed before
instance generation; one pre-run amendment logged). 120 NEW CS1
instances (sha256 `aa07f70d…`, zero geometry overlap with the
hypothesis-generating set), both arms per instance, single primary test.

**Result: CONFIRMED.** Slack fixed 17/120 vs NIM+slack feedback 30/120;
discordant 14 vs 1; exact McNemar **p = 0.00098**; effect **+0.108**
(paired bootstrap 95% CI [0.050, 0.167]). Zero collisions in all 240
episodes; one crashed feedback-arm episode (CS1-E13) counted as
non-success per protocol; zero new API requests. The CS1 feedback
effect on the slack base is now a preregistered, confirmed finding.

## Promotion checklist

- [x] 150 unique rows; 50 per family; balanced sides
- [x] setup/instance hashes match checkpoint and summary
- [x] zero provider calls in the mandatory phase
- [x] zero invalid-geometry / initial-collision episodes
- [x] zero crashed episodes (none needed diagnostics)
- [x] full test suite passes (308 passed, 2 skipped)
- [x] EE-only / simulator-only / non-whole-arm wording retained
