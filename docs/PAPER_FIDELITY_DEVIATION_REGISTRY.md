# Paper-Fidelity Deviation Registry

Single authoritative record of every detail in this reproduction that is
**not identical to the paper** or that the paper leaves **unpublished**.
Read this before making any paper-fidelity claim. Each entry states the
paper's wording (or its silence), what this repository does instead, why,
and where the code lives.

Paper: S. Song, D. Kang, C.-E. Park, "Safety-Aware Optimal Control With
Language-Guided Online Parameter Adjustment via Large Language Models,"
IEEE Access, vol. 14, 2026, DOI `10.1109/ACCESS.2026.3664145`.

Status legend:

- `UNPUBLISHED` — the paper declares or needs this detail but never
  publishes it; the repository value is a documented reconstruction choice.
- `DEVIATION` — the paper publishes the detail; the repository deliberately
  does something different (reason given).
- `AMBIGUOUS` — the paper's own statements conflict; the repository picks
  one reading and records the alternatives.
- `MATCH` — the paper publishes the detail and the repository reproduces it
  exactly; recorded here only when a previous entry claimed otherwise or
  non-paper variants exist alongside the faithful one.

## 1. Controller model and optimization

| # | Item | Status | Paper says | Repository does | Where |
|---|---|---|---|---|---|
| 1.1 | Discrete model, eq. (18) | MATCH (via `paper_state`; corrected 2026-07-17) | Printed matrices, verified visually against the rendered PDF (page 8, journal p. 25558): `A = [[I4, I4*dt],[0_4, 0_4]]`, `B = [[0_4],[I4]]`, so `d_next = u`: `u` is the commanded velocity. Self-consistent with the prose ("The constraint set U bounds the gripper's linear and rotational velocities", `|u| <= 0.2`). | `paper_state` in `controller.py` implements the printed eq. (18) exactly and is what paper-fidelity profiles must use (`paper_fidelity_v3_eq18_static.json`). Non-paper variants kept for comparison: `paper_increment` (`A = [[I4, I4*dt],[0_4, I4]]`, `d_next = d + u`) and `double_integrator` (`u` = acceleration, exact discretization; frozen 8-D benchmark). **Correction history:** this entry originally misread A's lower-right block as `I4`, labeled the model AMBIGUOUS (increment reading makes `|u| <= 0.2` bound only the increment), and called `paper_increment` "the printed eq. 18, exact" — all wrong; `configs/paper_fidelity_v2_eq18_static.json` was built under that misreading and is superseded by v3. | `dynamics_matrices()`, `DYNAMICS_MODES` |
| 1.2 | Terminal constraint, eq. (3e) | UNPUBLISHED | `x_{k+N|k} ∈ X_f` is declared in the formulation; `X_f` is never specified numerically. | Default: terminal cost only, no terminal set. Opt-in reconstruction: terminal safe set `h(p_{k+N}) >= 0` via a stage-gated constraint (`terminal_safe_set_enabled`), chosen because it is the weakest set consistent with the CBF forward-invariance argument (eq. 10-15). | `smooth_dynamic_demo.py` (`ReferenceObstacleTVP`) |
| 1.3 | IPOPT options | UNPUBLISHED | "We use the Python library do-mpc with the Interior Point Optimizer (IPOPT) and CasADi" — no tolerances, iteration limits, or time limits. | Verified 2026-07-17 at pinned revisions: `elena-ecn/mpc-cbf` `mpc_cbf.py` sets **no** solver options (the `MA27` line is commented out) and `HybridRobotics/NMPC-DCLF-DCBF` `matlab/cdc2021/CBFDT.m` uses `sdpsettings('solver','ipopt','verbose',0)` — both run IPOPT at **library defaults**. Sourced reconstruction: `IpoptConfig.reference_defaults()` (tol `1e-8`, acceptable `1e-6`, constr_viol `1e-4`, max_iter `3000`, no CPU limit, monotone mu, no warm start), selectable via `SmoothDynamicConfig.ipopt_profile="reference_defaults"`. The frozen repository profile (constr_viol `1e-6`, max_iter `200`, `0.035 s` CPU limit, adaptive mu, warm start) remains the default; its extra knobs are unsourced real-time choices. | `solver.py`, `smooth_dynamic_demo.py` |
| 1.4 | Solver failure handling | UNPUBLISHED | Not discussed. | Fail-closed: rejected solves are counted and produce the zero command (or the opt-in CBF-screened braking fallback). | `solver.py`, `smooth_dynamic_demo.py` |
| 1.5 | Feasibility remedies | DEVIATION (opt-in only) | Not in the paper. | `cbf_constraint_scope="first_step"` (one-step D-GCBF) and `solver_fallback_mode="brake"` exist as opt-in remedy profiles; defaults preserve the paper-shaped full-horizon hard constraint. | `smooth_dynamic_demo.py` |

## 2. Environment, sensing, obstacle

| # | Item | Status | Paper says | Repository does | Where |
|---|---|---|---|---|---|
| 2.1 | Obstacle prediction over horizon | DEVIATION (default) / match (opt-in) | "we assume that the obstacle is static along the prediction horizon." | Default `prediction_mode="velocity_tube"` (constant-velocity + uncertainty tube). Paper-faithful runs must set `prediction_mode="static"` — the paper-fidelity profile does. | `SmoothDynamicConfig` |
| 2.2 | Scene / task | MATCH (quantitative scope) / DEVIATION (qualitative demo; corrected 2026-07-17) | Colored cubes plus one spherical dynamic obstacle; full state access; scene description given to the LLM. The paper's **quantitative** experiment (Table 4 ablation, 50 episodes) is a reach task — initial query "Move gripper to red cube". Multi-cube **pick/place is the qualitative demo only**, and the paper is internally inconsistent about it: Fig. 1 says "pick up the *blue* cube and put it on the *red* cube", Fig. 3's OF example says "Safely pick up the *red* cube". | Core benchmarks are EE-reach-only — this **matches** the paper's quantitative scope (previous entry wrongly treated reach-only as a task deviation and stated the pick/place direction as red-onto-blue). The language-guided pick/place workflow on `PandaBuildL-v3` (four cubes + moving sphere, blue-onto-red per Fig. 1) mirrors the qualitative demo; mechanical end-to-end validation 2026-07-17 places the cube but grazes the moving obstacle (−2.6 mm) during the descend phase — systematic, seed-independent. Exact cube layout, obstacle path, and randomization remain unpublished. | `language_guided_pick_place.py`, `build_l_demo.py` |
| 2.3 | Obstacle geometry values | UNPUBLISHED | Spherical obstacle, constant velocity; radii values not given (r_obs "minimum radius that sufficiently encloses" the obstacle; r_collision "approximates the shape of the gripper"). | Scenario-suite values chosen locally (e.g. r_collision 0.035 m EE approximation; obstacle radius per scenario manifest). | setup JSONs under `configs/` |
| 2.4 | Whole-arm safety | DEVIATION (scope) | CBF is defined on gripper position only (eq. 13) — the paper also certifies only the gripper point. | Same point-mass CBF; repository explicitly disclaims any whole-arm certificate. | docs, claim audits |

## 3. Language pipeline

| # | Item | Status | Paper says | Repository does | Where |
|---|---|---|---|---|---|
| 3.1 | LLM | DEVIATION (authorized substitution) | GPT-4o via OpenAI API for both TP and OF. | NVIDIA NIM `meta/llama-3.1-8b-instruct` (user-directed substitution; no OpenAI spend). All alignment metrics must be re-measured for the substitute. | `nvidia_nim_gamma.py`, `hf_llm.py` |
| 3.2 | OF output format | DEVIATION (security) | OF emits executable CasADi Python code (Fig. 3) that configures the controller. | **Production path** (`language_dsl.py`, `trusted_executor.py`): LLM output is parsed as a structured payload and validated against the canonical expression template; the controller is always built by trusted repository code. LLM-generated code is **never executed directly** here. Equivalence to Fig. 3 is checked textually/structurally. **Separate, opt-in research artifact** (`code_as_policies.py`, added 2026-07-20, Code-as-Policies-style [Liang et al. 2023]): implements the paper's literal code-execution OF to measure it empirically — never wired into the production loop or any benchmark arm. Two independent defenses: an AST allowlist gate (import/exec/eval/dunder/loops all rejected; only a fixed 5-arg entry point signature accepted) and restricted execution (no builtins, SIGALRM timeout). Architectural invariant regardless of either gate: the snippet's signature has no access to obstacle state or the CBF expression, so it can only contribute an additive objective term — the CBF constraint is always injected afterward by trusted code, unconditionally, so the hard rule (never weaken the collision barrier) holds even under a theoretical sandbox escape. Battery of 8 known attack payloads: 8/8 caught (`scripts/run_code_as_policies_ablation.py`). | `language_dsl.py`, `trusted_executor.py`, `code_as_policies.py` (research-only) |
| 3.3 | gamma mapping | DEVIATION -> being aligned | Continuous gamma in (0, 1], Table 1 examples 0.001-1, Table 2 sub-interval labels. | Historical runs used discrete {0.03, 0.07} and capped gamma <= 0.15. Paper-fidelity path accepts continuous gamma in (0, 1] with Table 2 labels. | `contextual_gamma.py`, `language_dsl.py` |
| 3.4 | TP/OF prompts | UNPUBLISHED | "pre-defined system prompts", scene description, 1-3 few-shot examples — none reproduced in the paper. | Prompts written locally; recorded verbatim in provider checkpoints for auditability. | prompt constants in language modules |
| 3.5 | Feedback trigger | UNPUBLISHED | User interjects naturally ("Watch out!") during execution; timing distribution not specified. | Benchmarks use predeclared feedback times / TTC triggers; documented per setup manifest. | setup JSONs |
| 3.6 | Alignment study | DEVIATION (scale) | 50 queries, 5 human Likert raters, Spearman 0.85 / Kendall 0.75 / Pearson 0.85. | Local N=2 alignment probe only (`docs/LANGUAGE_ALIGNMENT_N2A1.md`); no human-rater panel. Paper metrics are not reproduced. | language alignment docs |

## 4. Experiments and statistics

| # | Item | Status | Paper says | Repository does | Where |
|---|---|---|---|---|---|
| 4.1 | Scenario definitions and experiment scope | UNPUBLISHED (geometry) / DEVIATION (repo extension; clarified 2026-07-17) | The paper tests **one scenario geometry**: Fig. 5 trajectory comparison uses a single head-on encounter with identical initial/target positions ("all tested scenarios" there = the swept gamma values, not geometries); the Table 4 ablation runs 50 episodes of the same single scenario type randomizing **only obstacle speed** (uniform 0.025–0.2 m/s). No numeric scenario tables, seeds, or positions are published. | Locally designed 50-scenario suites reproduce the ablation's shape (50 episodes, reach, feedback vs fixed). The **executed** 150-episode three-family benchmark (`docs/SAFE_PANDA_CORE_SCENARIOS_150_RESULT.md`, three encounter geometries x 50 LHC episodes) is a robustness **extension beyond the paper** and must never be quoted as a paper reproduction. All local suites publish manifests and hashes. | `configs/`, scenario plan doc, result doc |
| 4.2 | Success metrics | PARTIALLY PUBLISHED (corrected 2026-07-17) | The ablation **does** define success: "successful if the gripper reached the target without colliding with the dynamic obstacle", with bootstrap 95% CI over 10,000 resamples (previous entry wrongly said the counting protocol was absent). Per-episode data, seeds, and any paired-test protocol are not published. | Predeclared `collision_free_goal_success` matches the paper's definition; repository adds paired exact McNemar, Wilson intervals, and Holm correction as predeclared extensions. Bootstrap resamples kept at 10,000 to match. | benchmark modules |
| 4.3 | Feedback-advantage result | Not yet reproduced | LaMPC-CBF with feedback improves safety vs non-interactive variant. | 8-D: no difference (p=1.0). 3-D: +0.08 (p=0.125, ns). | `docs/MASTER_PLAN_SETUP.md` |

## Maintenance rule

Any new work that (a) fills in an unpublished detail or (b) knowingly departs
from the paper MUST add or update one row here, in the same commit as the
change. When an UNPUBLISHED item later becomes published (e.g. author code
release), update the row and re-audit the affected claims.
