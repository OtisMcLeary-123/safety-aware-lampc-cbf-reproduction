# GitHub Repository Registry

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: repository audit supporting experiment planning
- Origin Date: 2026-07-14
- Verification Status: ANALYZED
- Version Label: github_registry_v1

## Scope and audit method

This registry records the GitHub repositories that can be traced from the
current tree and Git history of this reproduction. It distinguishes a runtime
dependency from a repository that was inspected for an algorithmic pattern.

The 2026-07-14 audit used:

1. GitHub URLs in the current `README.md`, `docs/`, and `pyproject.toml`.
2. GitHub URLs present in earlier commits of those files.
3. Current GitHub repository metadata and root contents.
4. GitHub commit lookup for every revision explicitly recorded as reviewed.
5. Source and protocol references that connect an external idea to a local
   implementation or experiment.

The audit proves documentary traceability. It cannot prove every repository a
person may have viewed outside the recorded project history.

## Evidence levels

| Level | Meaning |
|---|---|
| Runtime | The repository is installed or executed by this project. |
| Primary source | The repository is cited by the paper or its authors. |
| Reviewed | The repository was inspected at a recorded commit and influenced a protocol or implementation decision. |
| Precedent | The repository supplied an architectural pattern; no direct runtime dependency is claimed. |
| Workspace | The repository contains this reproduction and its generated evidence. |

## Repository inventory

| Repository | Evidence level | Recorded revision | Role in this reproduction | Local evidence and boundary |
|---|---|---|---|---|
| [sohonsong/safety-aware-llm-control](https://github.com/sohonsong/safety-aware-llm-control) | Primary source | Not pinned | Public repository associated with the paper. It is the first place to check for author code, prompts, scenes, and raw data. | Live audit found only `.gitignore`, `LICENSE`, and `README.md`. It does not currently supply controller code, prompts, experiment scripts, or raw results. See `docs/REPRODUCIBILITY.md`. |
| [tohsin/Safe-panda-gym](https://github.com/tohsin/Safe-panda-gym) | Primary source | Historical dependency `f6cf3031f489f2f11c9e9274f3a17dda5670a649` | Paper-cited simulator fork and upstream source for the safe Panda environments. | The historical pin exposed legacy/merged Gym API incompatibilities on Python 3.12. It is no longer the installed dependency. Compatibility work was proposed upstream in pull request 1. |
| [OtisMcLeary-123/Safe-panda-gym](https://github.com/OtisMcLeary-123/Safe-panda-gym) | Runtime | Current `c2c2bae9ee0b738fd7c5a5f6259a3a37da95718c`; earlier `97a7bbf6c619e5e20ba3bde3c5f423b435a3062a` | Compatibility fork actually installed by `pyproject.toml`. It restores safe environment exports and supports Gymnasium/Python 3.12. | This fork makes the simulator executable but does not recover the unpublished paper scene, observation schema, action adapter, or randomization process. |
| [OtisMcLeary-123/safety-aware-lampc-cbf-reproduction](https://github.com/OtisMcLeary-123/safety-aware-lampc-cbf-reproduction) | Workspace | Current project commit | Owns the clean-room controller, tests, protocols, scripts, and artifacts. | It is the reproduction under evaluation, not independent evidence for the paper's claims. |
| [HybridRobotics/MPC-CBF](https://github.com/HybridRobotics/MPC-CBF) | Reviewed | `3a4ea54` | Reference for finite-horizon discrete MPC-CBF, the constraint `h[k+1]-(1-gamma)h[k] >= 0`, explicit infeasibility handling, and gamma/horizon ablations. | Influences `src/lampc_cbf/symbolic.py`, `src/lampc_cbf/controller.py`, solver diagnostics, and the paper-fidelity benchmark profile. The repository is currently archived; the recorded commit remains accessible. |
| [tkkim-robot/safe_control](https://github.com/tkkim-robot/safe_control) | Reviewed | `ac950ba` | Reference for optimal-decay MPC-CBF, C3BF/DPCBF, gatekeeper, backup policies, and safety-filter separation. | Influences optimal decay, `src/lampc_cbf/safety_reflex.py`, `src/lampc_cbf/formal_safety.py`, and the DPCBF ablation. The Cartesian DPCBF is explicitly a non-proof-preserving adaptation, not a direct port of the bicycle-model guarantee. |
| [learnsyslab/safe-control-gym](https://github.com/learnsyslab/safe-control-gym) | Reviewed | `6b5391d` | Reference for separating symbolic constraints, disturbance injection, controller performance, safety filters, terminal sets, and repeatable PyBullet/CasADi evaluation. | Influences the formal-scope audit and protocol-v5 bounded-error design. No safety certificate is inherited by this project. |
| [bardhh/cbfkit](https://github.com/bardhh/cbfkit) | Reviewed | `4e92cb3` | Reference for robust CBF-QP construction and dynamic obstacles represented in augmented state-time space. | Used to assess what a future certified local filter would require. It is not a dependency and its certificate does not apply to the current Panda controller. |
| [tkkim-robot/plcbf](https://github.com/tkkim-robot/plcbf) | Precedent | Not recorded | Policy-library CBF pattern: screen a finite set of safe fallback candidates before optimizing task progress. | Adapted architecturally in the collision-cone liveness policy library in `src/lampc_cbf/safety_reflex.py`. No controller formulation is claimed to be copied. |
| [sh-keyumarsi/Circulation-embedded-CBF](https://github.com/sh-keyumarsi/Circulation-embedded-CBF) | Precedent | Not recorded | Signed circulation pattern for avoiding undesirable equilibria and repeated side switching. | Motivates side latching and hysteresis in the liveness ablation. It does not prove liveness for the Safe Panda model. |
| [UMich-BipedLab/multi_object_avoidance_via_clf_cbf](https://github.com/UMich-BipedLab/multi_object_avoidance_via_clf_cbf) | Precedent | Not recorded | Navigation pattern that feeds a temporary liveness subgoal to the nominal planner. | Motivates the goal-biased tangential subgoal used by the selected C3BF liveness stack. It is not a runtime dependency. |

## Scenario-authoring framework review (2026-07-16)

This review supports `docs/SAFE_PANDA_CORE_SCENARIO_PLAN.md`. No maintained
GUI scenario editor or deployed documentation site was found specifically for
Safe Panda Gym. The selected design uses the Panda-Gym `Task` and
`RobotTaskEnv` APIs with the pinned Safe Panda/PyBullet runtime.

| Repository | Evidence level | Recorded revision | Scenario-authoring relevance | Adoption boundary |
|---|---|---|---|---|
| [qgallouedec/panda-gym](https://github.com/qgallouedec/panda-gym) | Reviewed | `10c4d8adaab07d8a29b2f8dc0aa85f8edd8532ae` | Defines the upstream `Task`, `RobotTaskEnv`, Panda robot, and PyBullet scene wrapper used to compose custom environments. | API and documentation reference; the installed runtime remains the compatibility fork. |
| [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Reviewed | `d3f3c7fd3b1519b9edff0db0d51b1d30c9245fb1` | Defines current custom-environment reset/step, deterministic seeding, registration, and environment-checking contracts. | Compliance reference only. |
| [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) | Runtime lineage | `63c4d67e337017f9d8b298c900e9aabdb69296e7` | Physics engine supporting primitive bodies, collisions, dynamics, and URDF/SDF/MJCF assets beneath Panda-Gym. | Low-level reference; scenarios should normally use the Panda-Gym wrapper. |
| [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | Alternative reviewed | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | Provides modular MuJoCo Arena, Object, Robot, Task, and placement-sampler abstractions with Panda support. | Not drop-in compatible; migration would invalidate direct PyBullet comparison. |
| [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill) | Alternative reviewed | `42b68244c1497cef889b04c4f4a78aa01c927f4e` | Provides reusable scenes, Panda tasks, and GPU-parallel simulation. | Not drop-in compatible; requires SAPIEN/ManiSkill migration. |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | Alternative reviewed | `b4c321024792976150ca55fddb26fa34480d974e` | Provides Isaac Sim GUI/USD scene authoring and configuration-driven `InteractiveScene` composition. | Heavy simulator and adapter migration; retained as a future visualization option only. |

### Scenario-editor UI references

| Repository | Evidence level | Recorded revision | UI pattern reviewed | Adoption boundary |
|---|---|---|---|---|
| [markusgrotz/mujoco-scene-editor](https://github.com/markusgrotz/mujoco-scene-editor) | Reviewed | `a401997a9f5d02ab5073d5901fb37df4ff278a17` | Local browser server, grouped scene controls, selected-element inspector, transform panel, reset/undo/export workflow. | Interaction precedent only; Viser and MuJoCo are not dependencies of Scenario Lab. |
| [cyberbotics/webots](https://github.com/cyberbotics/webots) | Reviewed | `22f33694b71c8954caf972406ab965eddf38f831` | Scene tree, field editor, and central simulation viewport separation. | Architectural precedent only; no Qt/C++ code is reused. |
| [carla-simulator/traffic-generation-editor](https://github.com/carla-simulator/traffic-generation-editor) | Reviewed | `98ce87b18eb1bd8265c97abba0c6408f307910bd` | Viewport-first workflow with docked environment/entity/maneuver forms and explicit import/export. | Interaction precedent only; OpenSCENARIO/QGIS are outside the Safe Panda data model. |
| [foxglove/studio](https://github.com/foxglove/studio) | Reviewed archive | `a8a589b801d1ad04915f4f22868989e222668f5e` | Dense robotics workspace, panel hierarchy, and clear data-view separation. | Visual precedent only; the reviewed repository is archived and not a dependency. |

## MPC-CBF feasibility remedy review (2026-07-17)

This review supports the solver-feasibility gate in `docs/MASTER_PLAN_SETUP.md`
(soft horizon CBF slack with exact penalty, hard one-step D-GCBF, braking
fallback). Two relevant repositories were already registered above:
`HybridRobotics/MPC-CBF` (`3a4ea54`) and `learnsyslab/safe-control-gym`
(`6b5391d`). The following are newly reviewed.

| Repository | Evidence level | Recorded revision | Feasibility relevance | Adoption boundary |
|---|---|---|---|---|
| [HybridRobotics/NMPC-DCLF-DCBF](https://github.com/HybridRobotics/NMPC-DCLF-DCBF) | Reviewed | `3f40c67578f49114301b02e744e5a86fa671a981` | Collection containing the CDC 2021 "Enhancing Feasibility and Safety of NMPC with Discrete-Time CBFs" code (`matlab/cdc2021`): generalized/relaxed DCBF with decay-rate decision variables and constraining fewer horizon steps — the direct algorithmic source for the D-GCBF remedy profile. Also contains the ACC 2023 iterative-convex high-order DCBF work. | MATLAB/Yalmip precedent only; formulations are re-derived in `symbolic.py`/`controller.py`, no code is ported. |
| [HybridRobotics/CBF-Pointwise-Feasibility](https://github.com/HybridRobotics/CBF-Pointwise-Feasibility) | Reviewed archive | `06ce5095cf979fb5d7361c91804a53c3c18e35fa` | Optimal-decay CBF-QP with guaranteed point-wise feasibility, including infeasibility demonstrations and penalty-hyperparameter studies relevant to exact-penalty tuning of the decay deviation. | MATLAB precedent validating the existing `optimal_decay_cbf_expression()`; the QP-level guarantee does not transfer to the NMPC horizon setting. |
| [elena-ecn/mpc-cbf](https://github.com/elena-ecn/mpc-cbf) | Reviewed | `aa6f12250d23b9645c830af4fcdd76c77c44e369` | Python MPC-DCBF on the same do-mpc/CasADi stack as this project: horizon DCBF constraint wiring in do-mpc, gamma sweeps, and an MPC-DC distance-constraint baseline. | Pattern reference for do-mpc constraint plumbing and a possible MPC-DC comparison profile; unicycle model and no feasibility remedy, so no formulation is inherited. |

## Test and protocol mapping

| Local test or protocol | External repositories that informed it | Function in the test |
|---|---|---|
| Paper-fidelity MPC-CBF and gamma ablation | `HybridRobotics/MPC-CBF`; author paper repository | Establish the discrete CBF form, gamma contrast, horizon-style controller behavior, and explicit infeasibility reporting. |
| Predictive, optimal-decay, and gatekeeper extensions | `tkkim-robot/safe_control`; `learnsyslab/safe-control-gym`; `CBFKit` | Separate feasibility, disturbance handling, final-input filtering, and formal-scope claims. These are extensions, not strict paper reproduction. |
| Formal safety protocol v5 | `safe_control`; `safe-control-gym`; `MPC-CBF` | Motivate a final command filter, bounded transition/measurement errors, and a backup/terminal contract. |
| Collision-cone liveness ablation | `plcbf`; `Circulation-embedded-CBF`; `multi_object_avoidance_via_clf_cbf` | Motivate policy screening, persistent avoidance side, and temporary MPC subgoals. |
| Cartesian DPCBF negative ablation | `tkkim-robot/safe_control` | Supply the source bicycle-model DPCBF structure and default parameters used for the labeled Cartesian adaptation. |
| Safe Panda runtime validation | `tohsin/Safe-panda-gym`; `OtisMcLeary-123/Safe-panda-gym` | Provide the paper-cited simulator lineage and the pinned compatibility runtime used by all simulation evidence. |
| Submission-phase agent loops | `gaasher/agent-loop-skills` via fork `OtisMcLeary-123/Agent-Loop-Skills`, pinned `f1169e6db0b0f8a83ced3a18562b7c57e14a748a` (MIT) | Five verification-gated loop skills copied verbatim into `.claude/skills/` (scientific-writer, claim-verify, scientific-figure, power-analysis, literature-search) to drive the IEEE draft critique/revise cycle, pre-publication claim verification, figure iteration, and the n=50 power analysis. Workflow tooling only — no controller, benchmark, or statistics code is adopted from it; provider rules still gate any keyed literature API (arXiv/OpenAlex are keyless). |

## Current audit findings

- Only `OtisMcLeary-123/Safe-panda-gym` is an external Git dependency in the
  current environment.
- The four controller/safety repositories recorded in protocol v4 still expose
  the exact reviewed commits listed above.
- The author repository still does not expose the missing implementation or raw
  experimental artifacts, so exact replication remains blocked on undisclosed
  inputs.
- The three liveness precedent repositories have no recorded commit or file-level
  inspection log. Their revisions must be pinned before their ideas are used in
  another confirmatory protocol.
- Repository licenses and exact inspected file paths are not yet captured for
  every precedent. This is a traceability gap, even where no source was copied.

## Update procedure

For every future GitHub repository review, append or update one row with:

1. Canonical URL, access date, default branch, and immutable commit SHA.
2. Exact files or examples inspected.
3. The local module, experiment, or decision influenced by the repository.
4. Adoption type: dependency, copied with attribution, adapted, conceptual only,
   or rejected after evaluation.
5. License and any compatibility or attribution requirement.
6. A statement of which guarantees do not transfer to this project.
7. The protocol or artifact that tests the adopted idea.

Do not cite a moving branch such as `main` as the sole reproducibility record.
