# Clean-room reconstruction strategy

The paper's public repository does not currently expose the controller, prompts, experiment scripts, scene configuration, or raw results. This prevents exact replication, but it does not prevent a transparent reconstruction.

## 1. Separate evidence from assumptions

Maintain three labels for every implementation decision:

- **Reported**: explicitly specified in the paper, such as equations, horizon, control period, weights, noise, and gamma interval.
- **Derived**: follows mathematically from reported content, such as the 0.6 s prediction duration.
- **Assumed**: required to execute but not disclosed, such as obstacle radius, initial pose, prompt wording, or intervention timing.

Do not silently promote an assumed value to a paper-reported value.

## 2. Reconstruct in layers

1. Verify barrier and regularizer formulas numerically.
2. Verify IPOPT feasibility and fail-closed control handling.
3. Verify Safe Panda observation/action mappings without an LLM.
4. Run fixed-gamma closed-loop MPC-CBF episodes.
5. Sweep gamma and check the qualitative monotonic relationship between conservatism and clearance.
6. Add a deterministic language-to-gamma lookup baseline.
7. Add the external LLM last, behind a recorded interface.

This ordering keeps simulator/controller defects separate from LLM variability.

## 3. Treat LLM output as experimental input

Record the model identifier, system prompt hash, user text, raw response, parsed gamma, request timestamp, latency, retry count, and parsing failures. Cache responses for deterministic reruns. Never make an external LLM call part of the safety-critical inner control loop without a bounded fallback gamma and timeout policy.

## 4. Use equivalence targets that are supportable

Without author code or raw trajectories, target:

- equation-level equivalence;
- parameter-level equivalence for reported constants;
- qualitative trend equivalence for the gamma sweep;
- protocol-level equivalence for 50-episode ablations;
- uncertainty intervals computed from newly generated data.

Do not claim exact numerical replication of Figure 5 or Figure 6 unless the hidden scene and intervention policies are recovered.

## 5. Request the missing artifacts

Open a concise issue or contact the authors requesting:

- exact Python/package versions;
- TP/OF system prompts and few-shot examples;
- environment ID and scene initialization;
- obstacle geometry, speed process, and collision radius;
- MPC initial conditions and target poses;
- intervention-timing distribution;
- raw per-episode outcomes and random seeds.

Archive the response or lack of response as part of the reproducibility record.

## 6. Preserve auditable releases

Pin Git commits and package versions, store configuration manifests with result artifacts, run experiments from a clean environment, and tag each reproduction milestone. A result is `VERIFIED` only when it can be rerun from a fresh environment with the same deterministic outputs or an explicitly defined stochastic tolerance.
