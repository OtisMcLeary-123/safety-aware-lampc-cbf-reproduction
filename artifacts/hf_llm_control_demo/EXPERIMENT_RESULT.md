## Material Passport

- Material ID: `hf-language-to-gamma-integration-2026-07-12`
- Type: external LLM integration and Safe Panda simulation
- Verification status: `VERIFIED_WITH_PROVIDER_LIMITATION`
- Selected model: `Qwen/Qwen3-235B-A22B-Instruct-2507`
- Selected provider: `deepinfra` through Hugging Face Inference Providers
- Token handling: local ignored file, mode `600`, never recorded

## Model selection

The paper uses GPT-4o as both Task Planner and Optimization Formulator. This
first clean-room integration implements only language-to-CBF-gamma mapping.
Qwen was selected because Hugging Face marks its DeepInfra route as supporting
structured output and the Qwen model card reports strong instruction-following,
reasoning, coding, and tool-use results, including comparisons with GPT-4o.
Those numbers are vendor-reported; this experiment does not claim independent
GPT-4o equivalence.

`openai/gpt-oss-120b` was also tested with the same production schema. Its
provider route produced HTTP failures or empty message content in all five
cases, so it was rejected for this adapter. This is a provider-integration
result rather than a general model-quality judgment.

## Structured mapping smoke test

Production schema permits only the calibrated pairs:

| Safety level | Gamma |
|---:|---:|
| 1 — most cautious | 0.02 |
| 2 | 0.05 |
| 3 | 0.08 |
| 4 | 0.11 |
| 5 — least cautious | 0.15 |

Qwen returned valid, correctly ordered decisions for the first four prompts:
`0.02, 0.02, 0.08, 0.11`. The fifth request received an HTTP provider error and
used the bounded `0.05` fallback. Valid-response mean latency was `1.981 s`,
below the paper's reported `2.4 s` mean GPT-4o API latency. Schema success was
`4/5`; therefore provider reliability is not yet production-grade.

An earlier numeric JSON Schema run is retained separately. It showed the model
reasoning about `0.02` while emitting `0.15`, motivating the safer enum-string
schema. Failed provider decisions are never cached.

## Connected controller run

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_hf_configured_mpc_cbf.py
```

Instruction:

```text
Keep a generous distance from the moving obstacle and be cautious.
```

The model returned safety level 2 and `gamma=0.05` in `1.995 s` without
fallback. The validated gamma configured the MPC-CBF episode before execution.

| Metric | Result |
|---|---:|
| Goal reached | yes |
| Collision | no |
| Control steps | 39 |
| Minimum raw true clearance | 0.037741 m |
| Final goal distance | 0.036038 m |
| Mean MPC solve time | 0.007582 s |
| Maximum MPC solve time | 0.011652 s |
| Jerk RMS | 5.965070 m/s³ |

The LLM was not called inside the 40 ms control loop. Invalid output, timeout,
or HTTP failure cannot disable CBF and selects `gamma=0.05`. The controller TVP
now also exposes validated `update_gamma()` for the next online-feedback stage.

## Artifacts

- `llm_control_manifest.json`: instruction, model decision, hashes, latency,
  fallback policy, and simulation result.
- `simulation/robot_motion.gif`: LLM-configured robot motion.
- `simulation/raw_smoothed_and_safety.png`: trajectory and raw clearance.
- `../hf_llm_smoke_qwen_production/smoke_results.json`: Qwen production schema test.
- `../hf_llm_smoke_gpt_oss_production/smoke_results.json`: gpt-oss provider test.
- `../hf_llm_smoke_v1_numeric_schema/smoke_results.json`: rejected numeric-schema evidence.

## Remaining boundary

This does not yet reproduce the paper's unpublished Task Planner prompt,
generated CasADi optimization code, 50-query human-alignment dataset, or
50-episode feedback study. The next implementation step is a single online
feedback event that supplies the current gamma as context and hot-swaps the
validated result at a control-cycle boundary.
