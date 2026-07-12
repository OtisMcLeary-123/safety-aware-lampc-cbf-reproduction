# Hugging Face LLM integration

## Paper role

The paper uses GPT-4o for a Task Planner and Optimization Formulator. The
reconstructed first integration intentionally implements only the narrowest
safety-critical boundary: mapping free-form safety intent to the CBF parameter
`gamma`. Smaller gamma means earlier, more conservative avoidance. The paper
uses few-shot examples and reports a 2.4 s mean OpenAI API latency.

## Selected model

Default: `Qwen/Qwen3-235B-A22B-Instruct-2507` through Hugging Face Inference
Providers and DeepInfra.

Reasons:

- The official model card reports strong instruction-following, reasoning,
  coding, and tool-use performance, including results competitive with the
  GPT-4o column. These are vendor-reported benchmarks, not an independent
  equivalence claim: <https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507>.
- Hugging Face currently marks this model/provider combination as supporting
  tools and structured output: <https://huggingface.co/inference/models>.
- The structured-output API accepts JSON Schema and avoids executing generated
  code: <https://huggingface.co/docs/inference-providers/en/guides/structured-output>.

`openai/gpt-oss-120b` remains a candidate. It supports function schemas, but
the tested provider returned empty content or HTTP failures in this integration,
so it is not the default. This is an integration finding, not a general model
quality verdict.

## Safety boundary

- `hftoken.txt` is ignored by Git, permission `600`, read only at runtime, and
  never copied to environment variables or output artifacts.
- Model output is restricted to five calibrated string values:
  `0.02, 0.05, 0.08, 0.11, 0.15`.
- Gamma and safety level must agree and are validated locally.
- Invalid output, timeout, or provider failure selects bounded fallback
  `gamma=0.05`; failures are not cached.
- The LLM never runs in the 40 ms MPC inner loop and never bypasses CBF.
- Successful responses record model, provider, hashes, latency, raw response,
  timestamp, and cache status. Tokens are never recorded.

## Commands

```bash
source .venv/bin/activate
python -m pip install -e '.[llm,simulation]'
PYTHONPATH=src python scripts/run_hf_gamma_smoke.py
PYTHONPATH=src python scripts/run_hf_configured_mpc_cbf.py
```

The next stage is online feedback: pass the current gamma with the user's new
instruction, validate the decision, then call `ReferenceObstacleTVP.update_gamma`
at a control-cycle boundary without rebuilding the MPC model.
