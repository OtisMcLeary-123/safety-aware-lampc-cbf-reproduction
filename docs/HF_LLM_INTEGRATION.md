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

## N2-A1 blinded alignment smoke test

The paper's Table 1 publishes only eight of the 50 alignment queries. The full
query set and the five human ratings per query are unavailable. The reproduction
therefore treats the eight published query/gamma pairs as an OF regression smoke
test, not as a reproduction of the paper's human-alignment claim.

The smoke evaluator is intentionally isolated from the controller mapper:

- it accepts the paper's theoretical domain `0 < gamma <= 1`;
- its prompt contains no Table 1 target examples or Table 2 label thresholds;
- it derives labels locally using the exact Table 2 intervals;
- it reports label correlations plus continuous-gamma diagnostics; and
- its output cannot enter the MPC controller, whose validated domain remains
  `0 < gamma <= 0.15`.

After explicit approval to send the eight published instructions to the selected
Hugging Face provider, run:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_language_alignment_smoke.py
```

Even a perfect result here only demonstrates agreement with eight published OF
examples. Reproducing Table 3 still requires a preregistered 50-query dataset and
five independent human ratings for every query.

### NVIDIA NIM fallback providers

When Hugging Face routed credits are unavailable, the approved R2 alternative is
the NVIDIA NIM free development endpoint with
`qwen/qwen3.5-397b-a17b`. This changes the evaluated model and provider, so its
results must be reported separately from the original Hugging Face configuration.

The model-specific hosted API documents seed control and disabling thinking mode,
but not server-side JSON Schema. The adapter therefore requests exactly one JSON
object in the blind prompt and rejects malformed, out-of-domain, or extra-key
responses locally. It never substitutes a fallback prediction.

Store the NVIDIA API key only in the ignored `nvidiatoken.txt` file with mode
`600`. After explicit external-upload approval, run the non-benchmark formatter
probe first:

```bash
PYTHONPATH=src python scripts/run_nvidia_nim_alignment_probe.py
```

Only after that probe passes, request separate approval to submit the eight public
Table 1 queries and run:

```bash
PYTHONPATH=src python scripts/run_nvidia_nim_alignment_smoke.py
```

Observed provider outcome on 2026-07-14: the Qwen formatter probes passed, but
benchmark attempts encountered a timeout, HTTP 500 responses, and a response
containing `reasoning_content` without final `content`. Those pilot responses are
not part of the reported metrics. The completed fallback run used
`z-ai/glm-5.2` on the same NVIDIA endpoint, with the same blind prompt,
`temperature=0`, `seed=11`, thinking disabled, and strict local validation:

```bash
PYTHONPATH=src python scripts/run_nvidia_nim_alignment_smoke.py \
  --model z-ai/glm-5.2 --timeout-seconds 120 --max-tokens 512 \
  --output-dir artifacts/language_alignment_smoke_nvidia_nim_glm52
```

See `LANGUAGE_ALIGNMENT_N2A1.md` for the claim-bounded validation report.
