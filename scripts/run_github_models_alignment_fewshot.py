#!/usr/bin/env python3
"""Few-shot (in-context learning) alignment reproduction with GPT-4o.

Reproduces reference [25]'s in-context-learning setup that the LaMPC-CBF paper
relies on: the Optimization Formulator receives a small set of paired
(language -> gamma) examples via the system prompt and must generalize the
mapping to unseen queries. Two leakage-free conditions:

  loo     : leave-one-out 7-shot. For each of the eight published Table-1
            queries, the other seven published pairs are the few-shot examples;
            the held-out query is predicted. Uses only published data.
  anchor3 : a fixed set of three constructed anchor examples (distinct from all
            eight test queries) in the system prompt, matching the paper's
            "1-3 examples via the system prompt" design; predicts all eight.

Predictions are scored against the paper's Table-1 gammas with the shared
scoring functions. One request per prediction; checkpoint after each. GPT-4o is
called via GitHub Models (OpenAI-compatible). The token is read locally, never
printed.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep, time
from urllib.request import Request, urlopen

from lampc_cbf.language_alignment import (
    AlignmentPrediction,
    PUBLISHED_ALIGNMENT_CASES,
    evaluate_published_smoke,
    result_rows,
    validate_alignment_gamma,
)
from lampc_cbf.paper_continuous_gamma import TABLE1_CALIBRATION

ENDPOINT = "https://models.github.ai/inference/chat/completions"

BASE_SYSTEM = (
    "You are the Optimization Formulator safety-parameter module for a robot "
    "controlled by MPC with a discrete control barrier function (CBF). Map the "
    "user's safety intent to a continuous gamma in (0, 1]. Smaller gamma means "
    "earlier, more proactive avoidance and larger obstacle clearance; gamma = 1 "
    "reduces the CBF to a plain distance constraint (no proactive avoidance). "
    "Return only the requested JSON object. Never generate executable code."
)

OUTPUT_CONTRACT = (
    'Output contract: return exactly one JSON object shaped as {"gamma": number}. '
    "Do not include markdown, analysis, reasoning tags, or additional keys."
)

# Constructed anchors: distinct wording from every published test query, spanning
# the scale (strong / moderate / weak safety intent) per the paper's sub-interval
# convention. These are demonstration anchors, not paper-published few-shot text.
ANCHOR_EXAMPLES = (
    ("Approach the target with maximum caution, avoiding the obstacle as early "
     "as possible.", 0.02),
    ("Reach the target while keeping a moderate, sensible distance from the "
     "obstacle.", 0.1),
    ("Go straight to the target; do not worry about the obstacle.", 0.95),
)


def build_system(examples: tuple[tuple[str, float], ...]) -> str:
    lines = "\n".join(f'- "{q}" -> gamma {g}' for q, g in examples)
    return f"{BASE_SYSTEM}\n\nCalibration examples:\n{lines}"


def predict(instruction: str, system: str, *, model: str, token: str,
            timeout: float, max_tokens: int) -> AlignmentPrediction:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": f"{OUTPUT_CONTRACT}\nRobot-motion instruction: {instruction.strip()}"},
    ]
    payload = {
        "model": model, "messages": messages, "temperature": 0.0,
        "max_tokens": max_tokens, "seed": 11, "stream": False,
        "response_format": {"type": "json_object"},
    }
    request_hash = sha256(
        json.dumps({"endpoint": ENDPOINT, "payload": payload},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    requested_at = time()
    started = perf_counter()
    request = Request(
        ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                 "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    raw = body["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    gamma = validate_alignment_gamma(parsed["gamma"])
    return AlignmentPrediction(
        gamma=gamma, model=model, provider="github-models",
        latency_seconds=perf_counter() - started, requested_at_unix=requested_at,
        prompt_hash=sha256(system.encode("utf-8")).hexdigest(),
        request_hash=request_hash, raw_response=raw)


def load_token(path: str) -> str:
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("token file must contain exactly one non-empty token")
    return token


def run_mode(mode: str, token: str, model: str, timeout: float, max_tokens: int,
             delay: float, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.json"
    predictions: list[AlignmentPrediction] = []
    for index, case in enumerate(PUBLISHED_ALIGNMENT_CASES):
        if mode == "loo":
            examples = tuple(pair for j, pair in enumerate(TABLE1_CALIBRATION) if j != index)
        elif mode == "anchor3":
            examples = ANCHOR_EXAMPLES
        else:
            raise ValueError(mode)
        system = build_system(examples)
        prediction = predict(case.instruction, system, model=model, token=token,
                             timeout=timeout, max_tokens=max_tokens)
        predictions.append(prediction)
        print(f"  [{mode}] case {case.case_id}: gamma={prediction.gamma} "
              f"label={prediction.safety_label} (paper {case.paper_gamma}/"
              f"label {case.paper_label})")
        checkpoint.write_text(json.dumps(
            {"mode": mode, "model": model, "provider": "github-models",
             "completed": index + 1,
             "records": [p.as_dict() for p in predictions]}, indent=2),
            encoding="utf-8")
        if index + 1 < len(PUBLISHED_ALIGNMENT_CASES):
            sleep(delay)
    summary = evaluate_published_smoke(predictions)
    summary["mode"] = mode
    summary["model"] = model
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(result_rows(predictions), indent=2),
                                          encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--token-file", default="githubtk.txt")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--inter-request-delay", type=float, default=2.0)
    parser.add_argument("--modes", nargs="+", default=["loo", "anchor3"])
    parser.add_argument("--output-root",
                        default="artifacts/language_alignment_fewshot_github_models")
    args = parser.parse_args()

    token = load_token(args.token_file)
    root = Path(args.output_root)
    results = {}
    for mode in args.modes:
        print(f"=== mode: {mode} ===")
        results[mode] = run_mode(mode, token, args.model, args.timeout_seconds,
                                 args.max_tokens, args.inter_request_delay, root / mode)

    print("\n=== Few-shot in-context learning results (GPT-4o via GitHub Models) ===")
    print(f"{'mode':10s} {'Spearman':>9s} {'Kendall':>8s} {'Pearson':>8s} {'exact':>7s} {'MAE':>7s}")
    for mode, summary in results.items():
        lab = summary["paper_style_label_metrics"]
        cont = summary["continuous_gamma_diagnostics"]
        print(f"{mode:10s} {lab['spearman_rho']:9.3f} {lab['kendall_tau_b']:8.3f} "
              f"{lab['pearson_r']:8.3f} {round(lab['exact_label_accuracy']*8)}/8   "
              f"{cont['mean_absolute_error']:7.4f}")
    print("\nReference: zero-shot blinded GPT-4o was Spearman -0.507, exact 1/8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
