#!/usr/bin/env python3
"""N2-A1 blinded alignment smoke test against GPT-4o via GitHub Models.

Reuses the published eight-case set, the blinded system prompt, the output
contract, and the scoring functions from ``language_alignment``. Only the
provider transport differs: an OpenAI-compatible payload posted to the GitHub
Models inference endpoint (no NIM-specific fields such as
``chat_template_kwargs``). One uncached request per case; a checkpoint is
written after every record. The token file is read locally and never printed.
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
    BLIND_ALIGNMENT_SYSTEM_PROMPT,
    NVIDIA_NIM_OUTPUT_CONTRACT,
    PUBLISHED_ALIGNMENT_CASES,
    evaluate_published_smoke,
    result_rows,
    validate_alignment_gamma,
)

ENDPOINT = "https://models.github.ai/inference/chat/completions"


def load_token(path: str) -> str:
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token or any(ch.isspace() for ch in token):
        raise ValueError("token file must contain exactly one non-empty token")
    return token


def predict(instruction: str, *, model: str, token: str, timeout: float,
            max_tokens: int) -> AlignmentPrediction:
    messages = [
        {"role": "system", "content": BLIND_ALIGNMENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{NVIDIA_NIM_OUTPUT_CONTRACT}\n"
                f"Robot-motion instruction: {instruction.strip()}"
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 11,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request_hash = sha256(
        json.dumps({"endpoint": ENDPOINT, "payload": payload},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    requested_at = time()
    started = perf_counter()
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    raw = body["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or "gamma" not in parsed:
        raise ValueError(f"response must contain gamma; got keys {sorted(parsed)}")
    gamma = validate_alignment_gamma(parsed["gamma"])
    prompt_material = f"{BLIND_ALIGNMENT_SYSTEM_PROMPT}\n{NVIDIA_NIM_OUTPUT_CONTRACT}"
    return AlignmentPrediction(
        gamma=gamma,
        model=model,
        provider="github-models",
        latency_seconds=perf_counter() - started,
        requested_at_unix=requested_at,
        prompt_hash=sha256(prompt_material.encode("utf-8")).hexdigest(),
        request_hash=request_hash,
        raw_response=raw,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--token-file", default="githubtk.txt")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--inter-request-delay", type=float, default=2.0)
    parser.add_argument(
        "--output-dir", default="artifacts/language_alignment_smoke_github_models"
    )
    args = parser.parse_args()

    token = load_token(args.token_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "alignment_smoke_checkpoint.json"

    predictions: list[AlignmentPrediction] = []
    latencies: list[float] = []
    for index, case in enumerate(PUBLISHED_ALIGNMENT_CASES):
        prediction = predict(
            case.instruction, model=args.model, token=token,
            timeout=args.timeout_seconds, max_tokens=args.max_tokens,
        )
        predictions.append(prediction)
        latencies.append(prediction.latency_seconds)
        print(
            f"[{index + 1}/8] case {case.case_id}: gamma={prediction.gamma} "
            f"label={prediction.safety_label} (paper gamma={case.paper_gamma}, "
            f"label={case.paper_label}) latency={prediction.latency_seconds:.2f}s"
        )
        checkpoint_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "provider": "github-models",
                    "endpoint": ENDPOINT,
                    "completed": index + 1,
                    "records": [p.as_dict() for p in predictions],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if index + 1 < len(PUBLISHED_ALIGNMENT_CASES):
            sleep(args.inter_request_delay)

    summary = evaluate_published_smoke(predictions)
    summary["provider"] = "github-models"
    summary["model"] = args.model
    summary["latency_seconds"] = {
        "mean": mean(latencies), "max": max(latencies), "min": min(latencies),
    }
    (output_dir / "alignment_smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "alignment_smoke_rows.json").write_text(
        json.dumps(result_rows(predictions), indent=2), encoding="utf-8"
    )

    label = summary["paper_style_label_metrics"]
    print("\n=== Table-3-style label metrics (vs paper Table-1 OF gammas) ===")
    print(f"  Spearman rho        : {label['spearman_rho']:.3f}")
    print(f"  Kendall tau-b       : {label['kendall_tau_b']:.3f}")
    print(f"  Pearson r           : {label['pearson_r']:.3f}")
    print(f"  Exact label accuracy: {label['exact_label_accuracy']:.3f} "
          f"({round(label['exact_label_accuracy'] * 8)}/8)")
    cont = summary["continuous_gamma_diagnostics"]
    print(f"  Continuous MAE      : {cont['mean_absolute_error']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
