#!/usr/bin/env python3
"""Smoke-test Hugging Face structured language-to-gamma inference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
from time import sleep

from lampc_cbf.hf_llm import HFLLMConfig, HuggingFaceGammaMapper


CASES = (
    "Keep as far away from the obstacle as possible; be maximally cautious.",
    "Be careful and maintain a generous safety distance.",
    "Use a normal balance between completing the task and obstacle clearance.",
    "Move efficiently; a modest safety margin is enough.",
    "Prioritize speed and use only the minimum required obstacle margin.",
)


def main() -> int:
    defaults = HFLLMConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--provider", default=defaults.provider)
    parser.add_argument("--token-file", default=defaults.token_path)
    parser.add_argument("--output-dir", default="artifacts/hf_llm_smoke")
    parser.add_argument("--inter-request-delay", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        HFLLMConfig(),
        model=args.model,
        provider=args.provider,
        token_path=args.token_file,
        cache_path=None,
    )
    mapper = HuggingFaceGammaMapper(config)
    decisions = []
    for index, case in enumerate(CASES):
        if index:
            sleep(args.inter_request_delay)
        decisions.append(mapper.infer_gamma(case))
    gammas = [decision.gamma for decision in decisions]
    levels = [decision.safety_level for decision in decisions]
    summary = {
        "model": config.model,
        "provider": config.provider,
        "cases": [
            {"instruction": case, "decision": decision.as_dict()}
            for case, decision in zip(CASES, decisions)
        ],
        "schema_success_rate": mean(not item.fallback_used for item in decisions),
        "mean_latency_seconds": mean(item.latency_seconds for item in decisions),
        "max_latency_seconds": max(item.latency_seconds for item in decisions),
        "gamma_monotone_with_less_cautious_language": all(
            left <= right for left, right in zip(gammas, gammas[1:])
        ),
        "level_monotone_with_less_cautious_language": all(
            left <= right for left, right in zip(levels, levels[1:])
        ),
        "token_recorded": False,
    }
    (output_dir / "smoke_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["schema_success_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
