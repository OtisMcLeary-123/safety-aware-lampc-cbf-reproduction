#!/usr/bin/env python3
"""Run N2-A1 against NVIDIA NIM after its compatibility probe passes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
from time import sleep

from lampc_cbf.language_alignment import (
    NvidiaNIMAlignmentConfig,
    NvidiaNIMBlindAlignmentMapper,
    PUBLISHED_ALIGNMENT_CASES,
    evaluate_published_smoke,
    result_rows,
)


def main() -> int:
    defaults = NvidiaNIMAlignmentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--token-file", default=defaults.token_path)
    parser.add_argument(
        "--output-dir", default="artifacts/language_alignment_smoke_nvidia_nim"
    )
    parser.add_argument("--inter-request-delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.inter_request_delay < 0.0:
        parser.error("--inter-request-delay must be non-negative")

    config = replace(
        defaults,
        model=args.model,
        token_path=args.token_file,
    )
    mapper = NvidiaNIMBlindAlignmentMapper(config)
    predictions = []
    for index, case in enumerate(PUBLISHED_ALIGNMENT_CASES):
        if index:
            sleep(args.inter_request_delay)
        predictions.append(mapper.predict(case.instruction))

    payload = {
        "protocol": {
            "name": "N2-A1 blinded published-example smoke test",
            "model": config.model,
            "provider": "nvidia-nim",
            "endpoint": config.endpoint,
            "temperature": config.temperature,
            "seed": config.seed,
            "paper_domain": "0 < gamma <= 1",
            "controller_isolation": True,
            "target_examples_in_prompt": False,
            "thinking_enabled": config.enable_thinking,
            "server_side_json_schema": False,
            "strict_local_validation": True,
            "token_recorded": False,
        },
        "metrics": evaluate_published_smoke(predictions),
        "latency": {
            "mean_seconds": mean(item.latency_seconds for item in predictions),
            "max_seconds": max(item.latency_seconds for item in predictions),
        },
        "cases": result_rows(predictions),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "alignment_smoke_results.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
