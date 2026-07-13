#!/usr/bin/env python3
"""Run the blinded eight-query language-alignment smoke benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
from time import sleep

from lampc_cbf.language_alignment import (
    AlignmentConfig,
    BlindAlignmentMapper,
    PUBLISHED_ALIGNMENT_CASES,
    evaluate_published_smoke,
    result_rows,
)


def main() -> int:
    defaults = AlignmentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--provider", default=defaults.provider)
    parser.add_argument("--token-file", default=defaults.token_path)
    parser.add_argument(
        "--output-dir", default="artifacts/language_alignment_smoke"
    )
    parser.add_argument("--inter-request-delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.inter_request_delay < 0.0:
        parser.error("--inter-request-delay must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        defaults,
        model=args.model,
        provider=args.provider,
        token_path=args.token_file,
    )
    mapper = BlindAlignmentMapper(config)
    predictions = []
    for index, case in enumerate(PUBLISHED_ALIGNMENT_CASES):
        if index:
            sleep(args.inter_request_delay)
        predictions.append(mapper.predict(case.instruction))

    payload = {
        "protocol": {
            "name": "N2-A1 blinded published-example smoke test",
            "model": config.model,
            "provider": config.provider,
            "temperature": config.temperature,
            "seed": config.seed,
            "paper_domain": "0 < gamma <= 1",
            "controller_isolation": True,
            "target_examples_in_prompt": False,
            "token_recorded": False,
        },
        "metrics": evaluate_published_smoke(predictions),
        "latency": {
            "mean_seconds": mean(item.latency_seconds for item in predictions),
            "max_seconds": max(item.latency_seconds for item in predictions),
        },
        "cases": result_rows(predictions),
    }
    output_path = output_dir / "alignment_smoke_results.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
