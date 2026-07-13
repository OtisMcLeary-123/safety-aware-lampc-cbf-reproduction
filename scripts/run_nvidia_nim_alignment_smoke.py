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
    alignment_prediction_from_dict,
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
        "--timeout-seconds", type=float, default=defaults.timeout_seconds
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only after validating the existing checkpoint",
    )
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
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "alignment_smoke_checkpoint.json"
    failure_path = output_dir / "alignment_smoke_failure.json"
    mapper = NvidiaNIMBlindAlignmentMapper(config)
    predictions = []
    if args.resume:
        if not checkpoint_path.is_file():
            parser.error("--resume requires an existing checkpoint")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_header = {
            "model": config.model,
            "provider": "nvidia-nim",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "thinking_enabled": config.enable_thinking,
        }
        actual_header = {
            key: checkpoint.get(key) for key in expected_header
        }
        if actual_header != expected_header:
            parser.error("checkpoint model/provider/max_tokens do not match")
        completed_rows = checkpoint.get("completed_cases")
        if not isinstance(completed_rows, list):
            parser.error("checkpoint completed_cases must be a list")
        for index, row in enumerate(completed_rows):
            if index >= len(PUBLISHED_ALIGNMENT_CASES):
                parser.error("checkpoint has too many completed cases")
            case = PUBLISHED_ALIGNMENT_CASES[index]
            if row.get("case_id") != case.case_id:
                parser.error("checkpoint case IDs are not a contiguous prefix")
            if row.get("instruction") != case.instruction:
                parser.error("checkpoint instruction does not match benchmark")
            predictions.append(
                alignment_prediction_from_dict(row.get("prediction", {}))
            )

    start_index = len(predictions)
    for index, case in enumerate(
        PUBLISHED_ALIGNMENT_CASES[start_index:], start=start_index
    ):
        if index > start_index:
            sleep(args.inter_request_delay)
        try:
            predictions.append(mapper.predict(case.instruction))
        except Exception as error:
            failure = {
                "model": config.model,
                "provider": "nvidia-nim",
                "failed_case_id": case.case_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_case_count": len(predictions),
                "completed_cases": result_rows(
                    predictions, PUBLISHED_ALIGNMENT_CASES[: len(predictions)]
                ),
                "token_recorded": False,
            }
            failure_path.write_text(
                json.dumps(failure, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise
        checkpoint = {
            "model": config.model,
            "provider": "nvidia-nim",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "thinking_enabled": config.enable_thinking,
            "completed_case_count": len(predictions),
            "pending_case_ids": [
                pending.case_id for pending in PUBLISHED_ALIGNMENT_CASES[index + 1 :]
            ],
            "completed_cases": result_rows(
                predictions, PUBLISHED_ALIGNMENT_CASES[: len(predictions)]
            ),
            "token_recorded": False,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    payload = {
        "protocol": {
            "name": "N2-A1 blinded published-example smoke test",
            "model": config.model,
            "provider": "nvidia-nim",
            "endpoint": config.endpoint,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "seed": config.seed,
            "paper_domain": "0 < gamma <= 1",
            "controller_isolation": True,
            "target_examples_in_prompt": False,
            "thinking_enabled": config.enable_thinking,
            "timeout_seconds": config.timeout_seconds,
            "max_tokens": config.max_tokens,
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
    output_path = output_dir / "alignment_smoke_results.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checkpoint_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
