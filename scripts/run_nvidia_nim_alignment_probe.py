#!/usr/bin/env python3
"""Probe NIM-Q1 JSON compatibility without using a benchmark query."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from lampc_cbf.language_alignment import (
    NvidiaNIMAlignmentConfig,
    NvidiaNIMBlindAlignmentMapper,
)


PROBE_INSTRUCTION = (
    "Synthetic formatter compatibility check only: assign gamma exactly 0.5."
)


def main() -> int:
    defaults = NvidiaNIMAlignmentConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--token-file", default=defaults.token_path)
    parser.add_argument(
        "--output-dir", default="artifacts/language_alignment_nim_probe"
    )
    args = parser.parse_args()
    config = replace(
        defaults,
        model=args.model,
        token_path=args.token_file,
    )
    prediction = NvidiaNIMBlindAlignmentMapper(config).predict(PROBE_INSTRUCTION)
    payload = {
        "protocol": {
            "name": "NIM-Q1 non-benchmark compatibility probe",
            "model": config.model,
            "provider": "nvidia-nim",
            "endpoint": config.endpoint,
            "benchmark_query_used": False,
            "expected_gamma": 0.5,
            "thinking_enabled": config.enable_thinking,
            "token_recorded": False,
        },
        "passed": prediction.gamma == 0.5,
        "prediction": prediction.as_dict(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nim_q1_probe.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
