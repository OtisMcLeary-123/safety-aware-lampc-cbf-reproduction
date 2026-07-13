#!/usr/bin/env python3
"""Run the 500-condition paired benchmark for the complete safety stack."""

from __future__ import annotations

import argparse
import json

from lampc_cbf.hf_llm import HuggingFaceGammaMapper
from lampc_cbf.paired_benchmark import PairedBenchmarkConfig, run_paired_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--output-dir", default="artifacts/paired_benchmark_500")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    feedback = HuggingFaceGammaMapper().infer_gamma(
        "Watch out! I think the robot is going to crash soon. Increase clearance now.",
        current_gamma=0.15,
        feedback=True,
    )
    if feedback.fallback_used:
        raise RuntimeError(
            f"validated LLM feedback is required; inference failed with {feedback.error_type}"
        )
    summary = run_paired_benchmark(
        feedback,
        PairedBenchmarkConfig(
            episodes=args.episodes,
            workers=args.workers,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            resume=not args.no_resume,
        ),
    )
    print(json.dumps(summary["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
