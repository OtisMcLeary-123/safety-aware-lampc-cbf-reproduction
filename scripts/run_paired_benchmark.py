#!/usr/bin/env python3
"""Run the staged paired benchmark for the complete safety stack."""

from __future__ import annotations

import argparse
import json

from lampc_cbf.hf_llm import HuggingFaceGammaMapper
from lampc_cbf.paired_benchmark import PairedBenchmarkConfig, run_paired_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "development", "confirmatory"),
        default="smoke",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override the preregistered stage size (12/100/500).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument(
        "--feedback-schedule-mode",
        choices=("ttc", "elapsed_time"),
        default="ttc",
    )
    parser.add_argument("--feedback-ttc-threshold", type=float, default=1.5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    stage_episodes = {"smoke": 12, "development": 100, "confirmatory": 500}
    episodes = args.episodes or stage_episodes[args.stage]
    output_dir = args.output_dir or f"artifacts/paired_benchmark_protocol_v2_{args.stage}_{episodes}"

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
            episodes=episodes,
            workers=args.workers,
            max_steps=args.max_steps,
            output_dir=output_dir,
            resume=not args.no_resume,
            feedback_schedule_mode=args.feedback_schedule_mode,
            feedback_ttc_threshold=args.feedback_ttc_threshold,
        ),
    )
    print(json.dumps(summary["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
