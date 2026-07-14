#!/usr/bin/env python3
"""Run one preregistered remediation stage and print its safety gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lampc_cbf.remediation_benchmark import (
    RemediationBenchmarkConfig,
    run_remediation_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ablation", "development", "confirmatory"), default="ablation")
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override the stage count for deterministic smoke grids.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--feedback-latency", type=float, default=0.4971896839560941)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prerequisite-summary", default=None)
    args = parser.parse_args()
    counts = {"ablation": 20, "development": 100, "confirmatory": 500}
    episodes = counts[args.stage] if args.episodes is None else args.episodes
    if episodes < 1:
        raise SystemExit("episodes must be positive")
    prerequisite_defaults = {
        "development": "artifacts/remediation_ablation_20/summary.json",
        "confirmatory": "artifacts/remediation_development_100/summary.json",
    }
    if args.stage != "ablation":
        prerequisite = Path(
            args.prerequisite_summary or prerequisite_defaults[args.stage]
        )
        if not prerequisite.is_file():
            raise SystemExit(f"refusing {args.stage}: prerequisite summary is missing: {prerequisite}")
        prior = json.loads(prerequisite.read_text(encoding="utf-8"))
        if prior.get("gate", {}).get("passed") is not True:
            raise SystemExit(f"refusing {args.stage}: prerequisite gate did not pass: {prerequisite}")
    summary = run_remediation_benchmark(
        RemediationBenchmarkConfig(
            episodes=episodes,
            workers=args.workers,
            max_steps=args.max_steps,
            feedback_latency=args.feedback_latency,
            output_dir=args.output_dir or f"artifacts/remediation_{args.stage}_{episodes}",
        )
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
