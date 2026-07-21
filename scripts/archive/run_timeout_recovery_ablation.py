#!/usr/bin/env python3
"""Run the paired timeout-recovery ablation."""

from __future__ import annotations

import argparse
import json

from lampc_cbf.timeout_recovery_benchmark import (
    TimeoutRecoveryBenchmarkConfig,
    run_timeout_recovery_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument(
        "--output-dir", default="artifacts/timeout_recovery_ablation_20"
    )
    args = parser.parse_args()
    summary = run_timeout_recovery_benchmark(
        TimeoutRecoveryBenchmarkConfig(
            episodes=args.episodes,
            workers=args.workers,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
