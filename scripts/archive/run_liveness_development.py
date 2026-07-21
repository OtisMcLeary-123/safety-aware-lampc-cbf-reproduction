#!/usr/bin/env python3
"""Run the gated, physics-budgeted 100-episode C3BF development stage."""

from __future__ import annotations

import argparse
import json

from lampc_cbf.liveness_development import (
    LivenessDevelopmentConfig,
    run_liveness_development,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="artifacts/collision_cone_liveness_development_100",
    )
    parser.add_argument(
        "--prerequisite-summary",
        default="artifacts/collision_cone_liveness_ablation_20/summary.json",
    )
    args = parser.parse_args()
    summary = run_liveness_development(
        LivenessDevelopmentConfig(
            episodes=args.episodes,
            workers=args.workers,
            output_dir=args.output_dir,
            prerequisite_summary=args.prerequisite_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
