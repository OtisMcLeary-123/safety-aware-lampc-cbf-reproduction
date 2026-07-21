#!/usr/bin/env python3
"""Run the paired collision-cone liveness ablation."""

from __future__ import annotations

import argparse
import json

from lampc_cbf.collision_cone_liveness_ablation import (
    CollisionConeLivenessAblationConfig,
    run_collision_cone_liveness_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument(
        "--output-dir",
        default="artifacts/collision_cone_liveness_ablation_20",
    )
    args = parser.parse_args()
    summary = run_collision_cone_liveness_ablation(
        CollisionConeLivenessAblationConfig(
            episodes=args.episodes,
            workers=args.workers,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
