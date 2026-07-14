#!/usr/bin/env python3
"""Run the paired C3BF versus Cartesian DPCBF ablation."""

from __future__ import annotations

import argparse
import json

from lampc_cbf.dpcbf_ablation import DPCBFAblationConfig, run_dpcbf_ablation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--output-dir", default="artifacts/dpcbf_ablation_20")
    args = parser.parse_args()
    summary = run_dpcbf_ablation(
        DPCBFAblationConfig(
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
