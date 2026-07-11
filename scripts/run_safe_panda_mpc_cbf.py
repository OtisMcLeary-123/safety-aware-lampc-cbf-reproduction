#!/usr/bin/env python3
"""Run and render the deterministic Safe Panda MPC-CBF demo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from lampc_cbf.demo import DemoConfig, run_safe_panda_mpc_cbf_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument(
        "--output-dir", default="artifacts/mpc_cbf_demo/gamma_0.10"
    )
    args = parser.parse_args()
    result = run_safe_panda_mpc_cbf_demo(
        DemoConfig(
            gamma=args.gamma,
            seed=args.seed,
            max_steps=args.max_steps,
            render_stride=args.render_stride,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.reached_goal and not result.collision else 2


if __name__ == "__main__":
    raise SystemExit(main())
