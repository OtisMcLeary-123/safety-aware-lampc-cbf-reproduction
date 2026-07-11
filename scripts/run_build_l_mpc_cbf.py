#!/usr/bin/env python3
"""Run the sequential Safe Panda Build-L MPC-CBF demonstration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from lampc_cbf.build_l_demo import BuildLDemoConfig, run_build_l_mpc_cbf_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/build_l_mpc_cbf")
    args = parser.parse_args()
    result = run_build_l_mpc_cbf_demo(
        BuildLDemoConfig(gamma=args.gamma, seed=args.seed, output_dir=args.output_dir)
    )
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())

