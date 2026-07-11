#!/usr/bin/env python3
"""Run the paper-aligned dynamic-obstacle Safe Panda experiment."""

from dataclasses import asdict
import json

from lampc_cbf.dynamic_obstacle_demo import run_dynamic_obstacle_demo


if __name__ == "__main__":
    print(json.dumps(asdict(run_dynamic_obstacle_demo()), indent=2))
