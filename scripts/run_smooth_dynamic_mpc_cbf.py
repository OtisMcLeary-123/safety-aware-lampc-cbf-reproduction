#!/usr/bin/env python3
"""Run one continuous B-spline MPC-CBF experiment."""

from dataclasses import asdict
import json

from lampc_cbf.smooth_dynamic_demo import run_smooth_dynamic_demo


if __name__ == "__main__":
    print(json.dumps(asdict(run_smooth_dynamic_demo()), indent=2))
