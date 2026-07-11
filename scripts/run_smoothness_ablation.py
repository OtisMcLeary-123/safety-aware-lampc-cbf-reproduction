#!/usr/bin/env python3
"""Run the complete dynamic-obstacle trajectory-smoothness ablation."""

import json

from lampc_cbf.smoothness_ablation import run_smoothness_ablation


if __name__ == "__main__":
    print(json.dumps(run_smoothness_ablation(), indent=2))
