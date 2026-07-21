#!/usr/bin/env python3
"""Render paired collision and CBF-avoidance examples for the hard scene."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from lampc_cbf.smooth_dynamic_demo import (
    SmoothDynamicConfig,
    run_smooth_dynamic_demo,
)


def main() -> int:
    root = Path("artifacts/hard_scene_study")
    common = {
        "delta_u_weight": 2.0,
        "seed": 1002,
        "max_steps": 140,
        "reference_speed": 0.08,
        "reference_mode": "straight",
        "obstacle_start_offset": (0.06, 0.44, 0.0),
        "obstacle_velocity": (0.0, -0.10, 0.0),
        "save_animation": True,
        "save_plots": True,
        "save_metrics": True,
        "render_stride": 2,
    }
    variants = {
        "distance_collision": {"safety_mode": "distance", "gamma": 0.15},
        "cbf_gamma_002_avoidance": {"safety_mode": "cbf", "gamma": 0.02},
    }
    results = {}
    for name, variant in variants.items():
        print(f"[example] rendering {name}", flush=True)
        result = run_smooth_dynamic_demo(
            SmoothDynamicConfig(
                **common,
                **variant,
                output_dir=str(root / name),
            )
        )
        results[name] = asdict(result)
    (root / "representative_examples.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
