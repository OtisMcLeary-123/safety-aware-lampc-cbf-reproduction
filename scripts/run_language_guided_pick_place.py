#!/usr/bin/env python3
"""Recreate the paper-style language-guided blue-to-red cube figure."""

from __future__ import annotations

from dataclasses import asdict
import json

from lampc_cbf.build_l_demo import BuildLDemoConfig, run_build_l_mpc_cbf_demo
from lampc_cbf.language_dsl import HuggingFaceSafeNarratePlanner


USER_INSTRUCTION = (
    "Safely pick up the blue cube and put it on the red cube. "
    "Keep a generous distance from the moving obstacle."
)


def main() -> int:
    planner = HuggingFaceSafeNarratePlanner()
    result = run_build_l_mpc_cbf_demo(
        BuildLDemoConfig(
            seed=7,
            max_move_steps=160,
            cube_indices=(0,),
            place_blue_on_red=True,
            dynamic_obstacle=True,
            obstacle_radius=0.055,
            obstacle_velocity=(0.0, -0.01, 0.0),
            render_stride=2,
            user_instruction=USER_INSTRUCTION,
            output_dir="artifacts/language_guided_pick_place",
        ),
        language_planner=planner,
    )
    payload = {"result": asdict(result)}
    print(json.dumps(payload, indent=2))
    return 0 if result.success and result.language_od_fallbacks == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
