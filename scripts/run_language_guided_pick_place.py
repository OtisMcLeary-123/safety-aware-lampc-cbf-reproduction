#!/usr/bin/env python3
"""Recreate the paper-style language-guided blue-to-red cube figure."""

from __future__ import annotations

from dataclasses import asdict
import json

from lampc_cbf.build_l_demo import BuildLDemoConfig, run_build_l_mpc_cbf_demo
from lampc_cbf.hf_llm import HuggingFaceGammaMapper


USER_INSTRUCTION = (
    "Safely pick up the blue cube and put it on the red cube. "
    "Keep a generous distance from the moving obstacle."
)


def main() -> int:
    decision = HuggingFaceGammaMapper().infer_gamma(USER_INSTRUCTION)
    result = run_build_l_mpc_cbf_demo(
        BuildLDemoConfig(
            gamma=decision.gamma,
            seed=7,
            max_move_steps=160,
            cube_indices=(0,),
            place_blue_on_red=True,
            dynamic_obstacle=True,
            obstacle_radius=0.055,
            obstacle_velocity=(0.0, -0.01, 0.0),
            render_stride=2,
            user_instruction=USER_INSTRUCTION,
            llm_model=decision.model,
            llm_safety_level=decision.safety_level,
            llm_latency_seconds=decision.latency_seconds,
            llm_fallback_used=decision.fallback_used,
            output_dir="artifacts/language_guided_pick_place",
        )
    )
    payload = {
        "llm_decision": decision.as_dict(),
        "result": asdict(result),
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.success and not decision.fallback_used else 2


if __name__ == "__main__":
    raise SystemExit(main())
