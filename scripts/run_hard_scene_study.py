#!/usr/bin/env python3
"""Run the paper-style 50-episode hard-scene feedback study."""

from __future__ import annotations

import json

from lampc_cbf.hard_scene_study import run_hard_scene_study
from lampc_cbf.hf_llm import HuggingFaceGammaMapper


def main() -> int:
    mapper = HuggingFaceGammaMapper()
    initial = mapper.infer_gamma(
        "Keep a generous distance from the moving obstacle and be cautious."
    )
    feedback = mapper.infer_gamma(
        "Watch out! I think it's going to crash soon.",
        current_gamma=0.15,
        feedback=True,
    )
    summary = run_hard_scene_study(initial, feedback)
    print(json.dumps(summary["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
