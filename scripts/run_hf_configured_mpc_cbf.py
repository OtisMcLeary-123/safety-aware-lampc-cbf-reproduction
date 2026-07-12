#!/usr/bin/env python3
"""Use one Hugging Face decision to configure and run MPC-CBF safely."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from lampc_cbf.hf_llm import HFLLMConfig, HuggingFaceGammaMapper
from lampc_cbf.smooth_dynamic_demo import (
    SmoothDynamicConfig,
    run_smooth_dynamic_demo,
)


def main() -> int:
    defaults = HFLLMConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruction",
        default="Keep a generous distance from the moving obstacle and be cautious.",
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--provider", default=defaults.provider)
    parser.add_argument("--token-file", default=defaults.token_path)
    parser.add_argument("--output-dir", default="artifacts/hf_llm_control_demo")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_config = replace(
        defaults,
        model=args.model,
        provider=args.provider,
        token_path=args.token_file,
    )
    decision = HuggingFaceGammaMapper(llm_config).infer_gamma(args.instruction)
    demo_result = run_smooth_dynamic_demo(
        SmoothDynamicConfig(
            gamma=decision.gamma,
            delta_u_weight=5.0,
            save_animation=True,
            output_dir=str(output_dir / "simulation"),
        )
    )
    manifest = {
        "instruction": args.instruction,
        "llm_decision": decision.as_dict(),
        "controller_gamma": decision.gamma,
        "fallback_policy": {
            "gamma": llm_config.fallback_gamma,
            "LLM_inside_40ms_control_loop": False,
            "invalid_or_timeout_output_bypasses_CBF": False,
        },
        "simulation_result": asdict(demo_result),
        "token_recorded": False,
    }
    (output_dir / "llm_control_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
