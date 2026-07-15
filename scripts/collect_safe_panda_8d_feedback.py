#!/usr/bin/env python3
"""Collect fresh feedback records for the Safe Panda 8-D profile.

This script is intentionally separate from the existing candidate checkpoint.
It is not run during local validation because it spends provider requests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep

from lampc_cbf.contextual_gamma import ContextualGammaConfig, ContextualNvidiaNIMGammaMapper
from lampc_cbf.paired_benchmark import load_feedback_checkpoint
from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.table4_scenarios import contextual_feedback_context, load_scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", default="configs/table4_candidate_50_scenarios.json")
    parser.add_argument("--setup-file", default="configs/safe_panda_8d_double_integrator_50_setup.json")
    parser.add_argument("--provider-manifest", default="configs/paper_fidelity_nvidia_nim_llama31.json")
    parser.add_argument("--checkpoint", default="artifacts/safe_panda_8d_double_integrator_50_contextual_nim_llama31/feedback_decisions_checkpoint.json")
    parser.add_argument("--request-interval-seconds", type=float, default=1.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.request_interval_seconds < 0.0:
        parser.error("request interval must be non-negative")
    setup = json.loads(Path(args.setup_file).read_text(encoding="utf-8"))
    scenarios = load_scenarios(args.scenario_file)
    manifest = PaperFidelityManifest.load(args.provider_manifest)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.no_resume:
        decisions = load_feedback_checkpoint(checkpoint, episodes=50, paper_manifest=manifest)
    else:
        decisions = []
    mapper = ContextualNvidiaNIMGammaMapper(
        ContextualGammaConfig(
            model=manifest.required_model_family,
            timeout_seconds=manifest.llm_timeout_seconds,
            max_tokens=manifest.llm_max_tokens,
        )
    )
    for index in range(len(decisions), len(scenarios)):
        decision = mapper.infer_gamma(setup["llm"]["feedback_query"], contextual_feedback_context(scenarios[index]))
        if decision.fallback_used:
            raise RuntimeError(f"feedback collection failed at episode {index + 1}")
        decisions.append(decision)
        checkpoint.write_text(json.dumps([item.as_dict() for item in decisions], indent=2), encoding="utf-8")
        print(f"[safe-panda-8d] collected {index + 1}/50", flush=True)
        if index + 1 < len(scenarios):
            sleep(args.request_interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
