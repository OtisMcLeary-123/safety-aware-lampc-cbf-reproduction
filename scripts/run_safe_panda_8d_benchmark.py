#!/usr/bin/env python3
"""Run the Safe Panda 8-D paper-state profile without provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lampc_cbf.hf_llm import GammaDecision
from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.safe_panda_8d_double_integrator_benchmark import run_safe_panda_8d_double_integrator_benchmark
from lampc_cbf.table4_scenarios import load_scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", default="configs/table4_candidate_50_scenarios.json")
    parser.add_argument("--setup-file", default="configs/safe_panda_8d_double_integrator_50_setup.json")
    parser.add_argument("--provider-manifest", default="configs/paper_fidelity_nvidia_nim_llama31.json")
    parser.add_argument("--feedback-checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/safe_panda_8d_double_integrator_50_benchmark")
    args = parser.parse_args()
    setup = json.loads(Path(args.setup_file).read_text(encoding="utf-8"))
    scenarios = load_scenarios(args.scenario_file)
    manifest = PaperFidelityManifest.load(args.provider_manifest)
    payload = json.loads(Path(args.feedback_checkpoint).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 50:
        raise ValueError("Safe Panda 8D benchmark requires exactly 50 decisions")
    decisions = []
    for item in payload:
        record = dict(item)
        record.setdefault("raw_response", None)
        decision = GammaDecision(**record)
        if decision.fallback_used or decision.cache_hit or not manifest.accepts_feedback_decision(model=decision.model, provider=decision.provider, cache_hit=decision.cache_hit):
            raise ValueError("checkpoint is not valid for the selected provider manifest")
        decisions.append(decision)
    summary = run_safe_panda_8d_double_integrator_benchmark(scenarios, decisions, output_dir=args.output_dir, seed_base=int(setup["benchmark"]["seed_base"]), bootstrap_resamples=int(setup["benchmark"]["bootstrap_resamples"]))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
