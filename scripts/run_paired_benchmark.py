#!/usr/bin/env python3
"""Run the staged paired benchmark for the complete safety stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lampc_cbf.hf_llm import GammaDecision, HFLLMConfig, HuggingFaceGammaMapper
from lampc_cbf.nvidia_nim_gamma import NvidiaNIMGammaConfig, NvidiaNIMGammaMapper
from lampc_cbf.paired_benchmark import PairedBenchmarkConfig, run_paired_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "development", "confirmatory"),
        default="smoke",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override the preregistered stage size (12/100/500).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override the frozen stage budget (140 smoke/development, 220 confirmatory).",
    )
    parser.add_argument(
        "--feedback-schedule-mode",
        choices=("ttc", "elapsed_time"),
        default="ttc",
    )
    parser.add_argument("--feedback-ttc-threshold", type=float, default=1.5)
    parser.add_argument(
        "--llm-provider", choices=("nvidia-nim", "hugging-face"),
        default="nvidia-nim",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=3.0)
    parser.add_argument(
        "--feedback-decision-json",
        default=None,
        help=(
            "Use a JSON list of previously validated per-episode GammaDecision "
            "objects (or a benchmark summary containing feedback_decisions)."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    stage_episodes = {"smoke": 12, "development": 100, "confirmatory": 500}
    episodes = args.episodes or stage_episodes[args.stage]
    max_steps = args.max_steps or (220 if args.stage == "confirmatory" else 140)
    output_dir = args.output_dir or f"artifacts/paired_benchmark_protocol_v5_{args.stage}_{episodes}"

    if args.feedback_decision_json:
        decision_payload = json.loads(
            Path(args.feedback_decision_json).read_text(encoding="utf-8")
        )
        if isinstance(decision_payload, dict) and "feedback_decisions" in decision_payload:
            decision_payload = decision_payload["feedback_decisions"]
        if not isinstance(decision_payload, list):
            raise ValueError(
                "feedback decision file must contain one validated decision per episode"
            )
        feedback_decisions = [
            GammaDecision(raw_response=None, **item)
            if "raw_response" not in item
            else GammaDecision(**item)
            for item in decision_payload
        ]
    elif args.llm_provider == "nvidia-nim":
        defaults = NvidiaNIMGammaConfig()
        mapper = NvidiaNIMGammaMapper(
            NvidiaNIMGammaConfig(
                model=args.llm_model or defaults.model,
                timeout_seconds=args.llm_timeout,
                cache_path=None,
            )
        )
    else:
        defaults = HFLLMConfig()
        mapper = HuggingFaceGammaMapper(
            HFLLMConfig(
                model=args.llm_model or defaults.model,
                timeout_seconds=args.llm_timeout,
                cache_path=None,
            )
        )
    if not args.feedback_decision_json:
        feedback_decisions = []
        for episode in range(episodes):
            feedback = mapper.infer_gamma(
                "Watch out! I think the robot is going to crash soon. Increase clearance now.",
                current_gamma=0.15,
                feedback=True,
            )
            if feedback.fallback_used:
                raise RuntimeError(
                    "validated per-episode LLM feedback is required; "
                    f"episode {episode} failed with {feedback.error_type}"
                )
            feedback_decisions.append(feedback)
            print(
                f"[feedback] collected {episode + 1}/{episodes}; "
                f"latency={feedback.latency_seconds:.3f}s",
                flush=True,
            )
    if len(feedback_decisions) != episodes:
        raise ValueError("feedback decision count must match episode count")
    summary = run_paired_benchmark(
        feedback_decisions,
        PairedBenchmarkConfig(
            stage=args.stage,
            episodes=episodes,
            workers=args.workers,
            max_steps=max_steps,
            output_dir=output_dir,
            resume=not args.no_resume,
            feedback_schedule_mode=args.feedback_schedule_mode,
            feedback_ttc_threshold=args.feedback_ttc_threshold,
        ),
    )
    print(json.dumps(summary["methods"], indent=2))
    print(json.dumps({"efficacy_gate": summary["efficacy_gate"]}, indent=2))
    print(json.dumps({"formal_contract_gate": summary["formal_contract_gate"]}, indent=2))
    if args.stage == "confirmatory" and summary["efficacy_gate"]["passed"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
