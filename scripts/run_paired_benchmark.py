#!/usr/bin/env python3
"""Run the staged paired benchmark for the complete safety stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lampc_cbf.hf_llm import GammaDecision, HFLLMConfig, HuggingFaceGammaMapper
from lampc_cbf.nvidia_nim_gamma import NvidiaNIMGammaConfig, NvidiaNIMGammaMapper
from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.paired_benchmark import (
    PairedBenchmarkConfig,
    load_feedback_checkpoint,
    run_paired_benchmark,
)


DEFAULT_FEEDBACK_PROMPT = (
    "Watch out! I think the robot is going to crash soon. Increase clearance now."
)


def _decision_from_item(item: object) -> GammaDecision:
    if not isinstance(item, dict):
        raise ValueError("feedback decision entries must be JSON objects")
    return (
        GammaDecision(raw_response=None, **item)
        if "raw_response" not in item
        else GammaDecision(**item)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "development", "paper-replication", "confirmatory"),
        default="smoke",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override the staged size (12/100/50/500 where permitted).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help=(
            "Override the frozen stage budget (140 smoke/development, "
            "manifest-controlled paper replication, 220 confirmatory)."
        ),
    )
    parser.add_argument(
        "--feedback-schedule-mode",
        choices=("ttc", "elapsed_time"),
        default=None,
    )
    parser.add_argument("--feedback-ttc-threshold", type=float, default=1.5)
    parser.add_argument(
        "--llm-provider", choices=("nvidia-nim", "hugging-face"),
        default="nvidia-nim",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument(
        "--feedback-decision-json",
        default=None,
        help=(
            "Use a JSON list of previously validated per-episode GammaDecision "
            "objects (or a benchmark summary containing feedback_decisions)."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Validated paper-fidelity manifest. Required by the "
            "paper-replication stage; defaults to configs/paper_fidelity.json."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    paper_manifest = None
    if args.stage == "paper-replication":
        paper_manifest = PaperFidelityManifest.load(
            args.manifest or "configs/paper_fidelity.json"
        )
        if args.feedback_decision_json is None and not paper_manifest.model_substitution:
            parser.error(
                "paper-replication requires 50 validated GPT-4o/OpenAI decisions "
                "via --feedback-decision-json"
            )
        if args.episodes is not None and args.episodes != paper_manifest.episodes:
            parser.error("paper-replication episode count is locked by the manifest")
        if args.max_steps is not None and args.max_steps != paper_manifest.max_steps:
            parser.error("paper-replication step budget is locked by the manifest")
        if (
            args.feedback_schedule_mode is not None
            and args.feedback_schedule_mode != paper_manifest.feedback_schedule_mode
        ):
            parser.error("paper-replication feedback schedule is locked by the manifest")
        episodes = paper_manifest.episodes
        output_dir = args.output_dir or paper_manifest.output_dir
        config_kwargs = paper_manifest.benchmark_kwargs()
        if paper_manifest.model_substitution:
            if args.llm_provider != paper_manifest.required_provider:
                parser.error("LLM provider must match the substitution manifest")
            if (
                args.llm_model is not None
                and args.llm_model != paper_manifest.required_model_family
            ):
                parser.error("LLM model must match the substitution manifest")
    else:
        if args.manifest is not None:
            parser.error("--manifest is only valid for the paper-replication stage")
        stage_episodes = {"smoke": 12, "development": 100, "confirmatory": 500}
        episodes = args.episodes or stage_episodes[args.stage]
        max_steps = args.max_steps or (
            220 if args.stage == "confirmatory" else 140
        )
        output_dir = args.output_dir or (
            f"artifacts/paired_benchmark_protocol_v5_{args.stage}_{episodes}"
        )
        config_kwargs = {
            "stage": args.stage,
            "episodes": episodes,
            "max_steps": max_steps,
            "feedback_schedule_mode": args.feedback_schedule_mode or "ttc",
        }

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
        feedback_decisions = [_decision_from_item(item) for item in decision_payload]
        if paper_manifest is not None:
            invalid = [
                decision
                for decision in feedback_decisions
                if not paper_manifest.accepts_feedback_decision(
                    model=decision.model,
                    provider=decision.provider,
                    cache_hit=decision.cache_hit,
                )
            ]
            if invalid:
                parser.error(
                    "paper-replication decisions must be uncached GPT-4o/OpenAI "
                    "responses matching the manifest"
                )
    elif args.llm_provider == "nvidia-nim":
        defaults = NvidiaNIMGammaConfig()
        mapper = NvidiaNIMGammaMapper(
            NvidiaNIMGammaConfig(
                model=(
                    args.llm_model
                    or (
                        paper_manifest.required_model_family
                        if paper_manifest is not None
                        and paper_manifest.model_substitution
                        else defaults.model
                    )
                ),
                timeout_seconds=(
                    args.llm_timeout
                    or (
                        paper_manifest.llm_timeout_seconds
                        if paper_manifest is not None
                        and paper_manifest.model_substitution
                        else defaults.timeout_seconds
                    )
                ),
                max_tokens=(
                    paper_manifest.llm_max_tokens
                    if paper_manifest is not None
                    and paper_manifest.model_substitution
                    else defaults.max_tokens
                ),
                cache_path=None,
                guided_json_enabled=(
                    paper_manifest.llm_guided_json_enabled
                    if paper_manifest is not None
                    and paper_manifest.model_substitution
                    else defaults.guided_json_enabled
                ),
                enable_thinking=(
                    paper_manifest.llm_enable_thinking
                    if paper_manifest is not None
                    and paper_manifest.model_substitution
                    else defaults.enable_thinking
                ),
            )
        )
    else:
        defaults = HFLLMConfig()
        mapper = HuggingFaceGammaMapper(
            HFLLMConfig(
                model=args.llm_model or defaults.model,
                timeout_seconds=args.llm_timeout or 3.0,
                cache_path=None,
            )
        )
    if not args.feedback_decision_json:
        feedback_checkpoint = Path(output_dir) / "feedback_decisions_checkpoint.json"
        feedback_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_resume and feedback_checkpoint.exists():
            feedback_decisions = load_feedback_checkpoint(
                feedback_checkpoint,
                episodes=episodes,
                paper_manifest=paper_manifest,
            )
            print(
                f"[feedback] resumed {len(feedback_decisions)}/{episodes} "
                f"from {feedback_checkpoint}",
                flush=True,
            )
        else:
            feedback_decisions = []
        for episode in range(len(feedback_decisions), episodes):
            feedback = mapper.infer_gamma(
                str(config_kwargs.get("feedback_prompt", DEFAULT_FEEDBACK_PROMPT)),
                current_gamma=0.15,
                feedback=True,
            )
            if feedback.fallback_used:
                raise RuntimeError(
                    "validated per-episode LLM feedback is required; "
                    f"episode {episode} failed with {feedback.error_type}"
                )
            feedback_decisions.append(feedback)
            feedback_checkpoint.write_text(
                json.dumps(
                    [decision.as_dict() for decision in feedback_decisions],
                    indent=2,
                ),
                encoding="utf-8",
            )
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
            workers=args.workers,
            output_dir=output_dir,
            resume=not args.no_resume,
            feedback_ttc_threshold=args.feedback_ttc_threshold,
            **config_kwargs,
        ),
    )
    print(json.dumps(summary["methods"], indent=2))
    print(json.dumps({"efficacy_gate": summary["efficacy_gate"]}, indent=2))
    print(json.dumps({"formal_contract_gate": summary["formal_contract_gate"]}, indent=2))
    if (
        args.stage in {"paper-replication", "confirmatory"}
        and summary["efficacy_gate"]["passed"] is not True
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
