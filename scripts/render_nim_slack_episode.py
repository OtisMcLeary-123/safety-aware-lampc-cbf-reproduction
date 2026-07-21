#!/usr/bin/env python3
"""Render single core-scenario episodes under the NIM + soft-slack feedback arm.

Reproduces the exact per-episode kwargs assembled by
``core_scenarios.run_core_benchmark`` for the
``safe_panda_core_nim_soft_slack_feedback_v1`` profile, but with plot and
animation rendering enabled. The provider decision replays from the committed
checkpoint (cache hit), so no API request is made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lampc_cbf.core_scenarios import (
    load_frozen_instances,
    load_llm_feedback_manifest,
    load_plan,
    runner_kwargs,
)
from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo

MANIFEST = "configs/safe_panda_core_nim_soft_slack_feedback_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_ids", nargs="+", help="e.g. CS2-E09 CS1-E11")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the frozen step cap (e.g. 520 for an extended-horizon "
        "illustration). Values other than the frozen 260 deviate from the "
        "benchmark contract and are exploratory renders, not benchmark rows.",
    )
    parser.add_argument(
        "--goal-scale",
        type=float,
        default=1.0,
        help="scale the start-to-goal offset (e.g. 1.5 stretches the reach "
        "50%% farther along the same axis). Values != 1.0 deviate from the "
        "frozen instance and are exploratory renders, not benchmark rows.",
    )
    parser.add_argument(
        "--obstacle-scale",
        type=float,
        default=0.8,
        help="multiply the frozen obstacle radius by this factor "
        "(default 0.8 = the locked-in 20%%-smaller visualization size). "
        "Values != 1.0 deviate from the frozen instance and are exploratory "
        "visualizations, not benchmark results; pass 1.0 for the frozen radius.",
    )
    args = parser.parse_args()

    plan = load_plan()
    payload, instances = load_frozen_instances()
    manifest = load_llm_feedback_manifest(
        MANIFEST, instances_sha256=payload["instances_sha256"]
    )

    from lampc_cbf.paper_continuous_gamma import (
        NIMContinuousGammaConfig,
        NIMContinuousGammaMapper,
    )

    checkpoint_dir = Path(manifest["provider"]["checkpoint_dir"])
    mapper = NIMContinuousGammaMapper(
        NIMContinuousGammaConfig(
            model=str(manifest["provider"]["model"]),
            checkpoint_path=str(checkpoint_dir / "decisions.jsonl"),
        )
    )

    by_id = {i.episode_id: i for i in instances}
    for episode_id in args.episode_ids:
        instance = by_id[episode_id]
        kwargs = runner_kwargs(instance, plan)

        if args.obstacle_scale != 1.0:
            kwargs["obstacle_radius"] = (
                float(kwargs["obstacle_radius"]) * args.obstacle_scale
            )
        if args.max_steps is not None:
            kwargs["max_steps"] = args.max_steps
        if args.goal_scale != 1.0:
            kwargs["goal_offset"] = tuple(
                float(value) * args.goal_scale for value in kwargs["goal_offset"]
            )

        protocol = manifest["feedback_protocol"]
        decision = mapper.infer(
            str(protocol["user_utterance"]),
            current_gamma=kwargs["gamma"],
            task="Move gripper to the reach target.",
        )
        if decision.cache_hit is False and decision.fallback_used is False:
            # replay-only guard: the committed checkpoint must satisfy this
            print(f"[warn] {episode_id}: decision not from cache")

        kwargs.update(manifest["controller_base"]["runner_kwargs_overrides"])
        kwargs["gamma_range_mode"] = str(protocol["gamma_range_mode"])
        if not decision.fallback_used:
            apply_time = (
                instance.feedback_intervention_time_s + decision.latency_seconds
            )
            kwargs["gamma_schedule"] = ((apply_time, decision.gamma),)
            kwargs["gamma_schedule_request_times"] = (
                instance.feedback_intervention_time_s,
            )
            kwargs["gamma_update_ttl"] = float(protocol["gamma_update_ttl_s"])

        suffix = (
            ""
            if args.obstacle_scale == 1.0
            else f"_obs{int(round(args.obstacle_scale * 100))}"
        )
        if args.max_steps is not None:
            suffix += f"_s{args.max_steps}"
        if args.goal_scale != 1.0:
            suffix += f"_goal{int(round(args.goal_scale * 100))}"
        output_dir = (
            Path(args.output_root) / f"render_nim_slack_random_{episode_id}{suffix}"
        )
        kwargs["output_dir"] = str(output_dir)
        kwargs["save_plots"] = True
        kwargs["save_animation"] = True
        kwargs["save_metrics"] = True

        print(
            f"[run] {episode_id}: gamma {kwargs['gamma']}->{decision.gamma} "
            f"(cache_hit={decision.cache_hit}) -> {output_dir}"
        )
        result = run_smooth_dynamic_demo(SmoothDynamicConfig(**kwargs))
        reached = getattr(result, "reached_goal", None)
        collision = getattr(result, "collision", None)
        steps = getattr(result, "steps", None)
        print(
            f"[done] {episode_id}: reached_goal={reached} collision={collision} "
            f"steps={steps}"
        )


if __name__ == "__main__":
    main()
