"""Paired Safe Panda runner for the separately labeled 8-D paper-state profile."""

from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path
from typing import Any, Sequence

from .hf_llm import GammaDecision
from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo
from .table4_scenarios import Table4Scenario, scenario_runner_kwargs


def _joint_success(result: Any) -> bool:
    return bool(result.outcome == "goal" and result.reached_goal and not result.collision)


def _mcnemar_pvalue(baseline: Sequence[bool], method: Sequence[bool]) -> float:
    baseline_only = sum(a and not b for a, b in zip(baseline, method))
    method_only = sum(b and not a for a, b in zip(baseline, method))
    discordant = baseline_only + method_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(baseline_only, method_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _run_one(
    scenario: Table4Scenario,
    method: str,
    feedback: GammaDecision | None,
    *,
    seed_base: int,
    output_dir: Path,
) -> dict[str, Any]:
    kwargs = scenario_runner_kwargs(scenario, seed_base=seed_base)
    kwargs.update(
        {
            "cbf_transition_mode": "double_integrator",
            "output_dir": str(output_dir / f"episode_{scenario.episode_id:02d}_{method}"),
            "save_plots": False,
            "save_metrics": False,
        }
    )
    kwargs["gamma"] = 0.15
    if feedback is not None:
        kwargs["gamma_schedule"] = ((scenario.intervention_time + feedback.latency_seconds, feedback.gamma),)
        kwargs["gamma_schedule_request_times"] = (scenario.intervention_time,)
    result = run_smooth_dynamic_demo(SmoothDynamicConfig(**kwargs))
    return {
        "episode_id": scenario.episode_id,
        "category": scenario.category,
        "method": method,
        "goal_contract": "ee_reach_only",
        "goal_offset_m": [0.0, 0.30, 0.0],
        "obstacle_start_pos": list(scenario.obstacle_start_pos),
        "obstacle_velocity": list(scenario.obstacle_velocity),
        "intervention_time": scenario.intervention_time,
        "noise_sigma": scenario.noise_sigma,
        "outcome": result.outcome,
        "joint_success": _joint_success(result),
        "reached_goal": result.reached_goal,
        "collision": result.collision,
        "steps": result.steps,
        "minimum_true_clearance": result.minimum_true_clearance,
        "minimum_true_barrier": result.minimum_true_barrier,
        "minimum_true_cbf_residual": result.minimum_true_cbf_residual,
        "final_goal_distance": result.final_goal_distance,
        "final_gamma": result.final_gamma,
        "gamma_updates_applied": result.gamma_updates_applied,
        "solver_failures": result.solver_failures,
        "solver_rejections": result.solver_rejections,
        "deadline_misses": result.deadline_misses,
        "emergency_fallbacks": result.emergency_fallbacks,
        "mean_model_transition_error": result.mean_model_transition_error,
        "max_model_transition_error": result.max_model_transition_error,
        "mean_action_tracking_error": result.mean_action_tracking_error,
        "max_action_tracking_error": result.max_action_tracking_error,
        "true_cbf_violation_steps": result.true_cbf_violation_steps,
        "minimum_true_cbf_residual": result.minimum_true_cbf_residual,
        "feedback_latency": feedback.latency_seconds if feedback is not None else 0.0,
        "feedback_available_time": (
            scenario.intervention_time + feedback.latency_seconds if feedback is not None else None
        ),
        "cbf_transition_mode": "double_integrator",
    }


def run_safe_panda_8d_double_integrator_benchmark(
    scenarios: Sequence[Table4Scenario],
    decisions: Sequence[GammaDecision],
    *,
    output_dir: str | Path,
    seed_base: int = 20260715,
    bootstrap_resamples: int = 10000,
) -> dict[str, Any]:
    if len(scenarios) != 50 or len(decisions) != 50:
        raise ValueError("Safe Panda 8D benchmark requires 50 scenarios and decisions")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scenario, decision in zip(scenarios, decisions):
        rows.append(_run_one(scenario, "safe_panda_8d_fixed_g015", None, seed_base=seed_base, output_dir=root))
        rows.append(_run_one(scenario, "safe_panda_8d_async_feedback", decision, seed_base=seed_base, output_dir=root))
    fixed = [row for row in rows if row["method"] == "safe_panda_8d_fixed_g015"]
    feedback = [row for row in rows if row["method"] == "safe_panda_8d_async_feedback"]
    fixed_success = [bool(row["joint_success"]) for row in fixed]
    feedback_success = [bool(row["joint_success"]) for row in feedback]
    differences = [float(b) - float(a) for a, b in zip(fixed_success, feedback_success)]
    summary = {
        "profile": "safe_panda_8d_double_integrator_scenario_extension",
        "goal_contract": "ee_reach_only",
        "cbf_transition_mode": "double_integrator",
        "episodes_per_method": 50,
        "rows": len(rows),
        "solver_failure_outcomes": sum(row["outcome"] == "solver_failure" for row in rows),
        "methods": {
            "safe_panda_8d_fixed_g015": {
                "successes": sum(fixed_success),
                "collisions": sum(bool(row["collision"]) for row in fixed),
                "solver_failures": sum(int(row["solver_failures"]) for row in fixed),
            },
            "safe_panda_8d_async_feedback": {
                "successes": sum(feedback_success),
                "collisions": sum(bool(row["collision"]) for row in feedback),
                "solver_failures": sum(int(row["solver_failures"]) for row in feedback),
            },
        },
        "paired_success_difference": sum(differences) / len(differences),
        "mcnemar_exact_p": _mcnemar_pvalue(fixed_success, feedback_success),
        "bootstrap_resamples": bootstrap_resamples,
        "fidelity": "not exact Table-4; Safe Panda 8D double-integrator extension",
        "transition_error_m": {
            "max_over_episodes": max(row["max_model_transition_error"] for row in rows),
            "mean_over_episodes": sum(row["mean_model_transition_error"] for row in rows) / len(rows),
        },
        "action_tracking_error_m": {
            "max_over_episodes": max(row["max_action_tracking_error"] for row in rows),
            "mean_over_episodes": sum(row["mean_action_tracking_error"] for row in rows) / len(rows),
        },
    }
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
