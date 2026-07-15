#!/usr/bin/env python3
"""Run a resumable paired fixed-vs-feedback benchmark with the opt-in 3-D profile."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from lampc_cbf.hf_llm import GammaDecision
from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo
from lampc_cbf.table4_scenarios import Table4Scenario, load_scenarios, scenario_runner_kwargs


FIXED_METHOD = "safe_panda_3d_fixed_g015"
FEEDBACK_METHOD = "safe_panda_3d_async_feedback"


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _profile_overrides(setup: dict[str, Any]) -> dict[str, Any]:
    controller = setup["controller"]
    waypoints = tuple(tuple(float(value) for value in item) for item in controller["avoidance_waypoint_offsets_m"])
    if len(waypoints) < 2 or any(len(item) != 3 for item in waypoints):
        raise ValueError("3-D profile requires at least two length-3 waypoint offsets")
    methods = tuple(setup["benchmark"]["paired_methods"])
    if methods != (FIXED_METHOD, FEEDBACK_METHOD):
        raise ValueError("setup paired_methods does not match the 3-D workflow contract")
    return {
        "gamma": float(controller["initial_gamma"]),
        "max_steps": int(setup["benchmark"]["max_steps"]),
        "goal_offset": tuple(float(value) for value in controller["goal_offset_m"]),
        "reference_speed": float(controller["reference_speed_mps"]),
        "position_q_weights": tuple(float(value) for value in controller["position_q_weights"]),
        "cbf_transition_mode": str(controller["cbf_transition_mode"]),
        "reference_mode": str(controller["reference_mode"]),
        "reference_route_profile": str(controller["reference_route_profile"]),
        "avoidance_waypoint_offsets": waypoints,
        "safety_reflex_enabled": bool(controller["safety_reflex_enabled"]),
        "reflex_barrier_mode": str(controller["reflex_barrier_mode"]),
        "reflex_policy_library_enabled": bool(controller["reflex_policy_library_enabled"]),
        "reflex_tangential_subgoal_enabled": bool(controller["reflex_tangential_subgoal_enabled"]),
        "save_plots": False,
        "save_metrics": False,
    }


def _load_decisions(path: str | Path, manifest: PaperFidelityManifest) -> tuple[GammaDecision, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 50:
        raise ValueError("3-D benchmark requires exactly 50 feedback decisions")
    decisions = []
    for item in payload:
        record = dict(item)
        record.setdefault("raw_response", None)
        decision = GammaDecision(**record)
        accepted = manifest.accepts_feedback_decision(
            model=decision.model,
            provider=decision.provider,
            cache_hit=decision.cache_hit,
        )
        if decision.fallback_used or decision.cache_hit or not accepted:
            raise ValueError("provider checkpoint is invalid for the selected manifest")
        decisions.append(decision)
    return tuple(decisions)


def _run_row(
    scenario: Table4Scenario,
    method: str,
    decision: GammaDecision | None,
    *,
    setup: dict[str, Any],
    output_dir: Path,
    runner: Callable[[SmoothDynamicConfig], Any] = run_smooth_dynamic_demo,
) -> dict[str, Any]:
    kwargs = scenario_runner_kwargs(scenario, seed_base=int(setup["benchmark"]["seed_base"]))
    kwargs.update(_profile_overrides(setup))
    scene = setup["scene"]
    kwargs["obstacle_start_offset"] = (
        float(scenario.obstacle_start_pos[0]),
        float(scenario.obstacle_start_pos[1]),
        float(scene["obstacle_height_offset_m"]),
    )
    kwargs["obstacle_velocity"] = (
        float(scenario.obstacle_velocity[0]),
        float(scenario.obstacle_velocity[1]),
        float(scene["obstacle_vertical_velocity_mps"]),
    )
    kwargs["obstacle_radius"] = float(scene["obstacle_radius_m"])
    kwargs["collision_radius"] = float(scene["ee_collision_radius_m"])
    kwargs["output_dir"] = str(output_dir / f"episode_{scenario.episode_id:02d}_{method}")
    if decision is not None:
        kwargs["gamma_schedule"] = ((scenario.intervention_time + decision.latency_seconds, decision.gamma),)
        kwargs["gamma_schedule_request_times"] = (scenario.intervention_time,)
    result = runner(SmoothDynamicConfig(**kwargs))
    return {
        "episode_id": scenario.episode_id,
        "category": scenario.category,
        "method": method,
        "outcome": result.outcome,
        "joint_success": bool(result.outcome == "goal" and result.reached_goal and not result.collision),
        "reached_goal": result.reached_goal,
        "collision": result.collision,
        "steps": result.steps,
        "minimum_true_clearance": result.minimum_true_clearance,
        "minimum_true_cbf_residual": result.minimum_true_cbf_residual,
        "final_goal_distance": result.final_goal_distance,
        "final_gamma": result.final_gamma,
        "gamma_updates_applied": result.gamma_updates_applied,
        "solver_failures": result.solver_failures,
        "solver_rejections": result.solver_rejections,
        "deadline_misses": result.deadline_misses,
        "emergency_fallbacks": result.emergency_fallbacks,
        "feedback_latency": decision.latency_seconds if decision is not None else 0.0,
        "feedback_available_time": scenario.intervention_time + decision.latency_seconds if decision is not None else None,
        "obstacle_start_pos": list(kwargs["obstacle_start_offset"]),
        "obstacle_velocity": list(kwargs["obstacle_velocity"]),
        "reference_route_profile": "3d_waypoints",
        "cbf_transition_mode": setup["controller"]["cbf_transition_mode"],
    }


def _mcnemar_pvalue(fixed: Sequence[bool], feedback: Sequence[bool]) -> float:
    fixed_only = sum(a and not b for a, b in zip(fixed, feedback))
    feedback_only = sum(b and not a for a, b in zip(fixed, feedback))
    discordant = fixed_only + feedback_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(fixed_only, feedback_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _summary(rows: Sequence[dict[str, Any]], bootstrap_resamples: int) -> dict[str, Any]:
    fixed = [row for row in rows if row["method"] == FIXED_METHOD]
    feedback = [row for row in rows if row["method"] == FEEDBACK_METHOD]
    if len(fixed) != 50 or len(feedback) != 50:
        raise ValueError("summary requires 50 completed paired episodes per method")
    fixed_success = [bool(row["joint_success"]) for row in fixed]
    feedback_success = [bool(row["joint_success"]) for row in feedback]
    fixed_only = [
        int(base["episode_id"])
        for base, method in zip(fixed, feedback)
        if bool(base["joint_success"]) and not bool(method["joint_success"])
    ]
    feedback_only = [
        int(method["episode_id"])
        for base, method in zip(fixed, feedback)
        if bool(method["joint_success"]) and not bool(base["joint_success"])
    ]
    method_summaries = {}
    for name, selected, successes in (
        (FIXED_METHOD, fixed, fixed_success),
        (FEEDBACK_METHOD, feedback, feedback_success),
    ):
        method_summaries[name] = {
            "successes": sum(successes),
            "collisions": sum(bool(row["collision"]) for row in selected),
            "solver_failures": sum(int(row["solver_failures"]) for row in selected),
            "solver_rejections": sum(int(row["solver_rejections"]) for row in selected),
            "deadline_misses": sum(int(row["deadline_misses"]) for row in selected),
            "emergency_fallbacks": sum(int(row["emergency_fallbacks"]) for row in selected),
            "gamma_updates_applied": sum(int(row["gamma_updates_applied"]) for row in selected),
            "mean_minimum_true_clearance_m": fmean(
                float(row["minimum_true_clearance"]) for row in selected
            ),
        }
    return {
        "profile": "safe_panda_3d_avoidance_50_extension",
        "episodes_per_method": 50,
        "rows": len(rows),
        "methods": method_summaries,
        "paired_success_difference": sum(float(b) - float(a) for a, b in zip(fixed_success, feedback_success)) / 50,
        "mcnemar_exact_p": _mcnemar_pvalue(fixed_success, feedback_success),
        "fixed_only_success_episode_ids": fixed_only,
        "feedback_only_success_episode_ids": feedback_only,
        "paired_mean_clearance_difference_m": fmean(
            float(method["minimum_true_clearance"])
            - float(base["minimum_true_clearance"])
            for base, method in zip(fixed, feedback)
        ),
        "bootstrap_resamples": bootstrap_resamples,
        "fidelity": "opt-in 3-D engineering extension; not exact Table-4 reproduction",
    }


def run_resumable_benchmark(
    scenarios: Sequence[Table4Scenario],
    decisions: Sequence[GammaDecision],
    setup: dict[str, Any],
    *,
    output_dir: Path,
    restart: bool = False,
    runner: Callable[[SmoothDynamicConfig], Any] = run_smooth_dynamic_demo,
) -> dict[str, Any]:
    if len(scenarios) != 50 or len(decisions) != 50:
        raise ValueError("3-D benchmark requires 50 scenarios and 50 decisions")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / str(setup["benchmark"]["resume_checkpoint"])
    fingerprint = sha256(json.dumps(setup, sort_keys=True).encode("utf-8")).hexdigest()
    if checkpoint.exists() and not restart:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("setup_sha256") != fingerprint:
            raise ValueError("resume checkpoint was created with a different setup")
        rows = list(state["rows"])
    else:
        rows = []
        _write_json_atomic(
            checkpoint,
            {"schema_version": 1, "setup_sha256": fingerprint, "rows": rows},
        )
    completed = {(int(row["episode_id"]), str(row["method"])) for row in rows}
    for scenario, decision in zip(scenarios, decisions):
        for method, feedback in ((FIXED_METHOD, None), (FEEDBACK_METHOD, decision)):
            key = (scenario.episode_id, method)
            if key in completed:
                continue
            row = _run_row(scenario, method, feedback, setup=setup, output_dir=output_dir, runner=runner)
            rows.append(row)
            completed.add(key)
            _write_json_atomic(checkpoint, {"schema_version": 1, "setup_sha256": fingerprint, "rows": rows})
            print(f"[safe-panda-3d] completed {len(rows)}/100: episode {scenario.episode_id} {method}", flush=True)
    rows.sort(key=lambda row: (int(row["episode_id"]), 0 if row["method"] == FIXED_METHOD else 1))
    summary = _summary(rows, int(setup["benchmark"]["bootstrap_resamples"]))
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json_atomic(output_dir / "benchmark_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-file", default="configs/safe_panda_3d_avoidance_50_setup.json")
    parser.add_argument("--scenario-file")
    parser.add_argument("--provider-manifest")
    parser.add_argument("--feedback-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--restart", action="store_true", help="discard the run checkpoint and start all 100 runs again")
    args = parser.parse_args()
    setup = json.loads(Path(args.setup_file).read_text(encoding="utf-8"))
    scenarios = load_scenarios(args.scenario_file or setup["scenario_path"])
    manifest = PaperFidelityManifest.load(args.provider_manifest or setup["llm"]["provider_manifest"])
    decisions = _load_decisions(args.feedback_checkpoint or setup["llm"]["checkpoint"], manifest)
    output_dir = Path(args.output_dir or setup["benchmark"]["output_dir"])
    summary = run_resumable_benchmark(scenarios, decisions, setup, output_dir=output_dir, restart=args.restart)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
