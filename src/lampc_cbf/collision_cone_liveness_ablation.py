"""Paired ablation for collision-cone side selection and liveness guidance."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .remediation_benchmark import deterministic_scenario_grid


@dataclass(frozen=True, slots=True)
class CollisionConeLivenessVariant:
    name: str
    side_latch: bool
    policy_library: bool
    tangential_subgoal: bool


VARIANTS: tuple[CollisionConeLivenessVariant, ...] = (
    CollisionConeLivenessVariant("collision_cone_baseline", False, False, False),
    CollisionConeLivenessVariant("side_latch", True, False, False),
    CollisionConeLivenessVariant("policy_library", True, True, False),
    CollisionConeLivenessVariant(
        "policy_library_tangential_subgoal", True, True, True
    ),
)


@dataclass(frozen=True, slots=True)
class CollisionConeLivenessAblationConfig:
    episodes: int = 20
    seed: int = 20260714
    workers: int = 4
    max_steps: int = 140
    output_dir: str = "artifacts/collision_cone_liveness_ablation_20"
    max_collision_rate: float = 0.05
    max_controller_stall_rate: float = 0.10
    minimum_intervention_reduction: float = 0.50
    max_mean_side_switches: float = 2.0
    max_solver_rejection_rate: float = 0.01
    max_deadline_miss_rate: float = 0.01
    p99_solve_time_limit: float = 0.04

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.workers < 1 or self.max_steps < 4:
            raise ValueError("positive episodes/workers and max_steps >= 4 are required")
        if self.p99_solve_time_limit <= 0.0 or self.max_mean_side_switches < 0.0:
            raise ValueError("timing and side-switch limits must be non-negative")
        for value in (
            self.max_collision_rate,
            self.max_controller_stall_rate,
            self.minimum_intervention_reduction,
            self.max_solver_rejection_rate,
            self.max_deadline_miss_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rate gates must be in [0, 1]")


def _run_condition(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo

    condition = payload["condition"]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        result = run_smooth_dynamic_demo(
            SmoothDynamicConfig(
                delta_u_weight=2.0,
                gamma=0.15,
                seed=int(condition["seed"]),
                max_steps=int(payload["max_steps"]),
                reference_speed=0.08,
                reference_mode="straight",
                safety_mode="cbf",
                prediction_mode="velocity_tube",
                safety_reflex_enabled=True,
                optimal_decay_weight=10.0,
                feedback_ttc_threshold=1.5,
                feedback_response_latency=0.4971896839560941,
                feedback_gamma=0.02,
                reject_feedback_without_causal_opportunity=True,
                safety_profile_recovery_enabled=True,
                robot_velocity_feedback_enabled=True,
                reflex_backup_selection="task_consistent",
                reflex_committed_backup_enabled=True,
                sampled_data_margin_enabled=True,
                reflex_barrier_mode="collision_cone",
                reflex_side_latch_enabled=variant.side_latch,
                reflex_policy_library_enabled=variant.policy_library,
                reflex_tangential_subgoal_enabled=variant.tangential_subgoal,
                reflex_tangential_subgoal_distance=0.06,
                obstacle_start_offset=(condition["lateral_offset"], 0.44, 0.0),
                obstacle_velocity=(0.0, -condition["obstacle_speed"], 0.0),
                save_animation=False,
                save_plots=False,
                save_metrics=False,
                output_dir=f"/tmp/lampc-cone-liveness-{os.getpid()}",
            )
        )
        rows.append(
            {
                **condition,
                "variant": variant.name,
                "outcome": result.outcome,
                "success": bool(result.reached_goal and not result.collision),
                "collision": result.collision,
                "steps": result.steps,
                "minimum_true_clearance": result.minimum_true_clearance,
                "final_goal_distance": result.final_goal_distance,
                "net_goal_progress": result.net_goal_progress,
                "reflex_interventions": result.reflex_interventions,
                "reflex_backups": result.reflex_backups,
                "reflex_side_switches": result.reflex_side_switches,
                "reflex_policy_selections": result.reflex_policy_selections,
                "reflex_robust_recoveries": result.reflex_robust_recoveries,
                "solver_rejections": result.solver_rejections,
                "deadline_misses": result.deadline_misses,
                "p99_solve_time": result.p99_solve_time,
            }
        )
    return rows


def summarize_collision_cone_liveness_rows(
    rows: Sequence[Mapping[str, Any]],
    config: CollisionConeLivenessAblationConfig,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant.name]
        if len(selected) != config.episodes:
            raise ValueError(f"{variant.name} does not have {config.episodes} episodes")
        steps = sum(int(row["steps"]) for row in selected)
        outcomes = {
            outcome: sum(row["outcome"] == outcome for row in selected)
            for outcome in (
                "goal",
                "collision",
                "safety_timeout",
                "controller_stall",
                "solver_failure",
                "environment_truncated",
            )
        }
        summaries[variant.name] = {
            "episodes": len(selected),
            "success_rate": sum(bool(row["success"]) for row in selected)
            / len(selected),
            "collision_rate": sum(bool(row["collision"]) for row in selected)
            / len(selected),
            "controller_stall_rate": outcomes["controller_stall"] / len(selected),
            "outcomes": outcomes,
            "mean_net_goal_progress": sum(
                float(row["net_goal_progress"]) for row in selected
            )
            / len(selected),
            "mean_minimum_true_clearance": sum(
                float(row["minimum_true_clearance"]) for row in selected
            )
            / len(selected),
            "reflex_interventions": sum(
                int(row["reflex_interventions"]) for row in selected
            ),
            "reflex_backups": sum(int(row["reflex_backups"]) for row in selected),
            "mean_side_switches": sum(
                int(row["reflex_side_switches"]) for row in selected
            )
            / len(selected),
            "policy_selections": sum(
                int(row["reflex_policy_selections"]) for row in selected
            ),
            "robust_recoveries": sum(
                int(row["reflex_robust_recoveries"]) for row in selected
            ),
            "solver_rejection_rate": sum(
                int(row["solver_rejections"]) for row in selected
            )
            / steps,
            "deadline_miss_rate": sum(
                int(row["deadline_misses"]) for row in selected
            )
            / steps,
            "maximum_episode_p99_solve_time": max(
                float(row["p99_solve_time"]) for row in selected
            ),
        }

    baseline = summaries["collision_cone_baseline"]
    target = summaries["policy_library_tangential_subgoal"]
    baseline_interventions = int(baseline["reflex_interventions"])
    intervention_reduction = (
        1.0 - int(target["reflex_interventions"]) / baseline_interventions
        if baseline_interventions
        else 0.0
    )
    checks = {
        "collision": target["collision_rate"] <= config.max_collision_rate,
        "controller_stall": (
            target["controller_stall_rate"] <= config.max_controller_stall_rate
        ),
        "intervention_reduction": (
            intervention_reduction >= config.minimum_intervention_reduction
        ),
        "side_switching": (
            target["mean_side_switches"] <= config.max_mean_side_switches
        ),
        "feasibility": (
            target["solver_rejection_rate"] <= config.max_solver_rejection_rate
        ),
        "timing": (
            target["deadline_miss_rate"] <= config.max_deadline_miss_rate
            and target["maximum_episode_p99_solve_time"]
            <= config.p99_solve_time_limit
        ),
    }
    return {
        "protocol": "collision-cone-liveness-ablation-v1",
        "config": asdict(config),
        "variants": [asdict(variant) for variant in VARIANTS],
        "methods": summaries,
        "paired_target_effect": {
            "intervention_reduction_vs_collision_cone_baseline": (
                intervention_reduction
            ),
            "stall_rate_delta": (
                target["controller_stall_rate"]
                - baseline["controller_stall_rate"]
            ),
            "mean_progress_delta": (
                target["mean_net_goal_progress"]
                - baseline["mean_net_goal_progress"]
            ),
        },
        "gate": {"passed": all(checks.values()), "checks": checks},
        "interpretation_rule": (
            "All variants use identical conditions. Robust-feasible policies are "
            "hard-rejected on negative robust rollout clearance or CBF residual. "
            "When tube expansion excludes the current state, explicitly labeled "
            "recovery policies retain hard physical collision filtering but are "
            "not counted as robust-feasible. Collision and clearance use raw "
            "simulated trajectories only."
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["episode"]), str(row["variant"])))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def run_collision_cone_liveness_ablation(
    config: CollisionConeLivenessAblationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CollisionConeLivenessAblationConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        {"condition": condition, "max_steps": cfg.max_steps}
        for condition in deterministic_scenario_grid(cfg.episodes, seed=cfg.seed)
    ]
    rows: list[dict[str, Any]] = []
    if cfg.workers == 1:
        for completed, payload in enumerate(payloads, start=1):
            rows.extend(_run_condition(payload))
            print(f"[cone-liveness] completed {completed}/{len(payloads)}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as executor:
            futures = [executor.submit(_run_condition, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                print(
                    f"[cone-liveness] completed {completed}/{len(payloads)}",
                    flush=True,
                )
    _write_rows(output_dir / "episodes.csv", rows)
    summary = summarize_collision_cone_liveness_rows(rows, cfg)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_collision_cone_liveness_ablation(), indent=2))
