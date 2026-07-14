"""Paired C3BF versus Cartesian-adapted DPCBF experiment."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collision_cone_liveness_ablation import run_liveness_condition
from .remediation_benchmark import deterministic_scenario_grid


VARIANTS: tuple[tuple[str, str], ...] = (
    ("collision_cone_policy_library", "collision_cone"),
    ("dynamic_parabolic_policy_library", "dynamic_parabolic"),
)


@dataclass(frozen=True, slots=True)
class DPCBFAblationConfig:
    episodes: int = 20
    seed: int = 20260714
    workers: int = 4
    max_steps: int = 140
    output_dir: str = "artifacts/dpcbf_ablation_20"
    max_collision_rate: float = 0.05
    max_controller_stall_rate: float = 0.10
    max_solver_rejection_rate: float = 0.01
    max_deadline_miss_rate: float = 0.01
    p99_solve_time_limit: float = 0.04

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.workers < 1 or self.max_steps < 4:
            raise ValueError("positive episodes/workers and max_steps >= 4 are required")
        if self.p99_solve_time_limit <= 0.0:
            raise ValueError("p99_solve_time_limit must be positive")
        for value in (
            self.max_collision_rate,
            self.max_controller_stall_rate,
            self.max_solver_rejection_rate,
            self.max_deadline_miss_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rate gates must be in [0, 1]")


def _run_condition(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    condition = payload["condition"]
    return [
        run_liveness_condition(
            condition,
            int(payload["max_steps"]),
            variant_name=name,
            barrier_mode=barrier_mode,
            side_latch=True,
            policy_library=True,
            tangential_subgoal=True,
        )
        for name, barrier_mode in VARIANTS
    ]


def summarize_dpcbf_rows(
    rows: Sequence[Mapping[str, Any]], config: DPCBFAblationConfig
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, barrier_mode in VARIANTS:
        selected = [row for row in rows if row["variant"] == name]
        if len(selected) != config.episodes:
            raise ValueError(f"{name} does not have {config.episodes} episodes")
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
        summaries[name] = {
            "barrier_mode": barrier_mode,
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

    baseline = summaries["collision_cone_policy_library"]
    target = summaries["dynamic_parabolic_policy_library"]
    checks = {
        "collision": target["collision_rate"] <= config.max_collision_rate,
        "controller_stall": (
            target["controller_stall_rate"] <= config.max_controller_stall_rate
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
        "protocol": "cartesian-dpcbf-ablation-v1",
        "config": asdict(config),
        "methods": summaries,
        "paired_target_effect": {
            "success_rate_delta": target["success_rate"] - baseline["success_rate"],
            "stall_rate_delta": (
                target["controller_stall_rate"]
                - baseline["controller_stall_rate"]
            ),
            "mean_progress_delta": (
                target["mean_net_goal_progress"]
                - baseline["mean_net_goal_progress"]
            ),
            "intervention_delta": (
                target["reflex_interventions"] - baseline["reflex_interventions"]
            ),
        },
        "gate": {"passed": all(checks.values()), "checks": checks},
        "interpretation_rule": (
            "The DPCBF line-of-sight parabola is adapted from a planar "
            "kinematic-bicycle implementation to Cartesian 3D velocity. This "
            "ablation is empirical and does not transfer the source proof. "
            "Raw simulator trajectories remain the safety evidence."
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["episode"]), str(row["variant"])))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def run_dpcbf_ablation(
    config: DPCBFAblationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DPCBFAblationConfig()
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
            print(f"[dpcbf-ablation] completed {completed}/{len(payloads)}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as executor:
            futures = [executor.submit(_run_condition, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                print(
                    f"[dpcbf-ablation] completed {completed}/{len(payloads)}",
                    flush=True,
                )
    _write_rows(output_dir / "episodes.csv", rows)
    summary = summarize_dpcbf_rows(rows, cfg)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_dpcbf_ablation(), indent=2))
