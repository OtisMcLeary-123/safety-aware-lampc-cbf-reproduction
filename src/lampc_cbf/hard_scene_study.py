"""Paper-style hard-scene and paired online-feedback Monte Carlo study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

from .hf_llm import GammaDecision


@dataclass(frozen=True, slots=True)
class HardSceneStudyConfig:
    episodes: int = 50
    seed: int = 20260712
    bootstrap_resamples: int = 10_000
    speed_lower: float = 0.025
    speed_upper: float = 0.20
    intervention_time_lower: float = 0.0
    intervention_time_upper: float = 0.40
    output_dir: str = "artifacts/hard_scene_study"

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.bootstrap_resamples < 1:
            raise ValueError("episode and bootstrap counts must be positive")
        if not 0.0 < self.speed_lower < self.speed_upper:
            raise ValueError("invalid obstacle speed interval")
        if not 0.0 <= self.intervention_time_lower <= self.intervention_time_upper:
            raise ValueError("invalid intervention-time interval")


def _bootstrap_success_interval(
    successes: list[bool], resamples: int, rng: Any
) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(successes, dtype=float)
    samples = rng.choice(values, size=(resamples, len(values)), replace=True)
    lower, upper = np.quantile(np.mean(samples, axis=1), [0.025, 0.975])
    return float(lower), float(upper)


def run_hard_scene_study(
    initial_decision: GammaDecision,
    feedback_decision: GammaDecision,
    config: HardSceneStudyConfig | None = None,
) -> dict[str, Any]:
    """Run paired methods on identical randomized hard-scene conditions."""

    import matplotlib.pyplot as plt
    import numpy as np

    from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo

    cfg = config or HardSceneStudyConfig()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    speeds = rng.uniform(cfg.speed_lower, cfg.speed_upper, cfg.episodes)
    intervention_times = rng.uniform(
        cfg.intervention_time_lower, cfg.intervention_time_upper, cfg.episodes
    )
    lateral_signs = rng.choice((-1.0, 1.0), size=cfg.episodes)
    lateral_offsets = lateral_signs * rng.uniform(0.055, 0.065, cfg.episodes)

    methods = {
        "distance_baseline": {"safety_mode": "distance", "gamma": 0.15},
        "fixed_gamma_015": {"safety_mode": "cbf", "gamma": 0.15},
        "llm_initial_gamma": {
            "safety_mode": "cbf", "gamma": initial_decision.gamma
        },
        "llm_online_feedback": {"safety_mode": "cbf", "gamma": 0.15},
    }
    rows: list[dict[str, Any]] = []
    scratch = root / "_scratch"
    for method_name, method in methods.items():
        print(f"[study] starting {method_name}", flush=True)
        for episode in range(cfg.episodes):
            schedule: tuple[tuple[float, float], ...] = ()
            if method_name == "llm_online_feedback":
                schedule = ((
                    float(intervention_times[episode] + feedback_decision.latency_seconds),
                    feedback_decision.gamma,
                ),)
            run_config = SmoothDynamicConfig(
                delta_u_weight=2.0,
                gamma=float(method["gamma"]),
                seed=1000 + episode,
                max_steps=140,
                reference_speed=0.08,
                reference_mode="straight",
                safety_mode=str(method["safety_mode"]),
                gamma_schedule=schedule,
                obstacle_start_offset=(float(lateral_offsets[episode]), 0.44, 0.0),
                obstacle_velocity=(0.0, -float(speeds[episode]), 0.0),
                save_animation=False,
                save_plots=False,
                save_metrics=False,
                output_dir=str(scratch),
            )
            result = run_smooth_dynamic_demo(run_config)
            rows.append({
                "method": method_name,
                "episode": episode,
                "seed": run_config.seed,
                "obstacle_speed": float(speeds[episode]),
                "lateral_offset": float(lateral_offsets[episode]),
                "intervention_time": float(intervention_times[episode]),
                "scheduled_update_time": schedule[0][0] if schedule else None,
                "gamma_initial": float(method["gamma"]),
                "gamma_final": result.final_gamma,
                "gamma_updates_applied": result.gamma_updates_applied,
                "success": bool(result.reached_goal and not result.collision),
                "reached_goal": result.reached_goal,
                "collision": result.collision,
                "steps": result.steps,
                "minimum_true_clearance": result.minimum_true_clearance,
                "final_goal_distance": result.final_goal_distance,
                "mean_solve_time": result.mean_solve_time,
                "max_solve_time": result.max_solve_time,
            })
            if (episode + 1) % 10 == 0:
                print(
                    f"[study] {method_name}: {episode + 1}/{cfg.episodes}",
                    flush=True,
                )

    summaries: dict[str, Any] = {}
    bootstrap_rng = np.random.default_rng(cfg.seed + 1)
    for method_name in methods:
        method_rows = [row for row in rows if row["method"] == method_name]
        successes = [bool(row["success"]) for row in method_rows]
        lower, upper = _bootstrap_success_interval(
            successes, cfg.bootstrap_resamples, bootstrap_rng
        )
        summaries[method_name] = {
            "successes": sum(successes),
            "episodes": len(successes),
            "success_rate": float(np.mean(successes)),
            "bootstrap_95_ci": [lower, upper],
            "collisions": sum(bool(row["collision"]) for row in method_rows),
            "mean_minimum_true_clearance": float(np.mean([
                row["minimum_true_clearance"] for row in method_rows
            ])),
            "mean_solve_time": float(np.mean([
                row["mean_solve_time"] for row in method_rows
            ])),
            "max_solve_time": float(np.max([
                row["max_solve_time"] for row in method_rows
            ])),
            "mean_simulated_duration": float(np.mean([
                row["steps"] * 0.04 for row in method_rows
            ])),
            "max_simulated_duration": float(np.max([
                row["steps"] * 0.04 for row in method_rows
            ])),
            "feedback_updates_applied": sum(
                int(row["gamma_updates_applied"]) for row in method_rows
            ),
        }

    fieldnames = list(rows[0])
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = list(summaries)
    rates = [summaries[label]["success_rate"] for label in labels]
    lower_errors = [
        rates[index] - summaries[label]["bootstrap_95_ci"][0]
        for index, label in enumerate(labels)
    ]
    upper_errors = [
        summaries[label]["bootstrap_95_ci"][1] - rates[index]
        for index, label in enumerate(labels)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(range(len(labels)), rates, yerr=[lower_errors, upper_errors], capsize=5)
    axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("collision-free goal success rate")
    axes[0].set_title(f"{cfg.episodes} paired episodes with bootstrap 95% CI")
    axes[0].grid(axis="y", alpha=0.3)
    for method_name in labels:
        method_rows = [row for row in rows if row["method"] == method_name]
        axes[1].scatter(
            [row["obstacle_speed"] for row in method_rows],
            [int(row["success"]) for row in method_rows],
            alpha=0.55, s=18, label=method_name,
        )
    axes[1].set_xlabel("obstacle speed [m/s]")
    axes[1].set_ylabel("success")
    axes[1].set_yticks([0, 1])
    axes[1].set_title("Outcome versus paired obstacle speed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.savefig(root / "success_rate_and_speed.png", dpi=160)
    plt.close(fig)

    summary = {
        "study_id": "paper-hard-scene-online-feedback-50",
        "config": asdict(cfg),
        "initial_llm_decision": initial_decision.as_dict(),
        "feedback_llm_decision": feedback_decision.as_dict(),
        "llm_metrics": {
            "initial_latency_seconds": initial_decision.latency_seconds,
            "feedback_latency_seconds": feedback_decision.latency_seconds,
            "initial_fallback_used": initial_decision.fallback_used,
            "feedback_fallback_used": feedback_decision.fallback_used,
        },
        "methods": summaries,
        "paired_randomization": True,
        "llm_api_policy": (
            "one real decision per unique prompt; validated decision and measured "
            "latency replayed across paired episodes"
        ),
        "scene": {
            "reference": "straight; no pre-shaped avoidance waypoint",
            "obstacle_motion": "head-on with randomized speed and lateral sign",
            "speed_interval_mps": [cfg.speed_lower, cfg.speed_upper],
            "sensor_period_seconds": 0.67,
            "measurement_noise_sigma_meters": 0.005,
        },
    }
    (root / "study_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[study] complete", flush=True)
    return summary
