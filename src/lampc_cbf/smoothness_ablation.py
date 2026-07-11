"""Reproducible smoothness ablation and evidence synthesis."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any


def _command_jerk_metrics(controls: Any, dt: float) -> dict[str, float]:
    import numpy as np

    values = np.asarray(controls, dtype=float)[:, :3]
    second_difference = np.diff(values, n=2, axis=0) / dt**2
    norm = np.linalg.norm(second_difference, axis=1)
    return {
        "command_jerk_rms": float(np.sqrt(np.mean(norm**2))),
        "command_jerk_max": float(np.max(norm)),
    }


def run_smoothness_ablation(
    output_root: str = "artifacts/smoothness_ablation",
) -> dict[str, Any]:
    """Run four Δu weights, one Δ²u variant, and decide on Ruckig."""

    import matplotlib.pyplot as plt
    import numpy as np

    from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo
    from .smoothness import calculate_smoothness_metrics

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(
        "artifacts/dynamic_obstacle_mpc_cbf/gamma_0.10/metrics.json"
    )
    if not baseline_path.exists():
        raise FileNotFoundError(
            "run scripts/run_dynamic_obstacle_mpc_cbf.py before the ablation"
        )
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    dt = float(baseline_payload["paper_alignment"]["control_period_seconds"])
    baseline_smoothness = calculate_smoothness_metrics(
        baseline_payload["positions"], dt
    )
    variants: dict[str, dict[str, Any]] = {
        "waypoint_baseline": {
            "delta_u_weight": 0.5,
            "jerk_weight": 0.0,
            "reached_goal": baseline_payload["result"]["reached_goal"],
            "collision": baseline_payload["result"]["collision"],
            "minimum_true_clearance": baseline_payload["result"]["minimum_true_clearance"],
            "final_goal_distance": baseline_payload["result"]["final_goal_distance"],
            "mean_solve_time": baseline_payload["result"]["mean_solve_time"],
            "max_solve_time": baseline_payload["result"]["max_solve_time"],
            "smoothness": baseline_smoothness.as_dict(),
            **_command_jerk_metrics(baseline_payload["controls"], dt),
            "positions": baseline_payload["positions"],
        }
    }

    for weight in (0.5, 1.0, 2.0, 5.0):
        name = f"spline_du_{weight:g}"
        result = run_smooth_dynamic_demo(
            SmoothDynamicConfig(
                delta_u_weight=weight,
                output_dir=str(root / name),
            )
        )
        payload = json.loads((root / name / "metrics.json").read_text(encoding="utf-8"))
        variants[name] = {
            "delta_u_weight": weight,
            "jerk_weight": 0.0,
            "reached_goal": result.reached_goal,
            "collision": result.collision,
            "minimum_true_clearance": result.minimum_true_clearance,
            "final_goal_distance": result.final_goal_distance,
            "mean_solve_time": result.mean_solve_time,
            "max_solve_time": result.max_solve_time,
            "smoothness": result.smoothness.as_dict(),
            **_command_jerk_metrics(payload["controls"], dt),
            "positions": payload["positions"],
        }

    feasible_spline_names = [
        name for name, value in variants.items()
        if name.startswith("spline_du_")
        and value["reached_goal"] and not value["collision"]
        and value["max_solve_time"] < dt
    ]
    best_mpc_name = min(
        feasible_spline_names,
        key=lambda name: variants[name]["smoothness"]["jerk_rms"],
    )
    best_delta_weight = variants[best_mpc_name]["delta_u_weight"]
    jerk_name = f"spline_du_{best_delta_weight:g}_jerk_5"
    jerk_result = run_smooth_dynamic_demo(
        SmoothDynamicConfig(
            delta_u_weight=best_delta_weight,
            jerk_weight=5.0,
            output_dir=str(root / jerk_name),
        )
    )
    jerk_payload = json.loads(
        (root / jerk_name / "metrics.json").read_text(encoding="utf-8")
    )
    variants[jerk_name] = {
        "delta_u_weight": best_delta_weight,
        "jerk_weight": 5.0,
        "reached_goal": jerk_result.reached_goal,
        "collision": jerk_result.collision,
        "minimum_true_clearance": jerk_result.minimum_true_clearance,
        "final_goal_distance": jerk_result.final_goal_distance,
        "mean_solve_time": jerk_result.mean_solve_time,
        "max_solve_time": jerk_result.max_solve_time,
        "smoothness": jerk_result.smoothness.as_dict(),
        **_command_jerk_metrics(jerk_payload["controls"], dt),
        "positions": jerk_payload["positions"],
    }

    baseline_jerk = baseline_smoothness.jerk_rms
    best_jerk = variants[best_mpc_name]["smoothness"]["jerk_rms"]
    jerk_reduction = 1.0 - best_jerk / baseline_jerk
    ruckig_required = not (
        variants[best_mpc_name]["reached_goal"]
        and not variants[best_mpc_name]["collision"]
        and variants[best_mpc_name]["max_solve_time"] < dt
        and jerk_reduction >= 0.25
        and variants[best_mpc_name]["smoothness"]["jerk_max"]
        < baseline_smoothness.jerk_max
    )

    labels = list(variants)
    colors = ["gray" if name == "waypoint_baseline" else "steelblue" for name in labels]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    quantities = (
        ("jerk_rms", "Jerk RMS [m/s³]"),
        ("jerk_max", "Maximum jerk [m/s³]"),
        ("acceleration_rms", "Acceleration RMS [m/s²]"),
        ("mean_curvature", "Mean curvature [1/m]"),
        ("path_length", "Path length [m]"),
    )
    for axis, (key, title) in zip(axes.flat[:5], quantities):
        axis.bar(
            range(len(labels)),
            [variants[name]["smoothness"][key] for name in labels],
            color=colors,
        )
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[5].bar(
        range(len(labels)),
        [variants[name]["minimum_true_clearance"] for name in labels],
        color=colors,
    )
    axes.flat[5].axhline(0.0, color="red", linestyle="--")
    axes.flat[5].set_xticks(
        range(len(labels)), labels, rotation=35, ha="right", fontsize=7
    )
    axes.flat[5].set_title("Raw minimum true clearance [m]")
    axes.flat[5].grid(axis="y", alpha=0.25)
    fig.savefig(root / "ablation_metrics.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name in ("waypoint_baseline", best_mpc_name, jerk_name):
        positions = np.asarray(variants[name]["positions"])
        axis.plot(positions[:, 0], positions[:, 1], label=name)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Raw end-effector trajectories")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.savefig(root / "raw_trajectory_comparison.png", dpi=160)
    plt.close(fig)

    serializable_variants = {
        name: {key: value for key, value in data.items() if key != "positions"}
        for name, data in variants.items()
    }
    summary = {
        "experiment_id": "dynamic-obstacle-smoothness-ablation-seed-11",
        "control_period_seconds": dt,
        "variants": serializable_variants,
        "selected_mpc_variant": best_mpc_name,
        "jerk_rms_reduction_vs_waypoint_baseline": jerk_reduction,
        "jerk_variant": jerk_name,
        "jerk_variant_improves_raw_jerk_rms": (
            variants[jerk_name]["smoothness"]["jerk_rms"] < best_jerk
        ),
        "ruckig_required": ruckig_required,
        "ruckig_integrated": False,
        "ruckig_decision_rule": (
            "integrate only if safe real-time MPC fails to reduce raw jerk RMS by 25% "
            "and maximum raw jerk does not improve"
        ),
        "safety_metric_source": "raw simulated positions; never visual spline",
    }
    (root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
