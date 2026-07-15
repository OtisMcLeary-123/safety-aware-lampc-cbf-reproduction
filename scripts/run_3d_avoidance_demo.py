#!/usr/bin/env python3
"""Run an opt-in Safe Panda demo with an explicit 3-D avoidance route."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo


def _vector(values: list[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} requires exactly three values")
    return tuple(values)  # type: ignore[return-value]


def _save_figure5_style_comparison(output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(output_dir)
    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    positions = np.asarray(payload["positions"], dtype=float)
    reference = np.asarray(payload["reference_path"], dtype=float)
    obstacles = np.asarray(payload["true_obstacles"], dtype=float)
    clearances = np.asarray(payload["true_clearances"], dtype=float)
    dt = 0.04
    figure = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    top = figure.add_subplot(1, 3, 1)
    diagonal = figure.add_subplot(1, 3, 2, projection="3d")
    clearance = figure.add_subplot(1, 3, 3)
    top.plot(reference[:, 0], reference[:, 1], "--", color="green", label="3-D reference")
    top.plot(positions[:, 0], positions[:, 1], color="navy", label="executed trajectory")
    top.plot(obstacles[:, 0], obstacles[:, 1], color="red", alpha=0.35, label="moving obstacle")
    top.scatter(*positions[0, :2], color="blue", marker="o", label="start")
    top.scatter(*positions[-1, :2], color="black", marker="*", label="finish")
    top.set(xlabel="x [m]", ylabel="y [m]", title="(a) Top view: lateral avoidance")
    top.grid(alpha=0.3)
    top.legend(fontsize=8)
    diagonal.plot(reference[:, 0], reference[:, 1], reference[:, 2], "--", color="green")
    diagonal.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="navy")
    diagonal.plot(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], color="red", alpha=0.35)
    diagonal.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="(b) 3-D diagonal view")
    time = np.arange(len(clearances)) * dt
    clearance.plot(time, clearances, color="navy", label="true clearance")
    clearance.axhline(0.0, color="red", linestyle="--", label="collision boundary")
    clearance.set(xlabel="time [s]", ylabel="distance to obstacle [m]", title="(c) Clearance over time")
    clearance.grid(alpha=0.3)
    clearance.legend(fontsize=8)
    figure.suptitle("Safe Panda MPC-CBF 3-D trajectory comparison", fontsize=14)
    figure.savefig(root / "trajectory_3d_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-mode", choices=("behind_spline",), default="behind_spline")
    parser.add_argument("--goal-offset", nargs=3, type=float, default=(0.0, 0.30, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--obstacle-offset", nargs=3, type=float, default=(0.0, 0.15, 0.06), metavar=("X", "Y", "Z"))
    parser.add_argument("--obstacle-velocity", nargs=3, type=float, default=(0.05, 0.0, -0.015), metavar=("VX", "VY", "VZ"))
    parser.add_argument("--route-offset", nargs=3, action="append", type=float, metavar=("X", "Y", "Z"), help="intermediate waypoint offset from start; repeat twice or more")
    parser.add_argument("--position-q-weights", nargs=3, type=float, default=(1.0, 1.4, 1.2), metavar=("WX", "WY", "WZ"))
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--tangential-subgoal", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/3d_avoidance_demo")
    args = parser.parse_args()
    route_offsets = args.route_offset or [[0.14, 0.08, 0.10], [0.14, 0.23, 0.10]]
    if len(route_offsets) < 2:
        parser.error("--route-offset must be supplied at least twice")
    config = SmoothDynamicConfig(
        goal_offset=_vector(args.goal_offset, "--goal-offset"),
        obstacle_start_offset=_vector(args.obstacle_offset, "--obstacle-offset"),
        obstacle_velocity=_vector(args.obstacle_velocity, "--obstacle-velocity"),
        position_q_weights=_vector(args.position_q_weights, "--position-q-weights"),
        reference_mode=args.reference_mode,
        reference_route_profile="3d_waypoints",
        avoidance_waypoint_offsets=tuple(_vector(item, "--route-offset") for item in route_offsets),
        reflex_tangential_subgoal_enabled=args.tangential_subgoal,
        reflex_policy_library_enabled=args.tangential_subgoal,
        reflex_barrier_mode="collision_cone" if args.tangential_subgoal else "radial_cbf",
        max_steps=args.max_steps,
        render_stride=args.render_stride,
        save_animation=args.save_animation,
        save_plots=True,
        save_metrics=True,
        output_dir=args.output_dir,
    )
    result = run_smooth_dynamic_demo(config)
    _save_figure5_style_comparison(config.output_dir)
    payload = asdict(result)
    payload["success"] = bool(result.outcome == "goal" and result.reached_goal and not result.collision)
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
