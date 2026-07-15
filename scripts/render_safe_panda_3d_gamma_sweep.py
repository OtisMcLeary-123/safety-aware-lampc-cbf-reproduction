#!/usr/bin/env python3
"""Run one 3-D scenario over five gamma values plus a nominal baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo
from lampc_cbf.table4_scenarios import load_scenarios, scenario_runner_kwargs


def _load_benchmark_module():
    path = Path(__file__).with_name("run_safe_panda_3d_benchmark.py")
    spec = importlib.util.spec_from_file_location("safe_panda_3d_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load 3-D benchmark helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_config(setup: dict, scenario, output_dir: Path, gamma: float | None):
    benchmark = _load_benchmark_module()
    kwargs = scenario_runner_kwargs(scenario, seed_base=int(setup["benchmark"]["seed_base"]))
    kwargs.update(benchmark._profile_overrides(setup))
    scene = setup["scene"]
    kwargs.update(
        {
            "obstacle_start_offset": (
                float(scenario.obstacle_start_pos[0]),
                float(scenario.obstacle_start_pos[1]),
                float(scene["obstacle_height_offset_m"]),
            ),
            "obstacle_velocity": (
                float(scenario.obstacle_velocity[0]),
                float(scenario.obstacle_velocity[1]),
                float(scene["obstacle_vertical_velocity_mps"]),
            ),
            "obstacle_radius": float(scene["obstacle_radius_m"]),
            "collision_radius": float(scene["ee_collision_radius_m"]),
            "save_animation": False,
            "save_plots": False,
            "save_metrics": True,
            "output_dir": str(output_dir),
        }
    )
    if gamma is None:
        kwargs.update(
            {
                "gamma": 0.15,
                "reference_mode": "direct_target",
                "reference_route_profile": "legacy_lateral",
                "safety_mode": "none",
                "safety_reflex_enabled": False,
                "reflex_policy_library_enabled": False,
                "reflex_tangential_subgoal_enabled": False,
            }
        )
    else:
        kwargs["gamma"] = gamma
    return run_smooth_dynamic_demo(SmoothDynamicConfig(**kwargs))


def _plot(output_dir: Path, traces: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"gamma=0.001": "red", "gamma=0.040": "green", "gamma=0.065": "blue", "gamma=0.100": "cyan", "gamma=0.150": "magenta"}
    figure = plt.figure(figsize=(17, 5.3))
    top = figure.add_subplot(1, 3, 1)
    diagonal = figure.add_subplot(1, 3, 2, projection="3d")
    clearance = figure.add_subplot(1, 3, 3)
    for name, trace in traces.items():
        positions = np.asarray(trace["positions"], dtype=float)
        reference = np.asarray(trace["reference_path"], dtype=float)
        obstacles = np.asarray(trace["true_obstacles"], dtype=float)
        clearances = np.asarray(trace["true_clearances"], dtype=float)
        if name == "baseline":
            color, style, label = "black", "--", "baseline (direct target, no CBF)"
        else:
            color, style, label = colors[name], "-", name
        top.plot(positions[:, 0], positions[:, 1], color=color, linestyle=style, linewidth=2.0, label=label)
        diagonal.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, linestyle=style, linewidth=2.0)
        time = np.arange(len(clearances)) * 0.04
        clearance.plot(time, clearances, color=color, linestyle=style, linewidth=1.8, label=label)
    first = next(iter(traces.values()))
    reference = np.asarray(first["reference_path"], dtype=float)
    obstacles = np.asarray(first["true_obstacles"], dtype=float)
    top.plot(reference[:, 0], reference[:, 1], color="0.45", linestyle=":", linewidth=1.5, label="3-D reference")
    top.plot(obstacles[:, 0], obstacles[:, 1], color="0.65", linewidth=1.5, label="moving obstacle")
    diagonal.plot(reference[:, 0], reference[:, 1], reference[:, 2], color="0.45", linestyle=":", linewidth=1.5)
    diagonal.plot(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], color="0.65", linewidth=1.5)
    top.set(xlabel="x [m]", ylabel="y [m]", title="(a) Top view")
    top.grid(alpha=0.3)
    top.legend(fontsize=7)
    diagonal.set(xlabel="x [m]", ylabel="y [m]", title="(b) 3-D diagonal view")
    diagonal.set_zlabel("z [m]", labelpad=7)
    clearance.axhline(0.0, color="red", linestyle=":", linewidth=1.5, label="collision boundary")
    clearance.set(xlabel="time [s]", ylabel="distance to obstacle [m]", title="(c) Distance to obstacle")
    clearance.grid(alpha=0.3)
    clearance.legend(fontsize=7)
    figure.suptitle("Safe Panda 3-D gamma sweep: episode 1", fontsize=14)
    figure.subplots_adjust(left=0.05, right=0.98, bottom=0.13, top=0.86, wspace=0.32)
    figure.savefig(output_dir / "trajectory_gamma_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", type=int, default=1)
    parser.add_argument("--setup-file", default="configs/safe_panda_3d_avoidance_50_setup.json")
    parser.add_argument("--output-dir", default="artifacts/safe_panda_3d_feedback_episode_01/gamma_sweep")
    args = parser.parse_args()
    if not 1 <= args.episode_id <= 50:
        parser.error("episode-id must be in [1, 50]")
    setup = json.loads(Path(args.setup_file).read_text(encoding="utf-8"))
    scenarios = load_scenarios(setup["scenario_path"])
    scenario = scenarios[args.episode_id - 1]
    PaperFidelityManifest.load(setup["llm"]["provider_manifest"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gammas = (0.001, 0.04, 0.065, 0.10, 0.15)
    traces: dict[str, dict] = {}
    summary: list[dict] = []
    for gamma in gammas:
        name = f"gamma={gamma:0.3f}"
        run_dir = output_dir / name.replace("=", "_")
        result = _run_config(setup, scenario, run_dir, gamma)
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        traces[name] = payload
        summary.append({"label": name, "gamma": gamma, "outcome": result.outcome, "success": result.outcome == "goal" and result.reached_goal and not result.collision, "collision": result.collision, "steps": result.steps, "minimum_true_clearance_m": result.minimum_true_clearance})
    baseline_dir = output_dir / "baseline"
    result = _run_config(setup, scenario, baseline_dir, None)
    payload = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    traces["baseline"] = payload
    summary.append({"label": "baseline", "gamma": None, "outcome": result.outcome, "success": result.outcome == "goal" and result.reached_goal and not result.collision, "collision": result.collision, "steps": result.steps, "minimum_true_clearance_m": result.minimum_true_clearance})
    _plot(output_dir, traces)
    with (output_dir / "gamma_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "gamma_sweep_summary.json").write_text(json.dumps({"episode_id": args.episode_id, "profile": setup["profile"], "runs": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"episode_id": args.episode_id, "output_dir": str(output_dir), "runs": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
