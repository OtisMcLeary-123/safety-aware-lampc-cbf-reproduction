#!/usr/bin/env python3
"""Render one successful async-feedback episode from the 3-D benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from lampc_cbf.hf_llm import GammaDecision
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


def _load_decision(path: Path, episode_id: int, manifest: PaperFidelityManifest) -> GammaDecision:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 50:
        raise ValueError("feedback checkpoint must contain exactly 50 decisions")
    item = dict(payload[episode_id - 1])
    decision = GammaDecision(**item)
    if decision.fallback_used or decision.cache_hit or not manifest.accepts_feedback_decision(
        model=decision.model, provider=decision.provider, cache_hit=decision.cache_hit
    ):
        raise ValueError("selected provider decision is not valid for the manifest")
    return decision


def _save_figure5_style_comparison(output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    positions = np.asarray(payload["positions"], dtype=float)
    reference = np.asarray(payload["reference_path"], dtype=float)
    obstacles = np.asarray(payload["true_obstacles"], dtype=float)
    clearances = np.asarray(payload["true_clearances"], dtype=float)
    figure = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    top = figure.add_subplot(1, 3, 1)
    diagonal = figure.add_subplot(1, 3, 2, projection="3d")
    clearance = figure.add_subplot(1, 3, 3)
    top.plot(reference[:, 0], reference[:, 1], "--", color="green", label="3-D reference")
    top.plot(positions[:, 0], positions[:, 1], color="navy", label="executed feedback trajectory")
    top.plot(obstacles[:, 0], obstacles[:, 1], color="red", alpha=0.35, label="moving obstacle")
    top.scatter(*positions[0, :2], color="blue", marker="o", label="start")
    top.scatter(*positions[-1, :2], color="black", marker="*", label="finish")
    top.set(xlabel="x [m]", ylabel="y [m]", title="(a) Top view")
    top.grid(alpha=0.3)
    top.legend(fontsize=8)
    diagonal.plot(reference[:, 0], reference[:, 1], reference[:, 2], "--", color="green")
    diagonal.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="navy")
    diagonal.plot(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], color="red", alpha=0.35)
    diagonal.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="(b) 3-D diagonal view")
    time = np.arange(len(clearances)) * 0.04
    clearance.plot(time, clearances, color="navy", label="true clearance")
    clearance.axhline(0.0, color="red", linestyle="--", label="collision boundary")
    clearance.set(xlabel="time [s]", ylabel="distance to obstacle [m]", title="(c) Clearance")
    clearance.grid(alpha=0.3)
    clearance.legend(fontsize=8)
    figure.suptitle("Safe Panda 3-D async-feedback episode", fontsize=14)
    figure.savefig(output_dir / "trajectory_3d_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", type=int, default=1)
    parser.add_argument("--setup-file", default="configs/safe_panda_3d_avoidance_50_setup.json")
    parser.add_argument("--output-dir", default="artifacts/safe_panda_3d_feedback_episode_01")
    parser.add_argument("--save-animation", action="store_true", default=True)
    args = parser.parse_args()
    if not 1 <= args.episode_id <= 50:
        parser.error("episode-id must be in [1, 50]")
    setup = json.loads(Path(args.setup_file).read_text(encoding="utf-8"))
    scenarios = load_scenarios(setup["scenario_path"])
    scenario = scenarios[args.episode_id - 1]
    manifest = PaperFidelityManifest.load(setup["llm"]["provider_manifest"])
    decision = _load_decision(Path(setup["llm"]["checkpoint"]), args.episode_id, manifest)
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
            "gamma_schedule": ((scenario.intervention_time + decision.latency_seconds, decision.gamma),),
            "gamma_schedule_request_times": (scenario.intervention_time,),
            "save_animation": True,
            "save_plots": True,
            "save_metrics": True,
            "output_dir": args.output_dir,
        }
    )
    result = run_smooth_dynamic_demo(SmoothDynamicConfig(**kwargs))
    output_dir = Path(args.output_dir)
    _save_figure5_style_comparison(output_dir)
    summary = {
        "episode_id": args.episode_id,
        "method": "safe_panda_3d_async_feedback",
        "provider_model": decision.model,
        "provider_latency_seconds": decision.latency_seconds,
        "feedback_gamma": decision.gamma,
        "outcome": result.outcome,
        "success": bool(result.outcome == "goal" and result.reached_goal and not result.collision),
        "collision": result.collision,
        "output_dir": str(output_dir),
    }
    (output_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
