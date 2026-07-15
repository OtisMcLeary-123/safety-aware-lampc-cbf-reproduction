#!/usr/bin/env python3
"""Collect a resumable 50-decision provider trace for the opt-in 3-D profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep

from lampc_cbf.contextual_gamma import (
    ContextualGammaConfig,
    ContextualNvidiaNIMGammaMapper,
    FeedbackHazardContext,
)
from lampc_cbf.paired_benchmark import load_feedback_checkpoint
from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.smooth_dynamic_demo import build_reference_route
from lampc_cbf.table4_scenarios import Table4Scenario, load_scenarios


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _nominal_3d_feedback_context(
    scenario: Table4Scenario, setup: dict[str, object]
) -> FeedbackHazardContext:
    """Approximate the intervention state on the declared 3-D spline."""

    import numpy as np

    controller = setup["controller"]
    scene = setup["scene"]
    assert isinstance(controller, dict) and isinstance(scene, dict)
    start = np.zeros(3)
    goal = np.asarray(controller["goal_offset_m"], dtype=float)
    waypoints = tuple(
        tuple(float(value) for value in item)
        for item in controller["avoidance_waypoint_offsets_m"]
    )
    combined_radius = float(scene["obstacle_radius_m"]) + float(
        scene["ee_collision_radius_m"]
    )
    obstacle_velocity = np.asarray(
        [
            scenario.obstacle_velocity[0],
            scenario.obstacle_velocity[1],
            float(scene["obstacle_vertical_velocity_mps"]),
        ],
        dtype=float,
    )
    _, reference = build_reference_route(
        start,
        goal,
        obstacle_velocity=obstacle_velocity,
        combined_radius=combined_radius,
        route_margin=0.08,
        profile=str(controller["reference_route_profile"]),
        waypoint_offsets=waypoints,
    )
    segments = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(segments)])
    target_distance = min(
        arc_length[-1],
        float(controller["reference_speed_mps"]) * scenario.intervention_time,
    )
    index = min(
        len(reference) - 1,
        int(np.searchsorted(arc_length, target_distance, side="left")),
    )
    robot_position = reference[index]
    next_index = min(index + 1, len(reference) - 1)
    tangent = reference[next_index] - robot_position
    tangent_norm = float(np.linalg.norm(tangent))
    robot_velocity = (
        tangent / tangent_norm * float(controller["reference_speed_mps"])
        if tangent_norm > 1e-12
        else np.zeros(3)
    )
    obstacle_start = np.asarray(
        [
            scenario.obstacle_start_pos[0],
            scenario.obstacle_start_pos[1],
            float(scene["obstacle_height_offset_m"]),
        ],
        dtype=float,
    )
    obstacle_position = obstacle_start + obstacle_velocity * scenario.intervention_time
    relative_position = obstacle_position - robot_position
    distance = float(np.linalg.norm(relative_position))
    clearance = distance - combined_radius
    relative_velocity = obstacle_velocity - robot_velocity
    closing_speed = (
        max(0.0, -float(np.dot(relative_position, relative_velocity)) / distance)
        if distance > 1e-12
        else float(np.linalg.norm(relative_velocity))
    )
    ttc = max(0.0, clearance) / closing_speed if closing_speed > 1e-12 else None
    return FeedbackHazardContext(
        current_gamma=float(controller["initial_gamma"]),
        obstacle_distance_m=distance,
        combined_radius_m=combined_radius,
        clearance_m=clearance,
        predicted_ttc_s=ttc,
        obstacle_speed_mps=float(np.linalg.norm(obstacle_velocity)),
        minimum_cbf_residual=None,
        intervention_time_s=scenario.intervention_time,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-file", default="configs/safe_panda_3d_avoidance_50_setup.json")
    parser.add_argument("--scenario-file")
    parser.add_argument("--provider-manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--request-interval-seconds", type=float, default=1.0)
    parser.add_argument("--restart", action="store_true", help="replace an existing provider checkpoint")
    args = parser.parse_args()
    if args.request_interval_seconds < 0.0:
        parser.error("request interval must be non-negative")

    setup = json.loads(Path(args.setup_file).read_text(encoding="utf-8"))
    scenario_file = args.scenario_file or setup["scenario_path"]
    manifest_file = args.provider_manifest or setup["llm"]["provider_manifest"]
    checkpoint = Path(args.checkpoint or setup["llm"]["checkpoint"])
    scenarios = load_scenarios(scenario_file)
    expected = int(setup["benchmark"]["episodes"])
    if expected != 50 or len(scenarios) != expected:
        raise ValueError("3-D provider workflow requires exactly 50 scenarios")
    manifest = PaperFidelityManifest.load(manifest_file)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    decisions = (
        []
        if args.restart or not checkpoint.exists()
        else load_feedback_checkpoint(checkpoint, episodes=expected, paper_manifest=manifest)
    )
    mapper = ContextualNvidiaNIMGammaMapper(
        ContextualGammaConfig(
            model=manifest.required_model_family,
            timeout_seconds=manifest.llm_timeout_seconds,
            max_tokens=manifest.llm_max_tokens,
        )
    )
    query = str(setup["llm"]["feedback_query"])
    for index in range(len(decisions), expected):
        decision = mapper.infer_gamma(
            query, _nominal_3d_feedback_context(scenarios[index], setup)
        )
        if decision.fallback_used:
            raise RuntimeError(f"provider collection failed at episode {index + 1}; checkpoint retained")
        decisions.append(decision)
        _write_json_atomic(checkpoint, [item.as_dict() for item in decisions])
        print(f"[safe-panda-3d] collected {index + 1}/{expected}", flush=True)
        if index + 1 < expected:
            sleep(args.request_interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
