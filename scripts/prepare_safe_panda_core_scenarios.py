#!/usr/bin/env python3
"""Freeze the 150 core-scenario instances and write preflight evidence."""

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from lampc_cbf.core_scenarios import (
    INSTANCES_PATH,
    PLAN_PATH,
    generate_instances,
    load_plan,
    pilot_episode_ids,
    smoke_episode_ids,
    write_frozen_instances,
)

OUTPUT_DIR = Path("artifacts/safe_panda_core_scenarios_150")


if __name__ == "__main__":
    plan = load_plan(PLAN_PATH)
    instances = generate_instances(plan)
    payload = write_frozen_instances(instances, path=INSTANCES_PATH, plan_path=PLAN_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight = {
        "plan_sha256": payload["plan_sha256"],
        "instances_sha256": payload["instances_sha256"],
        "instance_count": len(instances),
        "per_family_counts": {
            family["id"]: sum(
                instance.scenario_id == family["id"] for instance in instances
            )
            for family in plan["scenario_families"]
        },
        "side_balance": {
            family["id"]: {
                side: sum(
                    instance.scenario_id == family["id"]
                    and instance.crossing_side == side
                    for instance in instances
                )
                for side in (-1, 1)
            }
            for family in plan["scenario_families"]
            if any(
                spec["distribution"] == "balanced_categorical"
                for spec in family["perturbations"].values()
            )
        },
        "rejection_attempts": {
            "max": max(instance.preflight.attempts for instance in instances),
            "mean": mean(instance.preflight.attempts for instance in instances),
        },
        "initial_true_clearance_m": {
            "min": min(
                instance.preflight.initial_true_clearance_m for instance in instances
            ),
        },
        "encounter_margin_m": {
            "max": max(instance.preflight.encounter_margin_m for instance in instances),
        },
        "gate_definitions": {
            "goal_inflation_gate": "evaluated at t=0 with 0.05 m goal tolerance",
            "encounter_gate": "obstacle path vs straight start-goal segment on the 0.04 s episode grid",
        },
        "smoke_episode_ids": smoke_episode_ids(instances),
        "pilot_episode_ids": pilot_episode_ids(instances),
    }
    (OUTPUT_DIR / "preflight_summary.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    print(json.dumps(preflight, indent=2))
