import hashlib
import json

import pytest

from lampc_cbf.language_dsl import DSL_VERSION, SceneObject
from lampc_cbf.language_replay import RecordedSafeNarratePlanner


def _target(name, relation="center"):
    return {
        "kind": "object",
        "object": name,
        "relation": relation,
        "vector_m": [0.0, 0.0, 0.0],
    }


def test_recorded_planner_revalidates_accepted_tp_od_without_network(tmp_path):
    instruction = "Safely move blue onto red."
    pick = _target("blue_cube")
    place = _target("red_cube", "above")
    steps = [
        {"action": "move", "target": pick, "avoid": ["moving_obstacle"]},
        {"action": "close_gripper", "target": None, "avoid": []},
        {"action": "move", "target": place, "avoid": ["moving_obstacle"]},
        {"action": "open_gripper", "target": None, "avoid": []},
    ]

    def raw_spec(target):
        return json.dumps({
            "version": DSL_VERSION,
            "objective": {
                "kind": "squared_position_error",
                "target": target,
                "q_weight": 1.0,
                "linear_delta_u_weight": 1.0,
            },
            "safety": {"gamma": 0.05},
            "limits": {
                "workspace_lower_m": [-0.3, -0.3, 0.0],
                "workspace_upper_m": [0.25, 0.3, 0.5],
                "linear_speed_limit_mps": 0.1,
            },
            "constraints": [{
                "kind": "collision_clearance",
                "object": "moving_obstacle",
                "clearance_m": 0.05,
                "value_m": None,
            }],
        })

    serialized_steps = []
    for step in steps:
        target = step["target"]
        serialized_target = None if target is None else {
            "kind": target["kind"],
            "object_name": target["object"],
            "relation": target["relation"],
            "vector_m": target["vector_m"],
        }
        serialized_steps.append({**step, "target": serialized_target})
    metrics = {
        "language_result": {
            "task_plan": {"version": DSL_VERSION, "steps": serialized_steps},
            "tp_latency_seconds": 1.2,
            "od_latency_seconds": 2.3,
            "od_fallbacks": 0,
            "model": "recorded-model",
            "provider": "recorded-provider",
            "instruction_hash": hashlib.sha256(instruction.encode()).hexdigest(),
            "od_attempts": [
                {
                    "task_step_index": 0,
                    "status": "accepted",
                    "latency_seconds": 1.0,
                    "raw_response": raw_spec(pick),
                    "cause_type": None,
                    "cause_message": None,
                },
                {
                    "task_step_index": 2,
                    "status": "accepted",
                    "latency_seconds": 1.3,
                    "raw_response": raw_spec(place),
                    "cause_type": None,
                    "cause_message": None,
                },
            ],
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    scene = (
        SceneObject("blue_cube", (-0.14, -0.12, 0.02), 0.018),
        SceneObject("red_cube", (-0.08, 0.14, 0.02), 0.018),
        SceneObject("moving_obstacle", (-0.1, 0.0, 0.18), 0.055),
    )

    planner = RecordedSafeNarratePlanner(path)
    result = planner.formulate(
        instruction,
        scene,
        current_position=(0.0, 0.0, 0.2),
        required_hazards=("moving_obstacle",),
    )
    assert planner.execution_source == "recorded_replay"
    assert result.od_fallbacks == 0
    assert len(result.od_attempts) == 2
    assert result.optimization_specs[0].safety.gamma == pytest.approx(0.05)
    assert result.provider == "recorded_replay:recorded-provider"


def test_recorded_planner_rejects_instruction_mismatch(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"language_result": {
        "model": "m",
        "instruction_hash": "wrong",
    }}), encoding="utf-8")
    planner = RecordedSafeNarratePlanner(path)
    with pytest.raises(ValueError, match="instruction hash"):
        planner.formulate("different", (), current_position=(0, 0, 0))
