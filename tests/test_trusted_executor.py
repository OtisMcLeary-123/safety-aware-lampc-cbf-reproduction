from dataclasses import replace

import pytest

from lampc_cbf.language_dsl import (
    DSL_VERSION,
    SafeNarrateResult,
    SceneObject,
    parse_optimization_spec,
    parse_task_plan,
)
from lampc_cbf.trusted_executor import build_trusted_pick_place_macros


SCENE = (
    SceneObject("blue_cube", (-0.14, -0.12, 0.02), 0.018),
    SceneObject("red_cube", (-0.08, 0.14, 0.02), 0.018),
    SceneObject("moving_obstacle", (-0.10, 0.01, 0.18), 0.055),
)


def _target(name, relation="center"):
    return {
        "kind": "object",
        "object": name,
        "relation": relation,
        "vector_m": [0.0, 0.0, 0.0],
    }


def _result():
    move_pick = {
        "action": "move",
        "target": _target("blue_cube"),
        "avoid": ["moving_obstacle"],
    }
    move_place = {
        "action": "move",
        "target": _target("red_cube", "above"),
        "avoid": ["moving_obstacle"],
    }
    plan = parse_task_plan(
        {
            "version": DSL_VERSION,
            "steps": [
                move_pick,
                {"action": "close_gripper", "target": None, "avoid": []},
                move_place,
                {"action": "open_gripper", "target": None, "avoid": []},
            ],
        },
        SCENE,
    )

    def spec(target):
        return parse_optimization_spec(
            {
                "version": DSL_VERSION,
                "objective": {
                    "kind": "squared_position_error",
                    "target": target,
                    "q_weight": 1.0,
                    "linear_delta_u_weight": 0.5,
                },
                "safety": {"gamma": 0.05},
                "limits": {
                    "workspace_lower_m": [-0.3, -0.3, 0.0],
                    "workspace_upper_m": [0.25, 0.3, 0.5],
                    "linear_speed_limit_mps": 0.2,
                },
                "constraints": [{
                    "kind": "collision_clearance",
                    "object": "moving_obstacle",
                    "clearance_m": 0.04,
                    "value_m": None,
                }],
            },
            SCENE,
            current_position=(0.0, 0.0, 0.2),
            required_avoid=("moving_obstacle",),
        )

    return SafeNarrateResult(
        task_plan=plan,
        optimization_specs=(
            spec(move_pick["target"]),
            None,
            spec(move_place["target"]),
            None,
        ),
        tp_latency_seconds=0.1,
        od_latency_seconds=0.2,
        od_fallbacks=0,
        model="fake",
        provider="fake",
        instruction_hash="abc",
    )


def test_trusted_executor_builds_approved_macro():
    macros = build_trusted_pick_place_macros(
        _result(), required_hazards=("moving_obstacle",)
    )
    assert len(macros) == 1
    assert macros[0].source.object_name == "blue_cube"
    assert macros[0].destination.object_name == "red_cube"
    assert macros[0].place_optimization.safety.gamma == pytest.approx(0.05)


def test_trusted_executor_rejects_misaligned_optimization():
    result = _result()
    specs = list(result.optimization_specs)
    specs[0] = specs[2]
    with pytest.raises(ValueError, match="does not match"):
        build_trusted_pick_place_macros(
            replace(result, optimization_specs=tuple(specs)),
            required_hazards=("moving_obstacle",),
        )


def test_trusted_executor_rejects_incomplete_macro():
    result = _result()
    with pytest.raises(ValueError, match="four-step"):
        build_trusted_pick_place_macros(
            replace(
                result,
                task_plan=replace(result.task_plan, steps=result.task_plan.steps[:-1]),
                optimization_specs=result.optimization_specs[:-1],
            )
        )
