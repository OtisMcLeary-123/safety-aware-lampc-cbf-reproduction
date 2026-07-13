import json

import pytest

from lampc_cbf.language_dsl import (
    DSL_VERSION,
    HuggingFaceSafeNarratePlanner,
    LanguageDSLInferenceError,
    SafeNarrateConfig,
    SceneObject,
    compile_optimization,
    optimization_from_task_step,
    parse_optimization_spec,
    parse_task_plan,
    resolve_target,
)


@pytest.fixture
def scene():
    return (
        SceneObject("blue_cube", (0.1, 0.2, 0.03), 0.03),
        SceneObject("red_cube", (0.2, 0.4, 0.03), 0.03),
        SceneObject("moving_obstacle", (0.15, 0.3, 0.12), 0.055),
    )


def _target(name="blue_cube", relation="center", vector=None):
    return {
        "kind": "object",
        "object": name,
        "relation": relation,
        "vector_m": [0.0, 0.0, 0.0] if vector is None else vector,
    }


def _move(target, avoid):
    return {"action": "move", "target": target, "avoid": avoid}


def _gripper(action):
    return {"action": action, "target": None, "avoid": []}


def test_parse_task_plan_tracks_grasp_and_avoid_semantics(scene):
    payload = {
        "version": DSL_VERSION,
        "steps": [
            _move(_target(), ["red_cube", "moving_obstacle"]),
            _gripper("close_gripper"),
            _move(_target("red_cube", "above"), ["moving_obstacle"]),
            _gripper("open_gripper"),
        ],
    }
    plan = parse_task_plan(json.dumps(payload), scene)
    assert len(plan.steps) == 4
    assert plan.steps[0].target.object_name == "blue_cube"
    assert plan.steps[2].avoid == ("moving_obstacle",)


def test_task_plan_rejects_source_injection_and_extra_keys(scene):
    payload = {
        "version": DSL_VERSION,
        "steps": [
            {
                **_move(_target(), []),
                "python": "__import__('os').system('touch /tmp/pwned')",
            }
        ],
    }
    with pytest.raises(ValueError, match="extra=.*python"):
        parse_task_plan(payload, scene)


def test_task_plan_rejects_unknown_objects_and_held_object_avoidance(scene):
    with pytest.raises(ValueError, match="unknown object"):
        parse_task_plan(
            {"version": 1, "steps": [_move(_target("missing"), [])]}, scene
        )
    payload = {
        "version": 1,
        "steps": [
            _move(_target(), []),
            _gripper("close_gripper"),
            _move(_target("red_cube", "above"), ["blue_cube"]),
        ],
    }
    with pytest.raises(ValueError, match="held object"):
        parse_task_plan(payload, scene)


def test_parse_optimization_spec_and_deterministic_od(scene):
    task = parse_task_plan(
        {"version": 1, "steps": [_move(_target(), ["moving_obstacle"])]}, scene
    ).steps[0]
    fallback = optimization_from_task_step(task, scene)
    assert fallback.constraints[0].kind == "collision_clearance"

    payload = {
        "version": 1,
        "objective": {
            "kind": "squared_position_error",
            "target": _target("red_cube", "above", [0.0, 0.0, 0.02]),
            "weight": 2.0,
        },
        "constraints": [
            {
                "kind": "collision_clearance",
                "object": "moving_obstacle",
                "clearance_m": 0.04,
                "value_m": None,
                "lower_m": None,
                "upper_m": None,
            },
            {
                "kind": "workspace_box",
                "object": None,
                "clearance_m": 0.0,
                "value_m": None,
                "lower_m": [-0.5, 0.0, 0.0],
                "upper_m": [0.5, 1.0, 0.8],
            },
        ],
    }
    spec = parse_optimization_spec(payload, scene)
    assert spec.objective.weight == 2.0
    assert resolve_target(spec.objective.target, scene, (0.0, 0.0, 0.0)) == pytest.approx(
        (0.2, 0.4, 0.08)
    )


def test_optimization_spec_rejects_expression_strings(scene):
    payload = {
        "version": 1,
        "objective": {
            "kind": "ca.norm_2(x)**2",
            "target": _target(),
            "weight": 1.0,
        },
        "constraints": [],
    }
    with pytest.raises(ValueError, match="not whitelisted"):
        parse_optimization_spec(payload, scene)


def test_compile_optimization_uses_whitelist(scene):
    ca = pytest.importorskip("casadi")
    x = ca.MX.sym("x", 8)
    spec = parse_optimization_spec(
        {
            "version": 1,
            "objective": {
                "kind": "squared_position_error",
                "target": _target("red_cube"),
                "weight": 1.0,
            },
            "constraints": [
                {
                    "kind": "minimum_height",
                    "object": None,
                    "clearance_m": 0.0,
                    "value_m": 0.05,
                    "lower_m": None,
                    "upper_m": None,
                }
            ],
        },
        scene,
    )
    compiled = compile_optimization(spec, scene, (0.0, 0.0, 0.0), x, ca)
    fun = ca.Function("dsl", [x], [compiled.objective, *compiled.inequalities])
    values = fun([0.2, 0.4, 0.03, 0, 0, 0, 0, 0])
    assert float(values[0]) == pytest.approx(0.0)
    assert float(values[1]) == pytest.approx(0.02)


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(next(self.responses))


def test_two_stage_hf_planner_validates_tp_and_falls_back_for_bad_od(scene, tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("test-token", encoding="utf-8")
    task_payload = {
        "version": 1,
        "steps": [_move(_target(), ["moving_obstacle"])],
    }
    client = _FakeClient([json.dumps(task_payload), '{"objective":"ca.eval(...)"}'])
    planner = HuggingFaceSafeNarratePlanner(
        SafeNarrateConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )
    result = planner.formulate("pick blue cube", scene, current_position=(0, 0, 0.1))
    assert result.od_fallbacks == 1
    assert result.optimization_specs[0].constraints[0].object_name == "moving_obstacle"
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"]["json_schema"]["name"] == "SafeTaskPlan"
    assert client.calls[1]["response_format"]["json_schema"]["name"] == "SafeOptimizationSpec"


def test_two_stage_hf_planner_fails_closed_for_invalid_tp(scene, tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("test-token", encoding="utf-8")
    client = _FakeClient(['{"steps":["run arbitrary code"]}'])
    planner = HuggingFaceSafeNarratePlanner(
        SafeNarrateConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )
    with pytest.raises(LanguageDSLInferenceError, match="failed closed"):
        planner.formulate("unsafe output", scene, current_position=(0, 0, 0))
    HuggingFaceSafeNarratePlanner,
    LanguageDSLInferenceError,
    SafeNarrateConfig,
