import json

import pytest

from lampc_cbf.language_dsl import (
    DSL_VERSION,
    HuggingFaceSafeNarratePlanner,
    LanguageDSLInferenceError,
    SafeNarrateConfig,
    SceneObject,
    TP_SYSTEM_PROMPT,
    compile_optimization,
    controller_config_from_optimization,
    optimization_from_task_step,
    parse_optimization_spec,
    parse_task_plan,
    resolve_target,
)


@pytest.fixture
def scene():
    return (
        SceneObject("blue_cube", (0.1, 0.2, 0.03), 0.03),
        SceneObject("red_cube", (0.2, 0.25, 0.03), 0.03),
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


def _collision(name="moving_obstacle", clearance=0.04):
    return {
        "kind": "collision_clearance",
        "object": name,
        "clearance_m": clearance,
        "value_m": None,
    }


def _optimization_payload(
    *, target=None, q_weight=1.0, delta_u_weight=0.5, gamma=0.05,
    lower=None, upper=None, speed_limit=0.2, constraints=None,
):
    return {
        "version": DSL_VERSION,
        "objective": {
            "kind": "squared_position_error",
            "target": _target() if target is None else target,
            "q_weight": q_weight,
            "linear_delta_u_weight": delta_u_weight,
        },
        "safety": {"gamma": gamma},
        "limits": {
            "workspace_lower_m": [-0.3, -0.3, 0.0] if lower is None else lower,
            "workspace_upper_m": [0.25, 0.3, 0.5] if upper is None else upper,
            "linear_speed_limit_mps": speed_limit,
        },
        "constraints": [] if constraints is None else constraints,
    }


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
            {"version": DSL_VERSION, "steps": [_move(_target("missing"), [])]}, scene
        )
    payload = {
        "version": DSL_VERSION,
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
        {"version": DSL_VERSION, "steps": [_move(_target(), ["moving_obstacle"])]}, scene
    ).steps[0]
    fallback = optimization_from_task_step(task, scene)
    assert fallback.constraints[0].kind == "collision_clearance"

    payload = _optimization_payload(
        target=_target("red_cube", "above", [0.0, 0.0, 0.02]),
        q_weight=2.0,
        delta_u_weight=2.0,
        gamma=0.02,
        constraints=[_collision()],
    )
    spec = parse_optimization_spec(
        payload,
        scene,
        current_position=(0.0, 0.0, 0.1),
        required_avoid=("moving_obstacle",),
    )
    assert spec.objective.q_weight == 2.0
    assert spec.objective.linear_delta_u_weight == 2.0
    assert spec.safety.gamma == pytest.approx(0.02)
    assert resolve_target(spec.objective.target, scene, (0.0, 0.0, 0.0)) == pytest.approx(
        (0.2, 0.25, 0.08)
    )


def test_optimization_spec_rejects_expression_strings(scene):
    payload = _optimization_payload()
    payload["objective"]["kind"] = "ca.norm_2(x)**2"
    with pytest.raises(ValueError, match="not whitelisted"):
        parse_optimization_spec(payload, scene, current_position=(0, 0, 0.1))


def test_compile_optimization_uses_whitelist(scene):
    ca = pytest.importorskip("casadi")
    x = ca.MX.sym("x", 8)
    spec = parse_optimization_spec(
        _optimization_payload(
            target=_target("red_cube"),
            constraints=[{
                "kind": "minimum_height",
                "object": None,
                "clearance_m": 0.0,
                "value_m": 0.05,
            }],
        ),
        scene,
        current_position=(0.0, 0.0, 0.1),
    )
    compiled = compile_optimization(spec, scene, (0.0, 0.0, 0.0), x, ca)
    fun = ca.Function("dsl", [x], [compiled.objective, *compiled.inequalities])
    values = fun([0.2, 0.25, 0.03, 0, 0, 0, 0, 0])
    assert float(values[0]) == pytest.approx(0.0)
    assert float(values[1]) == pytest.approx(0.02)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"gamma": 0.03}, "safety.gamma"),
        ({"q_weight": 10.1}, "q_weight"),
        ({"delta_u_weight": 3.0}, "linear_delta_u_weight"),
        ({"speed_limit": 0.21}, "linear_speed_limit_mps"),
        ({"lower": [-0.31, -0.3, 0.0]}, "Safe Panda envelope"),
        ({"upper": [0.25, 0.31, 0.5]}, "Safe Panda envelope"),
    ],
)
def test_a1_parameter_bounds_fail_closed(scene, changes, message):
    payload = _optimization_payload(**changes)
    with pytest.raises(ValueError, match=message):
        parse_optimization_spec(payload, scene, current_position=(0, 0, 0.1))


def test_a1_rejects_outside_target_and_missing_required_obstacle(scene):
    outside = _optimization_payload(
        target={
            "kind": "absolute",
            "object": None,
            "relation": "center",
            "vector_m": [0.26, 0.0, 0.1],
        }
    )
    with pytest.raises(ValueError, match="outside the selected workspace"):
        parse_optimization_spec(outside, scene, current_position=(0, 0, 0.1))

    missing = _optimization_payload()
    with pytest.raises(ValueError, match="omits a required"):
        parse_optimization_spec(
            missing,
            scene,
            current_position=(0, 0, 0.1),
            required_avoid=("moving_obstacle",),
        )


def test_a1_controller_adapter_preserves_fixed_paper_parameters(scene):
    spec = parse_optimization_spec(
        _optimization_payload(
            q_weight=4.0,
            delta_u_weight=5.0,
            gamma=0.12,
            lower=[-0.2, -0.25, 0.0],
            upper=[0.22, 0.28, 0.4],
            speed_limit=0.08,
        ),
        scene,
        current_position=(0, 0, 0.1),
    )
    config = controller_config_from_optimization(
        spec, scene, current_position=(0, 0, 0.1)
    )
    assert config.q_weight == pytest.approx(4.0)
    assert config.linear_delta_u_weight == pytest.approx(5.0)
    assert config.position_lower == pytest.approx((-0.2, -0.25, 0.0))
    assert config.position_upper == pytest.approx((0.22, 0.28, 0.4))
    assert config.linear_input_limit == pytest.approx(0.08)
    assert config.dt == pytest.approx(0.04)
    assert config.horizon == 15
    assert config.yaw_delta_u_weight == pytest.approx(1e-5)
    assert config.velocity_regularization == pytest.approx(0.1)
    assert config.yaw_regularization == pytest.approx(5e-5)
    assert config.linear_jerk_weight == 0.0
    assert config.optimal_decay_weight == 0.0


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
        "version": DSL_VERSION,
        "steps": [_move(_target(), ["moving_obstacle"])],
    }
    client = _FakeClient([json.dumps(task_payload), '{"objective":"ca.eval(...)"}'])
    planner = HuggingFaceSafeNarratePlanner(
        SafeNarrateConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )
    result = planner.formulate("pick blue cube", scene, current_position=(0, 0, 0.1))
    assert result.od_fallbacks == 1
    assert len(result.od_attempts) == 1
    assert result.od_attempts[0].status == "fallback"
    assert result.od_attempts[0].raw_response == '{"objective":"ca.eval(...)"}'
    assert result.od_attempts[0].cause_type == "ValueError"
    assert result.optimization_specs[0].constraints[0].object_name == "moving_obstacle"
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[1]["response_format"] == {"type": "json_object"}
    assert "output_schema" in json.loads(client.calls[0]["messages"][1]["content"])


def test_two_stage_hf_planner_audits_accepted_od(scene, tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("test-token", encoding="utf-8")
    task_payload = {
        "version": DSL_VERSION,
        "steps": [_move(_target(), ["moving_obstacle"])],
    }
    od_payload = _optimization_payload(constraints=[_collision()])
    client = _FakeClient([json.dumps(task_payload), json.dumps(od_payload)])
    planner = HuggingFaceSafeNarratePlanner(
        SafeNarrateConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )
    result = planner.formulate(
        "pick blue cube",
        scene,
        current_position=(0, 0, 0.1),
        required_hazards=("moving_obstacle",),
    )
    assert result.od_fallbacks == 0
    assert len(result.od_attempts) == 1
    assert result.od_attempts[0].status == "accepted"
    assert json.loads(result.od_attempts[0].raw_response) == od_payload
    assert result.od_attempts[0].cause_type is None


def test_two_stage_hf_planner_fails_closed_for_invalid_tp(scene, tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("test-token", encoding="utf-8")
    client = _FakeClient(['{"steps":["run arbitrary code"]}'])
    planner = HuggingFaceSafeNarratePlanner(
        SafeNarrateConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )
    with pytest.raises(LanguageDSLInferenceError, match="failed closed") as captured:
        planner.formulate("unsafe output", scene, current_position=(0, 0, 0))
    assert captured.value.stage == "task_planner"
    assert captured.value.cause_type == "ValueError"
    assert captured.value.raw_response == '{"steps":["run arbitrary code"]}'


def test_two_stage_hf_planner_rejects_missing_required_hazard(scene, tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("test-token", encoding="utf-8")
    task_payload = {
        "version": DSL_VERSION,
        "steps": [_move(_target(), [])],
    }
    client = _FakeClient([json.dumps(task_payload)])
    planner = HuggingFaceSafeNarratePlanner(
        SafeNarrateConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )
    with pytest.raises(LanguageDSLInferenceError, match="failed closed"):
        planner.formulate(
            "pick blue cube",
            scene,
            current_position=(0, 0, 0.1),
            required_hazards=("moving_obstacle",),
        )


def test_tp_prompt_encodes_the_strict_e1_gripper_grammar():
    assert "close_gripper with target=null and avoid=[]" in TP_SYSTEM_PROMPT
    assert "open_gripper with target=null and avoid=[]" in TP_SYSTEM_PROMPT
    assert "exactly this action grammar" in TP_SYSTEM_PROMPT
    HuggingFaceSafeNarratePlanner,
    LanguageDSLInferenceError,
    SafeNarrateConfig,
