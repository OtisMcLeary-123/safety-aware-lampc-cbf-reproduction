"""Safe JSON DSL for NARRATE-style task and optimization formulation.

The original NARRATE prompts ask an LLM to emit natural-language tasks and
CasADi source strings.  This port keeps the Task Planner (TP) -> Optimization
Designer (OD) split but replaces source strings with a small, closed JSON
language.  The parser rejects unknown keys and the compiler never calls
``eval`` or ``exec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence


DSL_VERSION = 2
MAX_TASK_STEPS = 24
MAX_CONSTRAINTS = 24
MAX_JSON_BYTES = 32_768
MAX_NAME_LENGTH = 64

GAMMA_LEVELS = (0.02, 0.05, 0.08, 0.12, 0.15)
DELTA_U_LEVELS = (0.5, 1.0, 2.0, 5.0)
Q_WEIGHT_MIN = 0.1
Q_WEIGHT_MAX = 10.0
CLEARANCE_MIN_M = 0.012
CLEARANCE_MAX_M = 0.10
LINEAR_SPEED_MIN_MPS = 0.02
LINEAR_SPEED_MAX_MPS = 0.20
SAFE_PANDA_WORKSPACE_LOWER_M = (-0.30, -0.30, 0.00)
SAFE_PANDA_WORKSPACE_UPPER_M = (0.25, 0.30, 0.50)

TASK_ACTIONS = frozenset({"move", "open_gripper", "close_gripper"})
TARGET_KINDS = frozenset({"object", "absolute", "current_offset"})
RELATIONS = frozenset({"center", "above", "front", "behind", "left", "right"})
CONSTRAINT_KINDS = frozenset(
    {"collision_clearance", "minimum_height", "maximum_height"}
)


TASK_PLAN_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "SafeTaskPlan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "version": {"type": "integer", "const": DSL_VERSION},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_TASK_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": sorted(TASK_ACTIONS)},
                            "target": {
                                "type": ["object", "null"],
                                "properties": {
                                    "kind": {"type": "string", "enum": sorted(TARGET_KINDS)},
                                    "object": {"type": ["string", "null"]},
                                    "relation": {"type": "string", "enum": sorted(RELATIONS)},
                                    "vector_m": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 3,
                                        "maxItems": 3,
                                    },
                                },
                                "required": ["kind", "object", "relation", "vector_m"],
                                "additionalProperties": False,
                            },
                            "avoid": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                        },
                        "required": ["action", "target", "avoid"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["version", "steps"],
            "additionalProperties": False,
        },
    },
}


OPTIMIZATION_SPEC_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "SafeOptimizationSpec",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "version": {"type": "integer", "const": DSL_VERSION},
                "objective": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "squared_position_error"},
                        "target": TASK_PLAN_SCHEMA["json_schema"]["schema"]["properties"]["steps"]["items"]["properties"]["target"],
                        "q_weight": {
                            "type": "number", "minimum": Q_WEIGHT_MIN,
                            "maximum": Q_WEIGHT_MAX,
                        },
                        "linear_delta_u_weight": {
                            "type": "number", "enum": list(DELTA_U_LEVELS),
                        },
                    },
                    "required": [
                        "kind", "target", "q_weight", "linear_delta_u_weight"
                    ],
                    "additionalProperties": False,
                },
                "safety": {
                    "type": "object",
                    "properties": {
                        "gamma": {"type": "number", "enum": list(GAMMA_LEVELS)},
                    },
                    "required": ["gamma"],
                    "additionalProperties": False,
                },
                "limits": {
                    "type": "object",
                    "properties": {
                        "workspace_lower_m": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": 3, "maxItems": 3,
                        },
                        "workspace_upper_m": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": 3, "maxItems": 3,
                        },
                        "linear_speed_limit_mps": {
                            "type": "number", "minimum": LINEAR_SPEED_MIN_MPS,
                            "maximum": LINEAR_SPEED_MAX_MPS,
                        },
                    },
                    "required": [
                        "workspace_lower_m", "workspace_upper_m",
                        "linear_speed_limit_mps"
                    ],
                    "additionalProperties": False,
                },
                "constraints": {
                    "type": "array",
                    "maxItems": MAX_CONSTRAINTS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": sorted(CONSTRAINT_KINDS)},
                            "object": {"type": ["string", "null"]},
                            "clearance_m": {
                                "type": "number", "minimum": 0.0,
                                "maximum": CLEARANCE_MAX_M,
                            },
                            "value_m": {"type": ["number", "null"]},
                        },
                        "required": ["kind", "object", "clearance_m", "value_m"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["version", "objective", "safety", "limits", "constraints"],
            "additionalProperties": False,
        },
    },
}


TP_SYSTEM_PROMPT = """You are the NARRATE Task Planner for a robot gripper.
Return only SafeTaskPlan JSON matching the supplied schema. Use only scene
object names. Every move must explicitly list avoided objects. Never emit
Python, CasADi, prose, function calls, or fields outside the schema.

For each single pick-and-place operation, emit exactly this action grammar:
1. move to the source object with relation=center;
2. close_gripper with target=null and avoid=[];
3. move to the destination object with relation=above or relation=center;
4. open_gripper with target=null and avoid=[].

The target field of every open_gripper or close_gripper action MUST be the JSON
literal null, and its avoid field MUST be an empty JSON array. Do not attach an
object name, position, hazard, or explanation to a gripper action. Repeat the
same four-step grammar for additional pick-and-place operations."""

OD_SYSTEM_PROMPT = """You are the NARRATE Optimization Designer. Convert one
validated move step to SafeOptimizationSpec JSON. Select only bounded MPC
parameters and constraints from the supplied schema. The only objective is
squared_position_error. Every avoided object requires a collision_clearance
constraint. The resolved target must remain inside the selected workspace.
Copy the move_step target object exactly into objective.target; do not change
its kind, object, relation, or vector_m. Return every required root field:
version, objective, safety, limits, and constraints. For collision_clearance,
use object=<avoided object>, clearance_m within the schema bounds, and
value_m=null. The chosen workspace must contain the resolved target and remain
inside the hard Safe Panda envelope included in the schema and scene context.
Never emit mathematical source code, Python, CasADi, prose, or extra fields."""


def _exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ValueError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _enum_number(value: Any, allowed: Sequence[float], label: str) -> float:
    converted = _finite_number(value, label)
    if converted not in allowed:
        raise ValueError(f"{label} must be one of {tuple(allowed)}")
    return converted


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be a three-element JSON array")
    return tuple(_finite_number(item, label) for item in value)  # type: ignore[return-value]


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{label} must be a non-empty string up to {MAX_NAME_LENGTH} chars")
    if not all(character.isalnum() or character in {"_", "-"} for character in value):
        raise ValueError(f"{label} contains forbidden characters")
    return value


def _decode_json(payload: str | bytes | Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _object(value, label)


@dataclass(frozen=True, slots=True)
class SceneObject:
    name: str
    position: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        _name(self.name, "scene object name")
        if len(self.position) != 3 or any(not isfinite(v) for v in self.position):
            raise ValueError("scene object position must be a finite 3-vector")
        if not isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("scene object radius must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TargetSpec:
    kind: str
    object_name: str | None
    relation: str
    vector_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TaskStep:
    action: str
    target: TargetSpec | None
    avoid: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskPlan:
    version: int
    steps: tuple[TaskStep, ...]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    target: TargetSpec
    q_weight: float
    linear_delta_u_weight: float


@dataclass(frozen=True, slots=True)
class SafetySpec:
    gamma: float


@dataclass(frozen=True, slots=True)
class LimitsSpec:
    workspace_lower_m: tuple[float, float, float]
    workspace_upper_m: tuple[float, float, float]
    linear_speed_limit_mps: float


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    kind: str
    object_name: str | None
    clearance_m: float
    value_m: float | None


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    version: int
    objective: ObjectiveSpec
    safety: SafetySpec
    limits: LimitsSpec
    constraints: tuple[ConstraintSpec, ...]


def _parse_target(value: Any, known_objects: set[str], label: str) -> TargetSpec:
    target = _object(value, label)
    _exact_keys(target, {"kind", "object", "relation", "vector_m"}, label)
    kind = target["kind"]
    relation = target["relation"]
    if kind not in TARGET_KINDS or relation not in RELATIONS:
        raise ValueError(f"{label} uses a non-whitelisted kind or relation")
    object_name = None if target["object"] is None else _name(target["object"], f"{label}.object")
    vector = _vector3(target["vector_m"], f"{label}.vector_m")
    if kind == "object":
        if object_name not in known_objects:
            raise ValueError(f"{label} references unknown object {object_name!r}")
    elif object_name is not None or relation != "center":
        raise ValueError(f"{label} non-object targets require object=null and relation=center")
    if any(abs(component) > 3.0 for component in vector):
        raise ValueError(f"{label}.vector_m exceeds the 3 m DSL bound")
    return TargetSpec(kind, object_name, relation, vector)


def parse_task_plan(
    payload: str | bytes | Mapping[str, Any], scene_objects: Sequence[SceneObject]
) -> TaskPlan:
    """Parse and semantically validate an untrusted Task Planner response."""

    data = _decode_json(payload, "task plan")
    _exact_keys(data, {"version", "steps"}, "task plan")
    if data["version"] != DSL_VERSION:
        raise ValueError("unsupported task-plan DSL version")
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_TASK_STEPS:
        raise ValueError(f"steps must contain 1..{MAX_TASK_STEPS} entries")
    known = {item.name for item in scene_objects}
    if len(known) != len(scene_objects):
        raise ValueError("scene object names must be unique")
    gripper_open = True
    held: str | None = None
    last_move_target: str | None = None
    parsed: list[TaskStep] = []
    for index, raw in enumerate(raw_steps):
        step = _object(raw, f"steps[{index}]")
        _exact_keys(step, {"action", "target", "avoid"}, f"steps[{index}]")
        action = step["action"]
        if action not in TASK_ACTIONS:
            raise ValueError(f"steps[{index}] has an unknown action")
        avoid_raw = step["avoid"]
        if not isinstance(avoid_raw, list):
            raise ValueError(f"steps[{index}].avoid must be an array")
        avoid = tuple(_name(item, f"steps[{index}].avoid") for item in avoid_raw)
        if len(set(avoid)) != len(avoid) or any(item not in known for item in avoid):
            raise ValueError(f"steps[{index}].avoid must contain unique known objects")
        target = None
        if action == "move":
            if step["target"] is None:
                raise ValueError(f"steps[{index}] move requires a target")
            target = _parse_target(step["target"], known, f"steps[{index}].target")
            if held is not None and held in avoid:
                raise ValueError(f"steps[{index}] cannot avoid the held object")
            last_move_target = target.object_name if target.kind == "object" and target.relation == "center" else None
        else:
            if step["target"] is not None or avoid:
                raise ValueError(f"steps[{index}] gripper action requires target=null and avoid=[]")
            if action == "close_gripper":
                if not gripper_open or held is not None or last_move_target is None:
                    raise ValueError("close_gripper requires an open gripper at an object center")
                held = last_move_target
                gripper_open = False
            else:
                if gripper_open or held is None:
                    raise ValueError("open_gripper requires a held object")
                held = None
                gripper_open = True
            last_move_target = None
        parsed.append(TaskStep(action, target, avoid))
    return TaskPlan(DSL_VERSION, tuple(parsed))


def parse_optimization_spec(
    payload: str | bytes | Mapping[str, Any],
    scene_objects: Sequence[SceneObject],
    *,
    current_position: Sequence[float],
    required_avoid: Sequence[str] = (),
) -> OptimizationSpec:
    """Parse an untrusted Optimization Designer response without source eval."""

    data = _decode_json(payload, "optimization spec")
    _exact_keys(
        data,
        {"version", "objective", "safety", "limits", "constraints"},
        "optimization spec",
    )
    if data["version"] != DSL_VERSION:
        raise ValueError("unsupported optimization-spec DSL version")
    known = {item.name for item in scene_objects}
    objective = _object(data["objective"], "objective")
    _exact_keys(
        objective,
        {"kind", "target", "q_weight", "linear_delta_u_weight"},
        "objective",
    )
    if objective["kind"] != "squared_position_error":
        raise ValueError("objective kind is not whitelisted")
    q_weight = _finite_number(objective["q_weight"], "objective.q_weight")
    if not Q_WEIGHT_MIN <= q_weight <= Q_WEIGHT_MAX:
        raise ValueError(
            f"objective.q_weight must be in [{Q_WEIGHT_MIN}, {Q_WEIGHT_MAX}]"
        )
    delta_u_weight = _enum_number(
        objective["linear_delta_u_weight"],
        DELTA_U_LEVELS,
        "objective.linear_delta_u_weight",
    )
    target = _parse_target(objective["target"], known, "objective.target")

    safety = _object(data["safety"], "safety")
    _exact_keys(safety, {"gamma"}, "safety")
    gamma = _enum_number(safety["gamma"], GAMMA_LEVELS, "safety.gamma")

    limits = _object(data["limits"], "limits")
    _exact_keys(
        limits,
        {"workspace_lower_m", "workspace_upper_m", "linear_speed_limit_mps"},
        "limits",
    )
    workspace_lower = _vector3(limits["workspace_lower_m"], "limits.workspace_lower_m")
    workspace_upper = _vector3(limits["workspace_upper_m"], "limits.workspace_upper_m")
    for axis, (lower, upper, hard_lower, hard_upper) in enumerate(
        zip(
            workspace_lower,
            workspace_upper,
            SAFE_PANDA_WORKSPACE_LOWER_M,
            SAFE_PANDA_WORKSPACE_UPPER_M,
        )
    ):
        if not hard_lower <= lower < upper <= hard_upper:
            raise ValueError(
                f"limits workspace axis {axis} must lie inside the Safe Panda envelope"
            )
    speed_limit = _finite_number(
        limits["linear_speed_limit_mps"], "limits.linear_speed_limit_mps"
    )
    if not LINEAR_SPEED_MIN_MPS <= speed_limit <= LINEAR_SPEED_MAX_MPS:
        raise ValueError(
            "limits.linear_speed_limit_mps must be in "
            f"[{LINEAR_SPEED_MIN_MPS}, {LINEAR_SPEED_MAX_MPS}]"
        )

    resolved_target = resolve_target(target, scene_objects, current_position)
    if any(
        coordinate < lower or coordinate > upper
        for coordinate, lower, upper in zip(
            resolved_target, workspace_lower, workspace_upper
        )
    ):
        raise ValueError("objective target lies outside the selected workspace")

    raw_constraints = data["constraints"]
    if not isinstance(raw_constraints, list) or len(raw_constraints) > MAX_CONSTRAINTS:
        raise ValueError(f"constraints must be an array of at most {MAX_CONSTRAINTS}")
    constraints: list[ConstraintSpec] = []
    for index, raw in enumerate(raw_constraints):
        label = f"constraints[{index}]"
        item = _object(raw, label)
        required = {"kind", "object", "clearance_m", "value_m"}
        _exact_keys(item, required, label)
        kind = item["kind"]
        if kind not in CONSTRAINT_KINDS:
            raise ValueError(f"{label}.kind is not whitelisted")
        object_name = None if item["object"] is None else _name(item["object"], f"{label}.object")
        clearance = _finite_number(item["clearance_m"], f"{label}.clearance_m")
        value_m = None if item["value_m"] is None else _finite_number(item["value_m"], f"{label}.value_m")
        if kind == "collision_clearance":
            if (
                object_name not in known
                or value_m is not None
                or not CLEARANCE_MIN_M <= clearance <= CLEARANCE_MAX_M
            ):
                raise ValueError(f"{label} collision constraint fields are inconsistent")
        else:
            if object_name is not None or value_m is None or clearance != 0.0:
                raise ValueError(f"{label} height constraint fields are inconsistent")
            z_value = float(value_m)
            if not workspace_lower[2] <= z_value <= workspace_upper[2]:
                raise ValueError(f"{label}.value_m lies outside the selected workspace")
        constraints.append(ConstraintSpec(kind, object_name, clearance, value_m))

    collision_objects = {
        constraint.object_name
        for constraint in constraints
        if constraint.kind == "collision_clearance"
    }
    if target.object_name in collision_objects:
        raise ValueError("objective target cannot also be a collision obstacle")
    if not set(required_avoid).issubset(collision_objects):
        raise ValueError("optimization spec omits a required collision obstacle")
    return OptimizationSpec(
        DSL_VERSION,
        ObjectiveSpec(target, q_weight, delta_u_weight),
        SafetySpec(gamma),
        LimitsSpec(workspace_lower, workspace_upper, speed_limit),
        tuple(constraints),
    )


def optimization_from_task_step(
    step: TaskStep,
    scene_objects: Sequence[SceneObject],
    *,
    objective_weight: float = 1.0,
    robot_radius: float = 0.012,
    gamma: float = 0.05,
) -> OptimizationSpec:
    """Deterministic fail-closed OD fallback for a validated move step."""

    if step.action != "move" or step.target is None:
        raise ValueError("only move steps have an optimization formulation")
    objects = {item.name: item for item in scene_objects}
    constraints = tuple(
        ConstraintSpec("collision_clearance", name, robot_radius, None)
        for name in step.avoid
        if name in objects
    )
    return OptimizationSpec(
        DSL_VERSION,
        ObjectiveSpec(step.target, objective_weight, 0.5),
        SafetySpec(_enum_number(gamma, GAMMA_LEVELS, "gamma")),
        LimitsSpec(
            SAFE_PANDA_WORKSPACE_LOWER_M,
            SAFE_PANDA_WORKSPACE_UPPER_M,
            LINEAR_SPEED_MAX_MPS,
        ),
        constraints,
    )


def resolve_target(
    target: TargetSpec,
    scene_objects: Sequence[SceneObject],
    current_position: Sequence[float],
) -> tuple[float, float, float]:
    """Resolve a validated target using SI units and fixed spatial semantics."""

    current = _vector3(list(current_position), "current_position")
    if target.kind == "absolute":
        return target.vector_m
    if target.kind == "current_offset":
        return tuple(a + b for a, b in zip(current, target.vector_m))  # type: ignore[return-value]
    objects = {item.name: item for item in scene_objects}
    obj = objects[target.object_name]  # guaranteed by parser
    relation_axes = {
        "center": (0.0, 0.0, 0.0),
        "above": (0.0, 0.0, obj.radius),
        "front": (obj.radius, 0.0, 0.0),
        "behind": (-obj.radius, 0.0, 0.0),
        "left": (0.0, obj.radius, 0.0),
        "right": (0.0, -obj.radius, 0.0),
    }
    return tuple(
        base + relation + offset
        for base, relation, offset in zip(
            obj.position, relation_axes[target.relation], target.vector_m
        )
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CompiledOptimization:
    objective: Any
    inequalities: tuple[Any, ...]
    target_position: tuple[float, float, float]


def controller_config_from_optimization(
    spec: OptimizationSpec,
    scene_objects: Sequence[SceneObject],
    current_position: Sequence[float],
) -> Any:
    """Translate a validated DSL spec to the fixed paper MPC configuration.

    Dynamics, horizon, yaw weights, regularizers, jerk, and optimal decay stay
    at :class:`PaperMPCConfig` defaults.  Only the A1-approved bounded fields
    are transferred from the untrusted-language trust boundary.
    """

    from .controller import PaperMPCConfig

    target = resolve_target(spec.objective.target, scene_objects, current_position)
    return PaperMPCConfig(
        target=(*target, 0.0, 0.0, 0.0, 0.0, 0.0),
        q_weight=spec.objective.q_weight,
        linear_delta_u_weight=spec.objective.linear_delta_u_weight,
        position_lower=spec.limits.workspace_lower_m,
        position_upper=spec.limits.workspace_upper_m,
        linear_input_limit=spec.limits.linear_speed_limit_mps,
    )


def compile_optimization(
    spec: OptimizationSpec,
    scene_objects: Sequence[SceneObject],
    current_position: Sequence[float],
    x: Any,
    casadi: Any,
) -> CompiledOptimization:
    """Compile a validated spec through a whitelist of CasADi constructors."""

    target = resolve_target(spec.objective.target, scene_objects, current_position)
    target_dm = casadi.DM(target)
    objective = spec.objective.q_weight * casadi.sumsqr(x[:3] - target_dm)
    objects = {item.name: item for item in scene_objects}
    inequalities: list[Any] = []
    for constraint in spec.constraints:
        if constraint.kind == "collision_clearance":
            obj = objects[constraint.object_name]
            radius = obj.radius + constraint.clearance_m
            inequalities.append(radius**2 - casadi.sumsqr(x[:3] - casadi.DM(obj.position)))
        elif constraint.kind == "minimum_height":
            inequalities.append(constraint.value_m - x[2])
        elif constraint.kind == "maximum_height":
            inequalities.append(x[2] - constraint.value_m)
        else:  # impossible for parsed dataclasses; protects manual construction
            raise ValueError(f"unsupported constraint kind: {constraint.kind}")
    return CompiledOptimization(objective, tuple(inequalities), target)


@dataclass(frozen=True, slots=True)
class SafeNarrateConfig:
    """External inference settings; credentials remain in a local file."""

    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    provider: str = "deepinfra"
    token_path: str = "hftoken.txt"
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    seed: int = 17
    response_format_mode: str = "json_object"

    def __post_init__(self) -> None:
        if not self.model or not self.provider:
            raise ValueError("model and provider must be non-empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.response_format_mode not in {"json_object", "json_schema"}:
            raise ValueError("response_format_mode must be json_object or json_schema")


@dataclass(frozen=True, slots=True)
class SafeNarrateResult:
    task_plan: TaskPlan
    optimization_specs: tuple[OptimizationSpec | None, ...]
    tp_latency_seconds: float
    od_latency_seconds: float
    od_fallbacks: int
    model: str
    provider: str
    instruction_hash: str
    od_attempts: tuple["ODAttemptAudit", ...] = ()


@dataclass(frozen=True, slots=True)
class ODAttemptAudit:
    task_step_index: int
    status: str
    latency_seconds: float
    raw_response: str | None
    cause_type: str | None
    cause_message: str | None

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "fallback"}:
            raise ValueError("OD attempt status must be accepted or fallback")
        if self.latency_seconds < 0.0:
            raise ValueError("OD attempt latency must be non-negative")
        if self.status == "accepted" and (
            self.cause_type is not None or self.cause_message is not None
        ):
            raise ValueError("accepted OD attempts cannot contain an error")
        if self.status == "fallback" and self.cause_type is None:
            raise ValueError("fallback OD attempts require an error type")


class LanguageDSLInferenceError(RuntimeError):
    """Raised when external inference cannot cross a DSL trust boundary."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        cause_type: str,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause_type = cause_type
        self.raw_response = raw_response


class HuggingFaceSafeNarratePlanner:
    """Two-stage TP/OD inference with validation at both trust boundaries.

    A rejected OD response uses :func:`optimization_from_task_step`, which is a
    deterministic whitelist compiler input.  A rejected TP response has no
    safe task-level fallback and therefore stops before motion.
    """

    def __init__(
        self,
        config: SafeNarrateConfig | None = None,
        *,
        client_factory: Callable[[SafeNarrateConfig, str], Any] | None = None,
    ) -> None:
        self.config = config or SafeNarrateConfig()
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(config: SafeNarrateConfig, token: str) -> Any:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise LanguageDSLInferenceError(
                "Install the LLM extra with: python -m pip install -e '.[llm]'",
                stage="client_setup",
                cause_type=type(exc).__name__,
            ) from exc
        return InferenceClient(
            provider=config.provider,
            api_key=token,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    def _content(response: Any) -> str:
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("inference response content is empty")
        return content

    def _completion(
        self, client: Any, system_prompt: str, user_payload: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> str:
        response_format: Mapping[str, Any] = schema
        wire_payload = dict(user_payload)
        if self.config.response_format_mode == "json_object":
            # DeepInfra's xgrammar backend rejects several standard schema
            # features used by this DSL. JSON-object mode constrains syntax;
            # the same strict local parser remains the semantic trust boundary.
            response_format = {"type": "json_object"}
            wire_payload["output_schema"] = schema["json_schema"]["schema"]
        response = client.chat_completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(wire_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            response_format=response_format,
            max_tokens=2048,
            temperature=self.config.temperature,
            seed=self.config.seed,
        )
        return self._content(response)

    def formulate(
        self,
        instruction: str,
        scene_objects: Sequence[SceneObject],
        *,
        current_position: Sequence[float],
        required_hazards: Sequence[str] = (),
    ) -> SafeNarrateResult:
        """Run TP once and OD once per move; never return unvalidated output."""

        if not instruction.strip():
            raise ValueError("instruction must be non-empty")
        known_objects = {item.name for item in scene_objects}
        hazards = tuple(required_hazards)
        if len(set(hazards)) != len(hazards) or any(
            hazard not in known_objects for hazard in hazards
        ):
            raise ValueError("required_hazards must contain unique known scene objects")
        # Import here so the DSL itself remains dependency-free and no token is
        # placed in environment variables or logs.
        from .hf_llm import load_hf_token

        tp_raw: str | None = None
        try:
            client = self._client_factory(
                self.config, load_hf_token(self.config.token_path)
            )
            scene_payload = [
                {"name": item.name, "position_m": item.position, "radius_m": item.radius}
                for item in scene_objects
            ]
            tp_started = perf_counter()
            tp_raw = self._completion(
                client,
                TP_SYSTEM_PROMPT,
                {"instruction": instruction.strip(), "scene_objects": scene_payload},
                TASK_PLAN_SCHEMA,
            )
            plan = parse_task_plan(tp_raw, scene_objects)
            for index, step in enumerate(plan.steps):
                if step.action != "move" or step.target is None:
                    continue
                if step.target.object_name in hazards:
                    raise ValueError("a required hazard cannot be a move target")
                if not set(hazards).issubset(step.avoid):
                    raise ValueError(
                        f"task-plan step {index} omits a required hazard"
                    )
                resolved = resolve_target(step.target, scene_objects, current_position)
                if any(
                    value < lower or value > upper
                    for value, lower, upper in zip(
                        resolved,
                        SAFE_PANDA_WORKSPACE_LOWER_M,
                        SAFE_PANDA_WORKSPACE_UPPER_M,
                    )
                ):
                    raise ValueError(
                        f"task-plan target at step {index} lies outside the Safe Panda workspace"
                    )
            tp_latency = perf_counter() - tp_started
        except Exception as error:
            raise LanguageDSLInferenceError(
                f"safe Task Planner inference failed closed: {type(error).__name__}",
                stage="task_planner",
                cause_type=type(error).__name__,
                raw_response=tp_raw,
            ) from error

        specs: list[OptimizationSpec | None] = []
        od_attempts: list[ODAttemptAudit] = []
        od_latency = 0.0
        fallbacks = 0
        for step_index, step in enumerate(plan.steps):
            if step.action != "move":
                specs.append(None)
                continue
            od_payload = {
                "move_step": {
                    "action": step.action,
                    "target": {
                        "kind": step.target.kind,
                        "object": step.target.object_name,
                        "relation": step.target.relation,
                        "vector_m": step.target.vector_m,
                    },
                    "avoid": step.avoid,
                },
                "current_position_m": tuple(float(value) for value in current_position),
                "scene_objects": scene_payload,
            }
            started = perf_counter()
            raw: str | None = None
            try:
                raw = self._completion(
                    client, OD_SYSTEM_PROMPT, od_payload, OPTIMIZATION_SPEC_SCHEMA
                )
                spec = parse_optimization_spec(
                    raw,
                    scene_objects,
                    current_position=current_position,
                    required_avoid=step.avoid,
                )
                elapsed = perf_counter() - started
                od_latency += elapsed
                specs.append(spec)
                od_attempts.append(
                    ODAttemptAudit(
                        step_index,
                        "accepted",
                        elapsed,
                        raw,
                        None,
                        None,
                    )
                )
            except Exception as error:
                elapsed = perf_counter() - started
                od_latency += elapsed
                fallbacks += 1
                specs.append(optimization_from_task_step(step, scene_objects))
                od_attempts.append(
                    ODAttemptAudit(
                        step_index,
                        "fallback",
                        elapsed,
                        raw,
                        type(error).__name__,
                        str(error),
                    )
                )
        return SafeNarrateResult(
            task_plan=plan,
            optimization_specs=tuple(specs),
            tp_latency_seconds=tp_latency,
            od_latency_seconds=od_latency,
            od_fallbacks=fallbacks,
            model=self.config.model,
            provider=self.config.provider,
            instruction_hash=sha256(instruction.strip().encode("utf-8")).hexdigest(),
            od_attempts=tuple(od_attempts),
        )
