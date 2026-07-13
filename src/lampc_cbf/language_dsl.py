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


DSL_VERSION = 1
MAX_TASK_STEPS = 24
MAX_CONSTRAINTS = 24
MAX_JSON_BYTES = 32_768
MAX_NAME_LENGTH = 64

TASK_ACTIONS = frozenset({"move", "open_gripper", "close_gripper"})
TARGET_KINDS = frozenset({"object", "absolute", "current_offset"})
RELATIONS = frozenset({"center", "above", "front", "behind", "left", "right"})
CONSTRAINT_KINDS = frozenset(
    {"collision_clearance", "minimum_height", "maximum_height", "workspace_box"}
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
                        "weight": {"type": "number", "minimum": 0.01, "maximum": 100.0},
                    },
                    "required": ["kind", "target", "weight"],
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
                            "clearance_m": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                            "value_m": {"type": ["number", "null"]},
                            "lower_m": {
                                "type": ["array", "null"], "items": {"type": "number"},
                                "minItems": 3, "maxItems": 3,
                            },
                            "upper_m": {
                                "type": ["array", "null"], "items": {"type": "number"},
                                "minItems": 3, "maxItems": 3,
                            },
                        },
                        "required": [
                            "kind", "object", "clearance_m", "value_m", "lower_m", "upper_m"
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["version", "objective", "constraints"],
            "additionalProperties": False,
        },
    },
}


TP_SYSTEM_PROMPT = """You are the NARRATE Task Planner for a robot gripper.
Return only SafeTaskPlan JSON matching the supplied schema. Use only scene
object names. Every move must explicitly list avoided objects. Never emit
Python, CasADi, prose, function calls, or fields outside the schema."""

OD_SYSTEM_PROMPT = """You are the NARRATE Optimization Designer. Convert one
validated move step to SafeOptimizationSpec JSON. The only objective is
squared_position_error and constraints must come from the supplied enum.
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
    weight: float


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    kind: str
    object_name: str | None
    clearance_m: float
    value_m: float | None
    lower_m: tuple[float, float, float] | None
    upper_m: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    version: int
    objective: ObjectiveSpec
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
    payload: str | bytes | Mapping[str, Any], scene_objects: Sequence[SceneObject]
) -> OptimizationSpec:
    """Parse an untrusted Optimization Designer response without source eval."""

    data = _decode_json(payload, "optimization spec")
    _exact_keys(data, {"version", "objective", "constraints"}, "optimization spec")
    if data["version"] != DSL_VERSION:
        raise ValueError("unsupported optimization-spec DSL version")
    known = {item.name for item in scene_objects}
    objective = _object(data["objective"], "objective")
    _exact_keys(objective, {"kind", "target", "weight"}, "objective")
    if objective["kind"] != "squared_position_error":
        raise ValueError("objective kind is not whitelisted")
    weight = _finite_number(objective["weight"], "objective.weight")
    if not 0.01 <= weight <= 100.0:
        raise ValueError("objective.weight must be in [0.01, 100]")
    target = _parse_target(objective["target"], known, "objective.target")
    raw_constraints = data["constraints"]
    if not isinstance(raw_constraints, list) or len(raw_constraints) > MAX_CONSTRAINTS:
        raise ValueError(f"constraints must be an array of at most {MAX_CONSTRAINTS}")
    constraints: list[ConstraintSpec] = []
    for index, raw in enumerate(raw_constraints):
        label = f"constraints[{index}]"
        item = _object(raw, label)
        required = {"kind", "object", "clearance_m", "value_m", "lower_m", "upper_m"}
        _exact_keys(item, required, label)
        kind = item["kind"]
        if kind not in CONSTRAINT_KINDS:
            raise ValueError(f"{label}.kind is not whitelisted")
        object_name = None if item["object"] is None else _name(item["object"], f"{label}.object")
        clearance = _finite_number(item["clearance_m"], f"{label}.clearance_m")
        value_m = None if item["value_m"] is None else _finite_number(item["value_m"], f"{label}.value_m")
        lower = None if item["lower_m"] is None else _vector3(item["lower_m"], f"{label}.lower_m")
        upper = None if item["upper_m"] is None else _vector3(item["upper_m"], f"{label}.upper_m")
        if not 0.0 <= clearance <= 0.5:
            raise ValueError(f"{label}.clearance_m must be in [0, 0.5]")
        if kind == "collision_clearance":
            if object_name not in known or value_m is not None or lower is not None or upper is not None:
                raise ValueError(f"{label} collision constraint fields are inconsistent")
        elif kind in {"minimum_height", "maximum_height"}:
            if object_name is not None or value_m is None or lower is not None or upper is not None or clearance != 0.0:
                raise ValueError(f"{label} height constraint fields are inconsistent")
        else:
            if object_name is not None or value_m is not None or lower is None or upper is None or clearance != 0.0:
                raise ValueError(f"{label} workspace constraint fields are inconsistent")
            if any(lo >= hi for lo, hi in zip(lower, upper)):
                raise ValueError(f"{label} workspace lower bounds must be below upper bounds")
        constraints.append(ConstraintSpec(kind, object_name, clearance, value_m, lower, upper))
    return OptimizationSpec(DSL_VERSION, ObjectiveSpec(target, weight), tuple(constraints))


def optimization_from_task_step(
    step: TaskStep,
    scene_objects: Sequence[SceneObject],
    *,
    objective_weight: float = 1.0,
    robot_radius: float = 0.012,
) -> OptimizationSpec:
    """Deterministic fail-closed OD fallback for a validated move step."""

    if step.action != "move" or step.target is None:
        raise ValueError("only move steps have an optimization formulation")
    objects = {item.name: item for item in scene_objects}
    constraints = tuple(
        ConstraintSpec(
            "collision_clearance", name, robot_radius, None, None, None
        )
        for name in step.avoid
        if name in objects
    )
    return OptimizationSpec(
        DSL_VERSION, ObjectiveSpec(step.target, objective_weight), constraints
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
    objective = spec.objective.weight * casadi.sumsqr(x[:3] - target_dm)
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
        elif constraint.kind == "workspace_box":
            for axis in range(3):
                inequalities.append(constraint.lower_m[axis] - x[axis])
                inequalities.append(x[axis] - constraint.upper_m[axis])
        else:  # impossible for parsed dataclasses; protects manual construction
            raise ValueError(f"unsupported constraint kind: {constraint.kind}")
    return CompiledOptimization(objective, tuple(inequalities), target)


@dataclass(frozen=True, slots=True)
class SafeNarrateConfig:
    """External inference settings; credentials remain in a local file."""

    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    provider: str = "deepinfra"
    token_path: str = "hftoken.txt"
    timeout_seconds: float = 8.0
    temperature: float = 0.0
    seed: int = 17

    def __post_init__(self) -> None:
        if not self.model or not self.provider:
            raise ValueError("model and provider must be non-empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")


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


class LanguageDSLInferenceError(RuntimeError):
    """Raised when external TP inference cannot produce a safe plan."""


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
                "Install the LLM extra with: python -m pip install -e '.[llm]'"
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
        response = client.chat_completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            response_format=schema,
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
    ) -> SafeNarrateResult:
        """Run TP once and OD once per move; never return unvalidated output."""

        if not instruction.strip():
            raise ValueError("instruction must be non-empty")
        # Import here so the DSL itself remains dependency-free and no token is
        # placed in environment variables or logs.
        from .hf_llm import load_hf_token

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
            tp_latency = perf_counter() - tp_started
        except Exception as error:
            raise LanguageDSLInferenceError(
                f"safe Task Planner inference failed closed: {type(error).__name__}"
            ) from error

        specs: list[OptimizationSpec | None] = []
        od_latency = 0.0
        fallbacks = 0
        for step in plan.steps:
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
            try:
                started = perf_counter()
                raw = self._completion(
                    client, OD_SYSTEM_PROMPT, od_payload, OPTIMIZATION_SPEC_SCHEMA
                )
                od_latency += perf_counter() - started
                specs.append(parse_optimization_spec(raw, scene_objects))
            except Exception:
                fallbacks += 1
                specs.append(optimization_from_task_step(step, scene_objects))
        return SafeNarrateResult(
            task_plan=plan,
            optimization_specs=tuple(specs),
            tp_latency_seconds=tp_latency,
            od_latency_seconds=od_latency,
            od_fallbacks=fallbacks,
            model=self.config.model,
            provider=self.config.provider,
            instruction_hash=sha256(instruction.strip().encode("utf-8")).hexdigest(),
        )
