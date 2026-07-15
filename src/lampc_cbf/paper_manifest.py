"""Load and validate the immutable paper-fidelity experiment manifest."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from typing import Any, Mapping


PAPER_DOI = "10.1109/ACCESS.2026.3664145"
PAPER_REPLICATION_METHODS = (
    "fixed_cbf_static_g015",
    "paper_async_feedback_static",
)
PAPER_FIDELITY_PROFILE = "paper_fidelity"
MODEL_SUBSTITUTION_PROFILE = "paper_fidelity_model_substitution"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def _require_close(actual: object, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} must be {expected}, got {value}")


@dataclass(frozen=True, slots=True)
class PaperFidelityManifest:
    """Validated values used by the dedicated 50-episode paper stage."""

    path: str
    manifest_hash: str
    payload: Mapping[str, Any]
    profile: str
    model_substitution: bool
    stage: str
    episodes: int
    seed: int
    bootstrap_resamples: int
    max_steps: int
    speed_lower: float
    speed_upper: float
    fixed_lateral_offset: float
    intervention_time_lower: float
    intervention_time_upper: float
    feedback_schedule_mode: str
    feedback_request_policy: str
    feedback_requests_per_episode: int
    latency_trace_mode: str
    reference_speed: float
    method_names: tuple[str, ...]
    output_dir: str
    sensor_period: float
    measurement_noise_sigma: float
    cbf_transition_mode: str
    goal_offset: tuple[float, float, float]
    obstacle_start_forward_offset: float
    obstacle_start_vertical_offset: float
    obstacle_radius: float
    collision_radius: float
    gamma_update_ttl: float
    solver_max_constraint_violation: float
    solver_max_cpu_time: float
    control_deadline: float
    reject_deadline_miss: bool
    initial_query: str
    feedback_query: str
    required_model_family: str
    required_provider: str
    llm_timeout_seconds: float
    llm_max_tokens: int
    llm_guided_json_enabled: bool
    llm_enable_thinking: bool

    @classmethod
    def load(cls, path: str | Path) -> "PaperFidelityManifest":
        source_path = Path(path)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        root = dict(_mapping(payload, "manifest"))
        if "base_manifest" in root:
            base_name = root.get("base_manifest")
            if not isinstance(base_name, str) or not base_name:
                raise ValueError("base_manifest must be a non-empty string")
            base_path = source_path.parent / base_name
            base = dict(_mapping(
                json.loads(base_path.read_text(encoding="utf-8")),
                "base manifest",
            ))
            if base.get("profile") != PAPER_FIDELITY_PROFILE:
                raise ValueError("substitution base must use the paper_fidelity profile")
            language_override = _mapping(root.get("language"), "language override")
            benchmark_override = _mapping(root.get("benchmark"), "benchmark override")
            provenance_override = _mapping(
                root.get("provenance", {}), "provenance override"
            )
            root = base
            root["profile"] = payload.get("profile")
            root["language"] = dict(language_override)
            root["benchmark"] = dict(_mapping(base["benchmark"], "benchmark"))
            for key in (
                "output_dir",
                "feedback_requests_per_episode",
                "latency_trace_mode",
            ):
                if key in benchmark_override:
                    root["benchmark"][key] = benchmark_override[key]
            root["provenance"] = dict(_mapping(base["provenance"], "provenance"))
            root["provenance"]["model_substitution"] = provenance_override.get(
                "model_substitution"
            )
            root["resolved_from"] = base_name
        canonical = json.dumps(
            root, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

        _require_equal(root.get("schema_version"), 1, "schema_version")
        profile = root.get("profile")
        if profile not in {PAPER_FIDELITY_PROFILE, MODEL_SUBSTITUTION_PROFILE}:
            raise ValueError("unsupported paper manifest profile")
        model_substitution = profile == MODEL_SUBSTITUTION_PROFILE
        source = _mapping(root.get("source"), "source")
        _require_equal(source.get("doi"), PAPER_DOI, "source.doi")

        controller = _mapping(root.get("controller"), "controller")
        reported_controller = {
            "dt_seconds": 0.04,
            "horizon": 15,
            "state_order": ["x", "y", "z", "psi", "dx", "dy", "dz", "dpsi"],
            "input_order": ["u_x", "u_y", "u_z", "u_psi"],
            "state_weight": 1.0,
            "linear_delta_u_weight": 0.5,
            "yaw_delta_u_weight": 1e-5,
            "velocity_regularization_weight": 0.1,
            "yaw_regularization_weight": 5e-5,
            "position_lower_m": [-3.0, -3.0, 0.0],
            "yaw_bound_pi_multiplier": 0.55,
            "linear_input_limit": 0.2,
            "yaw_input_limit_pi_multiplier": 1.0,
            "prediction_mode": "static",
            "reference_mode": "direct_target",
            "cbf_transition_mode": "paper_state",
            "safety_reflex_enabled": False,
            "optimal_decay_weight": 0.0,
        }
        for key, expected in reported_controller.items():
            actual = controller.get(key)
            if isinstance(expected, float):
                _require_close(actual, expected, f"controller.{key}")
            else:
                _require_equal(actual, expected, f"controller.{key}")
        _require_close(controller.get("initial_gamma"), 0.15, "controller.initial_gamma")

        sensing = _mapping(root.get("sensing"), "sensing")
        _require_close(sensing.get("period_seconds"), 0.67, "sensing.period_seconds")
        _require_equal(
            sensing.get("measurement_noise_mode"),
            "gaussian",
            "sensing.measurement_noise_mode",
        )
        _require_close(
            sensing.get("measurement_noise_sigma_m"),
            0.005,
            "sensing.measurement_noise_sigma_m",
        )
        _require_equal(
            sensing.get("obstacle_motion_in_horizon"),
            "static",
            "sensing.obstacle_motion_in_horizon",
        )
        scene = _mapping(root.get("scene"), "scene")
        goal_offset = tuple(float(value) for value in scene.get("goal_offset_m", ()))
        if len(goal_offset) != 3:
            raise ValueError("scene.goal_offset_m must contain three values")
        for key in (
            "obstacle_start_forward_offset_m",
            "obstacle_radius_m",
            "gripper_collision_radius_m",
        ):
            if float(scene.get(key, 0.0)) <= 0.0:
                raise ValueError(f"scene.{key} must be positive")
        solver_policy = _mapping(root.get("solver_policy"), "solver_policy")
        for key in (
            "maximum_constraint_violation",
            "maximum_cpu_time_seconds",
            "control_deadline_seconds",
        ):
            if float(solver_policy.get(key, 0.0)) <= 0.0:
                raise ValueError(f"solver_policy.{key} must be positive")
        _require_equal(
            solver_policy.get("reject_deadline_miss"),
            False,
            "solver_policy.reject_deadline_miss",
        )
        language = _mapping(root.get("language"), "language")
        _require_equal(
            language.get("initial_query"),
            "Move gripper to red cube.",
            "language.initial_query",
        )
        _require_equal(
            language.get("feedback_query"),
            "Watch out! I think it's going to crash soon.",
            "language.feedback_query",
        )
        _require_equal(
            language.get("feedback_request_policy"),
            "one_shot_per_feedback_episode",
            "language.feedback_request_policy",
        )
        if model_substitution:
            if language.get("substitution_for") != "gpt-4o":
                raise ValueError("model substitution must explicitly replace gpt-4o")
            if not isinstance(language.get("required_model_family"), str):
                raise ValueError("substitution model family must be a string")
            if not isinstance(language.get("required_provider"), str):
                raise ValueError("substitution provider must be a string")
            if float(language.get("timeout_seconds", 0.0)) <= 0.0:
                raise ValueError("substitution timeout must be positive")
            if not isinstance(language.get("max_tokens"), int) or int(
                language["max_tokens"]
            ) < 1:
                raise ValueError("substitution max_tokens must be positive")
            for key in ("guided_json_enabled", "enable_thinking"):
                if not isinstance(language.get(key), bool):
                    raise ValueError(f"language.{key} must be boolean")
        else:
            _require_equal(
                language.get("required_model_family"),
                "gpt-4o",
                "language.required_model_family",
            )
            _require_equal(
                language.get("required_provider"),
                "openai",
                "language.required_provider",
            )

        extensions = _mapping(root.get("extensions"), "extensions")
        enabled = sorted(name for name, value in extensions.items() if value is not False)
        if enabled:
            raise ValueError(
                "paper_fidelity forbids enabled extensions: " + ", ".join(enabled)
            )

        benchmark = _mapping(root.get("benchmark"), "benchmark")
        _require_equal(
            benchmark.get("stage"), "paper-replication", "benchmark.stage"
        )
        _require_equal(benchmark.get("episodes"), 50, "benchmark.episodes")
        _require_equal(
            benchmark.get("bootstrap_resamples"),
            10_000,
            "benchmark.bootstrap_resamples",
        )
        _require_close(
            benchmark.get("obstacle_speed_lower_mps"),
            0.025,
            "benchmark.obstacle_speed_lower_mps",
        )
        _require_close(
            benchmark.get("obstacle_speed_upper_mps"),
            0.20,
            "benchmark.obstacle_speed_upper_mps",
        )
        _require_equal(
            benchmark.get("feedback_schedule_mode"),
            "elapsed_time",
            "benchmark.feedback_schedule_mode",
        )
        _require_equal(
            benchmark.get("feedback_requests_per_episode"),
            1,
            "benchmark.feedback_requests_per_episode",
        )
        _require_equal(
            benchmark.get("latency_trace_mode"),
            "precollected_uncached_per_episode_replay",
            "benchmark.latency_trace_mode",
        )
        method_names = tuple(str(value) for value in benchmark.get("method_names", ()))
        _require_equal(
            method_names, PAPER_REPLICATION_METHODS, "benchmark.method_names"
        )
        if not isinstance(benchmark.get("seed"), int):
            raise ValueError("benchmark.seed must be an integer")
        if not isinstance(benchmark.get("max_steps"), int):
            raise ValueError("benchmark.max_steps must be an integer")
        if float(benchmark.get("gamma_update_ttl_seconds", 0.0)) <= 0.0:
            raise ValueError("benchmark.gamma_update_ttl_seconds must be positive")
        output_dir = benchmark.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("benchmark.output_dir must be a non-empty string")

        return cls(
            path=str(source_path.resolve()),
            manifest_hash=sha256(canonical).hexdigest(),
            payload=root,
            profile=str(profile),
            model_substitution=model_substitution,
            stage="paper-replication",
            episodes=50,
            seed=int(benchmark["seed"]),
            bootstrap_resamples=10_000,
            max_steps=int(benchmark["max_steps"]),
            speed_lower=float(benchmark["obstacle_speed_lower_mps"]),
            speed_upper=float(benchmark["obstacle_speed_upper_mps"]),
            fixed_lateral_offset=float(benchmark["fixed_lateral_offset_m"]),
            intervention_time_lower=float(benchmark["intervention_time_lower_s"]),
            intervention_time_upper=float(benchmark["intervention_time_upper_s"]),
            feedback_schedule_mode="elapsed_time",
            feedback_request_policy=str(language["feedback_request_policy"]),
            feedback_requests_per_episode=int(
                benchmark["feedback_requests_per_episode"]
            ),
            latency_trace_mode=str(benchmark["latency_trace_mode"]),
            reference_speed=float(benchmark["reference_speed_mps"]),
            method_names=method_names,
            output_dir=output_dir,
            sensor_period=float(sensing["period_seconds"]),
            measurement_noise_sigma=float(sensing["measurement_noise_sigma_m"]),
            cbf_transition_mode=str(controller["cbf_transition_mode"]),
            goal_offset=goal_offset,
            obstacle_start_forward_offset=float(
                scene["obstacle_start_forward_offset_m"]
            ),
            obstacle_start_vertical_offset=float(
                scene["obstacle_start_vertical_offset_m"]
            ),
            obstacle_radius=float(scene["obstacle_radius_m"]),
            collision_radius=float(scene["gripper_collision_radius_m"]),
            gamma_update_ttl=float(benchmark["gamma_update_ttl_seconds"]),
            solver_max_constraint_violation=float(
                solver_policy["maximum_constraint_violation"]
            ),
            solver_max_cpu_time=float(solver_policy["maximum_cpu_time_seconds"]),
            control_deadline=float(solver_policy["control_deadline_seconds"]),
            reject_deadline_miss=bool(solver_policy["reject_deadline_miss"]),
            initial_query=str(language["initial_query"]),
            feedback_query=str(language["feedback_query"]),
            required_model_family=str(language["required_model_family"]),
            required_provider=str(language["required_provider"]),
            llm_timeout_seconds=float(language.get("timeout_seconds", 0.0)),
            llm_max_tokens=int(language.get("max_tokens", 0)),
            llm_guided_json_enabled=bool(
                language.get("guided_json_enabled", False)
            ),
            llm_enable_thinking=bool(language.get("enable_thinking", False)),
        )

    def benchmark_kwargs(self) -> dict[str, Any]:
        """Return the manifest-controlled `PairedBenchmarkConfig` fields."""

        return {
            "stage": self.stage,
            "episodes": self.episodes,
            "seed": self.seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "max_steps": self.max_steps,
            "speed_lower": self.speed_lower,
            "speed_upper": self.speed_upper,
            "fixed_lateral_offset": self.fixed_lateral_offset,
            "intervention_time_lower": self.intervention_time_lower,
            "intervention_time_upper": self.intervention_time_upper,
            "feedback_schedule_mode": self.feedback_schedule_mode,
            "feedback_request_policy": self.feedback_request_policy,
            "feedback_requests_per_episode": self.feedback_requests_per_episode,
            "latency_trace_mode": self.latency_trace_mode,
            "reference_speed": self.reference_speed,
            "method_names": self.method_names,
            "sensor_period": self.sensor_period,
            "measurement_noise_sigma": self.measurement_noise_sigma,
            "cbf_transition_mode": self.cbf_transition_mode,
            "manifest_path": self.path,
            "manifest_hash": self.manifest_hash,
            "manifest_profile": self.profile,
            "model_substitution": self.model_substitution,
            "goal_offset": self.goal_offset,
            "obstacle_start_forward_offset": self.obstacle_start_forward_offset,
            "obstacle_start_vertical_offset": self.obstacle_start_vertical_offset,
            "obstacle_radius": self.obstacle_radius,
            "collision_radius": self.collision_radius,
            "gamma_update_ttl": self.gamma_update_ttl,
            "solver_max_constraint_violation": self.solver_max_constraint_violation,
            "solver_max_cpu_time": self.solver_max_cpu_time,
            "control_deadline": self.control_deadline,
            "reject_deadline_miss": self.reject_deadline_miss,
            "initial_query": self.initial_query,
            "feedback_prompt": self.feedback_query,
        }

    def accepts_feedback_decision(
        self, *, model: str, provider: str, cache_hit: bool
    ) -> bool:
        """Return whether a recorded decision matches the paper LLM contract."""

        return (
            self.required_model_family.lower() in model.lower()
            and provider.lower() == self.required_provider
            and not cache_hit
        )
