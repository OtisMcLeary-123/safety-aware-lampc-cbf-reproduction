"""Stage-0 tests: deterministic generation, gates, freezing, resume."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lampc_cbf.core_scenarios import (
    PLAN_PATH,
    PreflightError,
    ScenarioInstance,
    check_preflight,
    generate_instances,
    holm_adjust,
    instances_hash,
    latin_hypercube,
    load_frozen_instances,
    load_plan,
    load_prediction_feedback_manifest,
    load_scripted_feedback_manifest,
    mcnemar_exact_pvalue,
    paired_comparison,
    pilot_episode_ids,
    run_core_benchmark,
    runner_kwargs,
    scripted_gamma_schedule,
    scripted_prediction_schedule,
    smoke_episode_ids,
    wilson_interval,
    write_frozen_instances,
)

SCRIPTED_MANIFEST_PATH = "configs/safe_panda_core_scripted_feedback_v1.json"
PREDICTION_MANIFEST_PATH = "configs/safe_panda_core_prediction_feedback_v1.json"
FROZEN_SHA = "e0f154604660f79d6dfb26d83ed5b60955b8977f991365a228e739ff8fededdf"


@pytest.fixture(scope="module")
def plan() -> dict:
    return load_plan(PLAN_PATH)


@pytest.fixture(scope="module")
def instances(plan: dict) -> list[ScenarioInstance]:
    return generate_instances(plan)


def test_generation_is_deterministic(plan: dict, instances: list) -> None:
    assert instances_hash(generate_instances(plan)) == instances_hash(instances)


def test_counts_families_and_balance(plan: dict, instances: list) -> None:
    assert len(instances) == 150
    for family_index, family in enumerate(plan["scenario_families"]):
        members = [i for i in instances if i.family_index == family_index]
        assert len(members) == 50
        assert all(i.scenario_id == family["id"] for i in members)
    for family_index in (1, 2):
        sides = [
            i.crossing_side for i in instances if i.family_index == family_index
        ]
        assert sides.count(-1) == 25 and sides.count(1) == 25
    assert all(i.crossing_side is None for i in instances if i.family_index == 0)


def test_ids_and_seeds_unique_and_blocked(instances: list) -> None:
    ids = [i.episode_id for i in instances]
    geometry = [i.geometry_seed for i in instances]
    measurement = [i.measurement_seed for i in instances]
    assert len(set(ids)) == 150
    assert len(set(geometry)) == 150
    assert len(set(geometry) & set(measurement)) == 0
    for instance in instances:
        block = 20260716 + instance.family_index * 10000
        assert instance.geometry_seed == block + instance.episode_index
        assert instance.measurement_seed == instance.geometry_seed + 500000


def test_sampled_parameters_stay_in_declared_ranges(
    plan: dict, instances: list
) -> None:
    for instance in instances:
        family = plan["scenario_families"][instance.family_index]
        for name, spec in family["perturbations"].items():
            if spec["distribution"] != "uniform":
                continue
            value = instance.parameters[name]
            assert spec["low"] <= value <= spec["high"], (
                instance.episode_id,
                name,
            )


def test_every_instance_passes_preflight_gates(plan: dict, instances: list) -> None:
    gates = plan["preflight_constraints"]
    for instance in instances:
        record = check_preflight(
            plan,
            instance.goal_offset,
            instance.obstacle_start_offset,
            instance.obstacle_velocity,
            instance.obstacle_radius_m,
            attempts=instance.preflight.attempts,
        )
        assert record is not None, instance.episode_id
        assert record.initial_true_clearance_m >= gates["initial_true_clearance_min_m"]
        assert record.encounter_margin_m <= gates["predicted_closest_approach_upper_m"]


def test_rejection_exhaustion_is_a_preflight_failure(plan: dict) -> None:
    impossible = copy.deepcopy(plan)
    family = impossible["scenario_families"][0]["perturbations"]
    # Obstacle far outside any meaningful-encounter reach of the goal path.
    family["obstacle_velocity_y_mps"] = {
        "distribution": "uniform",
        "low": 0.0001,
        "high": 0.0002,
    }
    family["obstacle_start_y_m"] = {
        "distribution": "uniform",
        "low": 0.49,
        "high": 0.499,
    }
    family["obstacle_radius_m"] = {
        "distribution": "uniform",
        "low": 0.075,
        "high": 0.076,
    }
    impossible["preflight_constraints"]["maximum_sampling_attempts_per_episode"] = 25
    with pytest.raises(PreflightError, match="preflight exhausted"):
        generate_instances(impossible)


def test_latin_hypercube_stratifies_each_dimension() -> None:
    points = latin_hypercube(50, 4, np.random.default_rng(7))
    for dim in range(4):
        strata = np.floor(points[:, dim] * 50).astype(int)
        assert sorted(strata.tolist()) == list(range(50))


def test_freeze_is_idempotent_and_refuses_divergence(
    plan: dict, instances: list, tmp_path: Path
) -> None:
    target = tmp_path / "instances.json"
    first = write_frozen_instances(instances, path=target, plan_path=PLAN_PATH)
    second = write_frozen_instances(instances, path=target, plan_path=PLAN_PATH)
    assert first["instances_sha256"] == second["instances_sha256"]
    _, loaded = load_frozen_instances(target)
    assert instances_hash(loaded) == first["instances_sha256"]
    with pytest.raises(PreflightError, match="different hash"):
        write_frozen_instances(instances[:-1], path=target, plan_path=PLAN_PATH)


def test_runner_kwargs_build_valid_frozen_config(plan: dict, instances: list) -> None:
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    kwargs = runner_kwargs(instances[0], plan)
    config = SmoothDynamicConfig(**kwargs)
    assert config.gamma == 0.15
    assert config.max_steps == 260
    assert config.sensor_period == 0.67
    assert config.cbf_transition_mode == "double_integrator"
    assert config.reference_mode == "direct_target"
    assert config.collision_radius == 0.035
    assert not config.safety_reflex_enabled


def test_stage_subsets_are_deterministic(instances: list) -> None:
    smoke = smoke_episode_ids(instances)
    assert len(smoke) == 3
    assert [episode_id[:3] for episode_id in smoke] == ["CS1", "CS2", "CS3"]
    pilot = pilot_episode_ids(instances)
    assert len(pilot) == 15
    assert smoke == smoke_episode_ids(instances)


def test_wilson_interval_matches_reference_case() -> None:
    low, high = wilson_interval(45, 50)
    assert 0.786 < low < 0.792
    assert 0.952 < high < 0.958


class _StubResult:
    outcome = "goal"
    reached_goal = True
    collision = False
    steps = 100
    minimum_true_clearance = 0.05
    minimum_measured_clearance = 0.06
    minimum_true_cbf_residual = 0.001
    true_cbf_violation_steps = 0
    final_goal_distance = 0.03
    solver_failures = 0
    solver_rejections = 0
    deadline_misses = 0
    emergency_fallbacks = 0
    mean_model_transition_error = 0.0002
    max_model_transition_error = 0.0008

    class smoothness:  # noqa: N801 - mirrors SmoothnessMetrics attribute access
        path_length = 0.4
        acceleration_rms = 0.02


def test_run_checkpoints_every_row_and_resumes(
    plan: dict, instances: list, tmp_path: Path
) -> None:
    instance_file = tmp_path / "instances.json"
    write_frozen_instances(instances, path=instance_file, plan_path=PLAN_PATH)
    output = tmp_path / "out"
    calls: list[str] = []

    def stub_runner(**kwargs):
        calls.append(kwargs["output_dir"])
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return _StubResult()

    episode_ids = smoke_episode_ids(instances)
    summary = run_core_benchmark(
        instances_path=instance_file,
        plan_path=PLAN_PATH,
        output_dir=output,
        episode_ids=episode_ids,
        episode_runner=stub_runner,
    )
    assert summary["rows"] == 3
    checkpoint = json.loads((output / "run_checkpoint.json").read_text())
    assert checkpoint["completed_rows"] == 3
    with (output / "episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    crashed = [row for row in rows if row["outcome"] == "crashed"]
    assert len(crashed) == 1 and "simulated crash" in crashed[0]["error"]

    # Resume: already-completed rows never re-run, including the crashed one.
    calls.clear()
    summary = run_core_benchmark(
        instances_path=instance_file,
        plan_path=PLAN_PATH,
        output_dir=output,
        episode_ids=episode_ids,
        episode_runner=stub_runner,
    )
    assert calls == []
    assert summary["rows"] == 3


def test_scripted_manifest_loads_and_is_pinned(instances: list) -> None:
    manifest = load_scripted_feedback_manifest(
        SCRIPTED_MANIFEST_PATH, instances_sha256=FROZEN_SHA
    )
    assert manifest["method"] == "safe_panda_core_scripted_feedback"
    with pytest.raises(PreflightError, match="different frozen instance set"):
        load_scripted_feedback_manifest(
            SCRIPTED_MANIFEST_PATH, instances_sha256="other"
        )


def test_scripted_schedule_is_deterministic_and_sorted(instances: list) -> None:
    manifest = load_scripted_feedback_manifest(
        SCRIPTED_MANIFEST_PATH, instances_sha256=FROZEN_SHA
    )
    instance = instances[0]
    schedule, request_times = scripted_gamma_schedule(instance, manifest)
    assert len(schedule) == len(manifest["script"]) == len(request_times)
    times = [time for time, _ in schedule]
    assert times == sorted(times)
    assert times[0] == pytest.approx(instance.feedback_intervention_time_s)
    assert times[1] == pytest.approx(
        instance.feedback_intervention_time_s + 1.7
    )
    assert [gamma for _, gamma in schedule] == [0.07, 0.03]
    # Same instance always produces the identical script.
    assert scripted_gamma_schedule(instance, manifest) == (schedule, request_times)


def test_scripted_run_builds_valid_config_and_labels_method(
    plan: dict, instances: list, tmp_path: Path
) -> None:
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    instance_file = tmp_path / "instances.json"
    write_frozen_instances(instances, path=instance_file, plan_path=PLAN_PATH)
    captured: list[dict] = []

    def stub_runner(**kwargs):
        captured.append(kwargs)
        SmoothDynamicConfig(**kwargs)
        return _StubResult()

    summary = run_core_benchmark(
        instances_path=instance_file,
        plan_path=PLAN_PATH,
        output_dir=tmp_path / "out",
        episode_ids=smoke_episode_ids(instances),
        episode_runner=stub_runner,
        scripted_feedback_manifest=SCRIPTED_MANIFEST_PATH,
    )
    assert summary["method"] == "safe_panda_core_scripted_feedback"
    assert all(len(kwargs["gamma_schedule"]) == 2 for kwargs in captured)
    assert all(kwargs["gamma_update_ttl"] == 10.4 for kwargs in captured)


def test_controller_profile_manifest_validation(instances: list) -> None:
    from lampc_cbf.core_scenarios import load_controller_profile_manifest

    manifest = load_controller_profile_manifest(
        "configs/safe_panda_core_deadtime_margin_v1.json",
        instances_sha256=FROZEN_SHA,
    )
    assert manifest["method"] == "safe_panda_core_deadtime_margin"
    assert manifest["runner_kwargs_overrides"]["dead_time_margin_mode"] == "speed_bound"
    with pytest.raises(PreflightError, match="different frozen instance set"):
        load_controller_profile_manifest(
            "configs/safe_panda_core_deadtime_margin_v1.json",
            instances_sha256="other",
        )


def test_controller_profile_rejects_frozen_key_overrides(tmp_path: Path) -> None:
    from lampc_cbf.core_scenarios import load_controller_profile_manifest

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "method": "m",
                "provider_requests": 0,
                "instances_sha256": FROZEN_SHA,
                "runner_kwargs_overrides": {"gamma": 0.05},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen-contract keys"):
        load_controller_profile_manifest(bad, instances_sha256=FROZEN_SHA)


def test_dead_time_margin_grows_with_measurement_age() -> None:
    import numpy as np

    from lampc_cbf.smooth_dynamic_demo import ReferenceObstacleTVP

    path = np.linspace([0.0, 0.0, 0.0], [0.0, 0.3, 0.0], 50)
    tvp = ReferenceObstacleTVP(
        path,
        (0.0, 0.4, 0.0),
        reference_speed=0.08,
        obstacle_radius=0.1,
        collision_radius=0.035,
        gamma=0.15,
        dt=0.04,
        horizon=15,
        prediction_mode="static",
        direct_target=True,
        cbf_transition_mode="double_integrator",
        dead_time_margin_mode="speed_bound",
        dead_time_speed_bound=0.21,
    )
    tvp.control_time = 0.5  # observer timestamp starts at 0 -> age 0.5 s
    stage0 = tvp.prediction_at_stage(0)
    stage5 = tvp.prediction_at_stage(5)
    base = 0.1 + 0.035
    assert stage0.robust_radius == pytest.approx(base + 0.21 * 0.5)
    assert stage0.robust_radius_next == pytest.approx(base + 0.21 * 0.54)
    assert stage5.robust_radius == pytest.approx(base + 0.21 * 0.7)
    assert stage5.robust_radius > stage0.robust_radius


def test_dead_time_margin_off_preserves_frozen_baseline() -> None:
    import numpy as np

    from lampc_cbf.smooth_dynamic_demo import ReferenceObstacleTVP, SmoothDynamicConfig

    path = np.linspace([0.0, 0.0, 0.0], [0.0, 0.3, 0.0], 50)
    tvp = ReferenceObstacleTVP(
        path,
        (0.0, 0.4, 0.0),
        reference_speed=0.08,
        obstacle_radius=0.1,
        collision_radius=0.035,
        gamma=0.15,
        dt=0.04,
        horizon=15,
        prediction_mode="static",
        direct_target=True,
        cbf_transition_mode="double_integrator",
    )
    tvp.control_time = 0.5
    assert tvp.prediction_at_stage(0).robust_radius == pytest.approx(0.135)
    with pytest.raises(ValueError, match="requires prediction_mode static"):
        SmoothDynamicConfig(
            dead_time_margin_mode="speed_bound",
            prediction_mode="velocity_tube",
        )


def test_prediction_manifest_loads_and_is_pinned(instances: list) -> None:
    manifest = load_prediction_feedback_manifest(
        PREDICTION_MANIFEST_PATH, instances_sha256=FROZEN_SHA
    )
    assert manifest["method"] == "safe_panda_core_prediction_feedback"
    with pytest.raises(PreflightError, match="different frozen instance set"):
        load_prediction_feedback_manifest(
            PREDICTION_MANIFEST_PATH, instances_sha256="other"
        )


def test_prediction_schedule_never_touches_gamma(instances: list) -> None:
    manifest = load_prediction_feedback_manifest(
        PREDICTION_MANIFEST_PATH, instances_sha256=FROZEN_SHA
    )
    instance = instances[0]
    schedule, request_times = scripted_prediction_schedule(instance, manifest)
    assert schedule == ((instance.feedback_intervention_time_s, "velocity_tube"),)
    assert request_times == (instance.feedback_intervention_time_s,)
    # Same instance always produces the identical schedule.
    assert scripted_prediction_schedule(instance, manifest) == (schedule, request_times)


def test_prediction_feedback_run_isolates_the_prediction_channel(
    plan: dict, instances: list, tmp_path: Path
) -> None:
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    instance_file = tmp_path / "instances.json"
    write_frozen_instances(instances, path=instance_file, plan_path=PLAN_PATH)
    captured: list[dict] = []

    def stub_runner(**kwargs):
        captured.append(kwargs)
        SmoothDynamicConfig(**kwargs)
        return _StubResult()

    summary = run_core_benchmark(
        instances_path=instance_file,
        plan_path=PLAN_PATH,
        output_dir=tmp_path / "out",
        episode_ids=smoke_episode_ids(instances),
        episode_runner=stub_runner,
        prediction_feedback_manifest=PREDICTION_MANIFEST_PATH,
    )
    assert summary["method"] == "safe_panda_core_prediction_feedback"
    assert all("gamma_schedule" not in kwargs for kwargs in captured)
    assert all(kwargs["gamma"] == 0.15 for kwargs in captured)
    assert all(
        len(kwargs["prediction_mode_schedule"]) == 1 for kwargs in captured
    )
    assert all(
        kwargs["prediction_mode_schedule"][0][1] == "velocity_tube"
        for kwargs in captured
    )


def test_soft_slack_manifest_and_config_validation() -> None:
    from lampc_cbf.core_scenarios import load_controller_profile_manifest
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    manifest = load_controller_profile_manifest(
        "configs/safe_panda_core_soft_slack_v1.json",
        instances_sha256=FROZEN_SHA,
    )
    overrides = manifest["runner_kwargs_overrides"]
    assert overrides["cbf_slack_weight"] == 1000.0
    config = SmoothDynamicConfig(**overrides)
    assert config.cbf_slack_weight == 1000.0
    with pytest.raises(ValueError, match="non-negative"):
        SmoothDynamicConfig(cbf_slack_weight=-1.0)


def test_cbf_slack_input_appears_only_when_enabled() -> None:
    from lampc_cbf.controller import PaperMPCConfig

    assert not PaperMPCConfig().uses_cbf_slack
    assert PaperMPCConfig(cbf_slack_weight=1000.0).uses_cbf_slack
    with pytest.raises(ValueError, match="cbf_slack_weight"):
        PaperMPCConfig(cbf_slack_weight=-0.1)


def test_mcnemar_and_holm_reference_values() -> None:
    assert mcnemar_exact_pvalue([True, False], [True, False]) == 1.0
    # 5 discordant pairs, all favoring the method: p = 2 * C(5,0) / 2^5.
    base = [False] * 5 + [True] * 5
    meth = [True] * 10
    assert mcnemar_exact_pvalue(base, meth) == pytest.approx(2 * 1 / 32)
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)


def test_paired_comparison_requires_identical_episode_sets() -> None:
    row = {"episode_id": "CS1-E00", "scenario_id": "CS1", "joint_success": True}
    with pytest.raises(ValueError, match="identical episode sets"):
        paired_comparison([row], [])


def test_run_refuses_checkpoint_from_other_instance_set(
    plan: dict, instances: list, tmp_path: Path
) -> None:
    instance_file = tmp_path / "instances.json"
    write_frozen_instances(instances, path=instance_file, plan_path=PLAN_PATH)
    output = tmp_path / "out"
    output.mkdir()
    (output / "run_checkpoint.json").write_text(
        json.dumps({"instances_sha256": "not-the-same"}), encoding="utf-8"
    )
    with pytest.raises(PreflightError, match="different frozen instance set"):
        run_core_benchmark(
            instances_path=instance_file,
            plan_path=PLAN_PATH,
            output_dir=output,
            episode_ids=smoke_episode_ids(instances),
            episode_runner=lambda **kwargs: _StubResult(),
        )
