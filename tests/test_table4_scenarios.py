from __future__ import annotations

import json
from pathlib import Path

import pytest

from lampc_cbf.table4_scenarios import (
    CATEGORY_COUNTS,
    Table4Scenario,
    contextual_feedback_context,
    load_scenarios,
    scenario_runner_kwargs,
    validate_scenarios,
)


SCENARIO_PATH = Path("configs/table4_candidate_50_scenarios.json")


def test_candidate_suite_has_exact_fifty_validated_scenarios() -> None:
    scenarios = load_scenarios(SCENARIO_PATH)
    assert len(scenarios) == 50
    assert [scenario.episode_id for scenario in scenarios] == list(range(1, 51))
    assert {
        category: sum(item.category == category for item in scenarios)
        for category in CATEGORY_COUNTS
    } == CATEGORY_COUNTS
    assert min(item.reaction_margin_m for item in scenarios) >= 0.0


def test_candidate_suite_preserves_requested_noise_and_speed_sweeps() -> None:
    scenarios = load_scenarios(SCENARIO_PATH)
    assert scenarios[0].speed_mps == pytest.approx(0.0125)
    assert scenarios[14].speed_mps == pytest.approx(0.10)
    assert scenarios[30].intervention_time == pytest.approx(0.35)
    assert scenarios[39].intervention_time == pytest.approx(0.40)
    assert scenarios[40].noise_sigma == pytest.approx(0.01)
    assert scenarios[49].noise_sigma == pytest.approx(0.05)


def test_validator_rejects_missing_episode() -> None:
    payload = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))[:-1]
    scenarios = [Table4Scenario.from_mapping(item) for item in payload]
    with pytest.raises(ValueError, match="exactly 50"):
        validate_scenarios(scenarios)


def test_scenario_resolves_to_existing_runner_interface() -> None:
    scenario = load_scenarios(SCENARIO_PATH)[30]
    values = scenario_runner_kwargs(scenario)
    assert values["obstacle_start_offset"] == scenario.obstacle_start_pos
    assert values["obstacle_velocity"] == scenario.obstacle_velocity
    assert values["obstacle_radius"] == 0.0
    assert values["collision_radius"] == pytest.approx(0.035)
    assert values["cbf_transition_mode"] == "command_velocity"


def test_contextual_feedback_context_uses_vector_trajectory() -> None:
    scenario = load_scenarios(SCENARIO_PATH)[15]
    context = contextual_feedback_context(scenario)
    assert context.intervention_time_s == pytest.approx(scenario.intervention_time)
    assert context.obstacle_speed_mps == pytest.approx(scenario.speed_mps)
    assert context.combined_radius_m == pytest.approx(0.035)
    assert context.obstacle_distance_m > context.combined_radius_m
