from lampc_cbf.remediation_benchmark import deterministic_scenario_grid


def test_deterministic_grid_is_reproducible_unique_and_bounded() -> None:
    first = deterministic_scenario_grid(20, seed=7)
    second = deterministic_scenario_grid(20, seed=7)

    assert first == second
    assert len(first) == 20
    assert len({(row["obstacle_speed"], row["lateral_offset"]) for row in first}) == 20
    assert min(row["obstacle_speed"] for row in first) == 0.025
    assert max(row["obstacle_speed"] for row in first) == 0.20
    assert {row["lateral_offset"] > 0 for row in first} == {False, True}
