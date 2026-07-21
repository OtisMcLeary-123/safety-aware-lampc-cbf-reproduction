"""Continuous Table-1/2 gamma mapping and fail-closed NIM client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lampc_cbf.paper_continuous_gamma import (
    NIMContinuousGammaConfig,
    NIMContinuousGammaMapper,
    table2_label,
    validate_continuous_gamma,
)


def test_table2_labels_match_printed_ranges() -> None:
    assert table2_label(0.001) == 1
    assert table2_label(0.06) == 1
    assert table2_label(0.0601) == 2
    assert table2_label(0.08) == 2
    assert table2_label(0.081) == 3
    assert table2_label(0.1499) == 3
    assert table2_label(0.15) == 4  # printed overlap resolves to label 4
    assert table2_label(0.1501) == 5
    assert table2_label(1.0) == 5
    for invalid in (0.0, -0.1, 1.0001):
        with pytest.raises(ValueError):
            table2_label(invalid)


def test_validate_continuous_gamma_range() -> None:
    assert validate_continuous_gamma(1.0) == 1.0
    with pytest.raises(ValueError):
        validate_continuous_gamma(0.0)


def _stub_transport_factory(body: str):
    def transport(request, timeout):
        del request, timeout
        return json.dumps(
            {"choices": [{"message": {"content": body}}]}
        ).encode("utf-8")

    return transport


def test_mapper_parses_validates_and_checkpoints(tmp_path: Path) -> None:
    checkpoint = tmp_path / "decisions.jsonl"
    config = NIMContinuousGammaConfig(
        token_path=str(tmp_path / "token.txt"),
        checkpoint_path=str(checkpoint),
    )
    (tmp_path / "token.txt").write_text("stub-token", encoding="utf-8")
    mapper = NIMContinuousGammaMapper(
        config, transport=_stub_transport_factory('{"gamma": 0.03}')
    )
    decision = mapper.infer(
        "Watch out!", current_gamma=0.15, task="Move gripper to red cube."
    )
    assert decision.gamma == 0.03
    assert decision.table2_label == 1
    assert not decision.fallback_used and not decision.cache_hit
    assert checkpoint.is_file()
    # Identical request replays from the checkpoint without any transport.
    replayed = NIMContinuousGammaMapper(
        config, transport=_stub_transport_factory("must-not-be-called")
    ).infer("Watch out!", current_gamma=0.15, task="Move gripper to red cube.")
    assert replayed.cache_hit and replayed.gamma == 0.03


def test_mapper_fails_closed_on_bad_output(tmp_path: Path) -> None:
    (tmp_path / "token.txt").write_text("stub-token", encoding="utf-8")
    mapper = NIMContinuousGammaMapper(
        NIMContinuousGammaConfig(
            token_path=str(tmp_path / "token.txt"), checkpoint_path=None
        ),
        transport=_stub_transport_factory('{"gamma": 7.0}'),
    )
    decision = mapper.infer(
        "Watch out!", current_gamma=0.15, task="Move gripper to red cube."
    )
    assert decision.fallback_used
    assert decision.gamma == 0.15
    assert "ValueError" in decision.error


def test_gamma_range_mode_gates_schedule_range() -> None:
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    with pytest.raises(ValueError, match="configured gamma range"):
        SmoothDynamicConfig(gamma_schedule=((1.0, 0.4),))
    config = SmoothDynamicConfig(
        gamma_schedule=((1.0, 0.4),), gamma_range_mode="paper_continuous"
    )
    assert config.gamma_upper == 1.0
