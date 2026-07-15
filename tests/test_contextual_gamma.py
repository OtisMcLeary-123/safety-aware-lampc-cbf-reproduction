from __future__ import annotations

import json

import pytest

from lampc_cbf.contextual_gamma import (
    ContextualGammaConfig,
    ContextualNvidiaNIMGammaMapper,
    FeedbackHazardContext,
    feedback_context_from_condition,
)


def test_contextual_prompt_contains_formulation_and_hazard(tmp_path) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")
    captured = {}

    def transport(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return json.dumps(
            {"choices": [{"message": {"content": '{"gamma":0.03,"safety_level":1}'}}]}
        ).encode()

    mapper = ContextualNvidiaNIMGammaMapper(
        ContextualGammaConfig(token_path=str(token)), transport=transport
    )
    context = FeedbackHazardContext(
        current_gamma=0.15,
        obstacle_distance_m=0.14,
        combined_radius_m=0.135,
        clearance_m=0.005,
        predicted_ttc_s=0.4,
        obstacle_speed_mps=0.2,
        minimum_cbf_residual=-0.001,
        intervention_time_s=0.2,
    )
    decision = mapper.infer_gamma("Watch out, collision soon", context)

    assert decision.gamma == pytest.approx(0.03)
    prompt = captured["payload"]["messages"][1]["content"]
    assert "current_formulation" in prompt
    assert "predicted_ttc_s" in prompt
    assert "current_gamma" in prompt


def test_contextual_mapper_rejects_label_mismatch(tmp_path) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")

    def transport(request, timeout):
        del request, timeout
        return json.dumps(
            {"choices": [{"message": {"content": '{"gamma":0.03,"safety_level":4}'}}]}
        ).encode()

    mapper = ContextualNvidiaNIMGammaMapper(
        ContextualGammaConfig(token_path=str(token)), transport=transport
    )
    decision = mapper.infer_gamma("be cautious", FeedbackHazardContext.neutral())

    assert decision.fallback_used
    assert decision.error_type == "ValueError"
    assert decision.gamma == pytest.approx(0.05)


def test_condition_context_reflects_speed_and_intervention() -> None:
    slow = feedback_context_from_condition(
        {"intervention_time": 0.1, "obstacle_speed": 0.025, "lateral_offset": 0.0},
        current_gamma=0.15,
        forward_offset_m=0.44,
        combined_radius_m=0.135,
        reference_speed_mps=0.08,
    )
    fast = feedback_context_from_condition(
        {"intervention_time": 0.3, "obstacle_speed": 0.20, "lateral_offset": 0.0},
        current_gamma=0.15,
        forward_offset_m=0.44,
        combined_radius_m=0.135,
        reference_speed_mps=0.08,
    )

    assert fast.clearance_m < slow.clearance_m
    assert fast.predicted_ttc_s < slow.predicted_ttc_s
