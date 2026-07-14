from __future__ import annotations

import json

import pytest

from lampc_cbf.nvidia_nim_gamma import (
    NvidiaNIMGammaConfig,
    NvidiaNIMGammaMapper,
)


def _transport(content: str, captured: dict):
    def send(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode()

    return send


def test_nim_uses_guided_json_and_validates_feedback_context(tmp_path) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")
    captured = {}
    mapper = NvidiaNIMGammaMapper(
        NvidiaNIMGammaConfig(
            token_path=str(token), cache_path=None, timeout_seconds=1.5
        ),
        transport=_transport('{"gamma_text":"0.02","safety_level":1}', captured),
    )

    decision = mapper.infer_gamma(
        "Increase clearance now", current_gamma=0.15, feedback=True
    )

    assert decision.gamma == pytest.approx(0.02)
    assert decision.provider == "nvidia-nim"
    assert not decision.fallback_used
    assert captured["payload"]["nvext"]["guided_json"]["additionalProperties"] is False
    assert "current gamma=0.150000" in captured["payload"]["messages"][1]["content"]
    assert captured["authorization"] == "Bearer secret"
    assert captured["timeout"] == pytest.approx(1.5)


def test_nim_fails_closed_on_schema_mismatch_and_does_not_cache(tmp_path) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")
    cache = tmp_path / "cache.jsonl"
    mapper = NvidiaNIMGammaMapper(
        NvidiaNIMGammaConfig(token_path=str(token), cache_path=str(cache)),
        transport=_transport(
            '{"gamma_text":"0.02","safety_level":5,"extra":true}', {}
        ),
    )

    decision = mapper.infer_gamma("be safe")

    assert decision.fallback_used
    assert decision.gamma == pytest.approx(0.05)
    assert decision.error_type == "ValueError"
    assert not cache.exists()


def test_nim_success_cache_avoids_second_transport_call(tmp_path) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")
    calls = []

    def transport(request, timeout):
        calls.append((request, timeout))
        return json.dumps(
            {"choices": [{"message": {"content": '{"gamma_text":"0.08","safety_level":3}'}}]}
        ).encode()

    mapper = NvidiaNIMGammaMapper(
        NvidiaNIMGammaConfig(
            token_path=str(token), cache_path=str(tmp_path / "cache.jsonl")
        ),
        transport=transport,
    )
    first = mapper.infer_gamma("normal balance")
    second = mapper.infer_gamma("normal balance")

    assert not first.cache_hit
    assert second.cache_hit
    assert len(calls) == 1
