from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from lampc_cbf.nvidia_nim_gamma import (
    NVIDIA_NIM_GAMMA_OUTPUT_CONTRACT,
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


def test_nim_can_use_prompt_only_json_contract_for_models_without_guidance(
    tmp_path,
) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")
    captured = {}
    mapper = NvidiaNIMGammaMapper(
        NvidiaNIMGammaConfig(
            token_path=str(token),
            cache_path=None,
            guided_json_enabled=False,
            enable_thinking=False,
        ),
        transport=_transport('{"gamma_text":"0.02","safety_level":1}', captured),
    )

    decision = mapper.infer_gamma(
        "Increase clearance now", current_gamma=0.15, feedback=True
    )

    assert not decision.fallback_used
    assert "nvext" not in captured["payload"]
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert NVIDIA_NIM_GAMMA_OUTPUT_CONTRACT in (
        captured["payload"]["messages"][-1]["content"]
    )


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


def test_nim_records_http_status_without_exposing_response(tmp_path) -> None:
    token = tmp_path / "token.txt"
    token.write_text("secret", encoding="utf-8")

    def rate_limited(request, timeout):
        del request, timeout
        raise HTTPError("https://example.invalid", 429, "rate limited", {}, None)

    mapper = NvidiaNIMGammaMapper(
        NvidiaNIMGammaConfig(token_path=str(token), cache_path=None),
        transport=rate_limited,
    )
    decision = mapper.infer_gamma("increase clearance")

    assert decision.fallback_used
    assert decision.error_type == "HTTPError:429"
    assert decision.raw_response is None
