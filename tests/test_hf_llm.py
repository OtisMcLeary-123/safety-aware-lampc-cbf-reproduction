from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lampc_cbf.hf_llm import (
    HFLLMConfig,
    HuggingFaceGammaMapper,
    load_hf_token,
)


class FakeClient:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error

    def chat_completion(self, **kwargs):
        assert kwargs["response_format"]["type"] == "json_schema"
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_token_loader_does_not_change_or_expose_token(tmp_path: Path) -> None:
    path = tmp_path / "token.txt"
    path.write_text("hf_private_value\n", encoding="utf-8")

    assert load_hf_token(path) == "hf_private_value"


def test_structured_gamma_is_validated_without_network(tmp_path: Path) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test", encoding="utf-8")
    client = FakeClient(json.dumps({"gamma_text": "0.02", "safety_level": 1}))
    mapper = HuggingFaceGammaMapper(
        HFLLMConfig(token_path=str(token_path), cache_path=None),
        client_factory=lambda config, token: client,
    )

    decision = mapper.infer_gamma("Please stay far away.")

    assert decision.gamma == pytest.approx(0.02)
    assert not decision.fallback_used
    assert decision.raw_response is not None


def test_invalid_external_output_uses_bounded_fallback(tmp_path: Path) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test", encoding="utf-8")
    mapper = HuggingFaceGammaMapper(
        HFLLMConfig(token_path=str(token_path), fallback_gamma=0.04, cache_path=None),
        client_factory=lambda config, token: FakeClient("not-json"),
    )

    decision = mapper.infer_gamma("Move normally.")

    assert decision.gamma == pytest.approx(0.04)
    assert decision.fallback_used
    assert decision.error_type == "JSONDecodeError"


def test_feedback_includes_current_gamma(tmp_path: Path) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test", encoding="utf-8")
    captured = {}
    client = FakeClient(json.dumps({"gamma_text": "0.02", "safety_level": 1}))

    def call(**kwargs):
        captured.update(kwargs)
        return FakeClient.chat_completion(client, **kwargs)

    client.chat_completion = call
    mapper = HuggingFaceGammaMapper(
        HFLLMConfig(token_path=str(token_path), cache_path=None),
        client_factory=lambda config, token: client,
    )

    mapper.infer_gamma("Watch out!", current_gamma=0.1, feedback=True)

    assert "current gamma=0.100000" in captured["messages"][1]["content"]


def test_feedback_requires_current_gamma(tmp_path: Path) -> None:
    mapper = HuggingFaceGammaMapper(
        HFLLMConfig(token_path=str(tmp_path / "missing"), cache_path=None)
    )

    with pytest.raises(ValueError, match="requires current_gamma"):
        mapper.infer_gamma("Be safer.", feedback=True)


def test_fallback_is_not_cached(tmp_path: Path) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test", encoding="utf-8")
    cache_path = tmp_path / "cache.jsonl"
    mapper = HuggingFaceGammaMapper(
        HFLLMConfig(token_path=str(token_path), cache_path=str(cache_path)),
        client_factory=lambda config, token: FakeClient(error=TimeoutError()),
    )

    assert mapper.infer_gamma("Be safe.").fallback_used
    assert not cache_path.exists()
