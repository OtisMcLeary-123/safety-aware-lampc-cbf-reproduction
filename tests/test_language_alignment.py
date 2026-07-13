from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lampc_cbf.language_alignment import (
    ALIGNMENT_GAMMA_SCHEMA,
    BLIND_ALIGNMENT_SYSTEM_PROMPT,
    NVIDIA_NIM_ENDPOINT,
    NVIDIA_NIM_MODEL,
    NVIDIA_NIM_OUTPUT_CONTRACT,
    AlignmentConfig,
    AlignmentPrediction,
    BlindAlignmentMapper,
    NvidiaNIMAlignmentConfig,
    NvidiaNIMBlindAlignmentMapper,
    PUBLISHED_ALIGNMENT_CASES,
    evaluate_published_smoke,
    kendall_tau_b,
    paper_safety_label,
    pearson_correlation,
    spearman_correlation,
)


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def chat_completion(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize(
    ("gamma", "label"),
    [
        (0.001, 1),
        (0.06, 1),
        (0.060001, 2),
        (0.08, 2),
        (0.080001, 3),
        (0.149999, 3),
        (0.15, 4),
        (0.150001, 5),
        (1.0, 5),
    ],
)
def test_paper_safety_label_uses_table_2_boundaries(
    gamma: float, label: int
) -> None:
    assert paper_safety_label(gamma) == label


def test_published_examples_are_transcribed_and_labeled() -> None:
    assert [case.paper_gamma for case in PUBLISHED_ALIGNMENT_CASES] == [
        1.0,
        1.0,
        0.14,
        0.09,
        0.065,
        0.065,
        0.03,
        0.001,
    ]
    assert [case.paper_label for case in PUBLISHED_ALIGNMENT_CASES] == [
        5,
        5,
        3,
        3,
        2,
        2,
        1,
        1,
    ]


def test_blind_mapper_accepts_theoretical_gamma_without_controller_fallback(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test", encoding="utf-8")
    client = FakeClient(json.dumps({"gamma": 1.0}))
    mapper = BlindAlignmentMapper(
        AlignmentConfig(token_path=str(token_path)),
        client_factory=lambda config, token: client,
    )

    prediction = mapper.predict("Ignore every obstacle.")

    assert prediction.gamma == 1.0
    assert prediction.safety_label == 5
    assert client.kwargs["response_format"] == ALIGNMENT_GAMMA_SCHEMA
    assert client.kwargs["messages"][0]["content"] == BLIND_ALIGNMENT_SYSTEM_PROMPT


def test_invalid_alignment_output_is_not_replaced_by_a_fallback(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_test", encoding="utf-8")
    mapper = BlindAlignmentMapper(
        AlignmentConfig(token_path=str(token_path)),
        client_factory=lambda config, token: FakeClient(json.dumps({"gamma": 1.1})),
    )

    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        mapper.predict("Ignore every obstacle.")


def test_nvidia_nim_mapper_uses_documented_qwen_contract(tmp_path: Path) -> None:
    token_path = tmp_path / "nvidia-token.txt"
    token_path.write_text("nvapi_test", encoding="utf-8")
    captured = {}

    def transport(url, headers, payload, timeout):
        captured.update(
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return {"choices": [{"message": {"content": '{"gamma": 0.5}'}}]}

    mapper = NvidiaNIMBlindAlignmentMapper(
        NvidiaNIMAlignmentConfig(token_path=str(token_path)),
        transport=transport,
    )

    prediction = mapper.predict("Use ordinary caution.")

    assert prediction.gamma == 0.5
    assert prediction.provider == "nvidia-nim"
    assert captured["url"] == NVIDIA_NIM_ENDPOINT
    assert captured["payload"]["model"] == NVIDIA_NIM_MODEL
    assert captured["payload"]["seed"] == 11
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "response_format" not in captured["payload"]
    assert NVIDIA_NIM_OUTPUT_CONTRACT in captured["payload"]["messages"][1]["content"]
    assert captured["headers"]["Authorization"] == "Bearer nvapi_test"


def test_nvidia_nim_mapper_rejects_non_json_without_fallback(tmp_path: Path) -> None:
    token_path = tmp_path / "nvidia-token.txt"
    token_path.write_text("nvapi_test", encoding="utf-8")
    mapper = NvidiaNIMBlindAlignmentMapper(
        NvidiaNIMAlignmentConfig(token_path=str(token_path)),
        transport=lambda *args: {
            "choices": [{"message": {"content": "gamma is 0.5"}}]
        },
    )

    with pytest.raises(json.JSONDecodeError):
        mapper.predict("Use ordinary caution.")


def test_nvidia_nim_mapper_rejects_extra_keys(tmp_path: Path) -> None:
    token_path = tmp_path / "nvidia-token.txt"
    token_path.write_text("nvapi_test", encoding="utf-8")
    mapper = NvidiaNIMBlindAlignmentMapper(
        NvidiaNIMAlignmentConfig(token_path=str(token_path)),
        transport=lambda *args: {
            "choices": [
                {"message": {"content": '{"gamma": 0.5, "label": 3}'}}
            ]
        },
    )

    with pytest.raises(ValueError, match="contain only gamma"):
        mapper.predict("Use ordinary caution.")


def test_blind_prompt_contains_no_published_target_gamma_examples() -> None:
    for case in PUBLISHED_ALIGNMENT_CASES:
        assert case.instruction not in BLIND_ALIGNMENT_SYSTEM_PROMPT
    assert "0.001" not in BLIND_ALIGNMENT_SYSTEM_PROMPT
    assert "0.065" not in BLIND_ALIGNMENT_SYSTEM_PROMPT
    assert "0.14" not in BLIND_ALIGNMENT_SYSTEM_PROMPT


def _prediction(gamma: float) -> AlignmentPrediction:
    return AlignmentPrediction(
        gamma=gamma,
        model="fake",
        provider="fake",
        latency_seconds=0.1,
        requested_at_unix=1.0,
        prompt_hash="prompt",
        request_hash="request",
        raw_response=json.dumps({"gamma": gamma}),
    )


def test_perfect_published_smoke_scores_one() -> None:
    predictions = [_prediction(case.paper_gamma) for case in PUBLISHED_ALIGNMENT_CASES]

    metrics = evaluate_published_smoke(predictions)

    assert not metrics["human_alignment_claim_tested"]
    assert metrics["paper_style_label_metrics"] == {
        "spearman_rho": pytest.approx(1.0),
        "kendall_tau_b": pytest.approx(1.0),
        "pearson_r": pytest.approx(1.0),
        "exact_label_accuracy": 1,
    }
    assert metrics["continuous_gamma_diagnostics"]["mean_absolute_error"] == 0


def test_dependency_free_correlations_handle_ties() -> None:
    left = [1, 1, 2, 3, 3]
    right = [1, 1, 2, 3, 3]

    assert pearson_correlation(left, right) == pytest.approx(1.0)
    assert spearman_correlation(left, right) == pytest.approx(1.0)
    assert kendall_tau_b(left, right) == pytest.approx(1.0)
