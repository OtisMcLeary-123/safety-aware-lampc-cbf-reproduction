from __future__ import annotations

import json
from pathlib import Path

import pytest

from lampc_cbf.paper_manifest import (
    PAPER_REPLICATION_METHODS,
    PaperFidelityManifest,
)


MANIFEST_PATH = Path("configs/paper_fidelity.json")
NIM_MANIFEST_PATH = Path("configs/paper_fidelity_nvidia_nim.json")
LLAMA_NIM_MANIFEST_PATH = Path("configs/paper_fidelity_nvidia_nim_llama31.json")


def test_paper_fidelity_manifest_loads_and_hashes() -> None:
    manifest = PaperFidelityManifest.load(MANIFEST_PATH)

    assert manifest.stage == "paper-replication"
    assert manifest.episodes == 50
    assert manifest.bootstrap_resamples == 10_000
    assert manifest.method_names == PAPER_REPLICATION_METHODS
    assert manifest.feedback_schedule_mode == "elapsed_time"
    assert manifest.feedback_request_policy == "one_shot_per_feedback_episode"
    assert manifest.feedback_requests_per_episode == 1
    assert (
        manifest.latency_trace_mode
        == "precollected_uncached_per_episode_replay"
    )
    assert manifest.fixed_lateral_offset == 0.0
    assert manifest.cbf_transition_mode == "paper_state"
    assert manifest.goal_offset == (0.0, 0.3, 0.0)
    assert manifest.obstacle_radius == pytest.approx(0.1)
    assert manifest.collision_radius == pytest.approx(0.035)
    assert manifest.gamma_update_ttl == pytest.approx(8.8)
    assert manifest.initial_query == "Move gripper to red cube."
    assert manifest.feedback_query.startswith("Watch out!")
    assert manifest.required_model_family == "gpt-4o"
    assert manifest.required_provider == "openai"
    assert manifest.accepts_feedback_decision(
        model="gpt-4o-2024-08-06", provider="openai", cache_hit=False
    )
    assert not manifest.accepts_feedback_decision(
        model="Qwen3", provider="deepinfra", cache_hit=False
    )
    assert not manifest.accepts_feedback_decision(
        model="gpt-4o", provider="openai", cache_hit=True
    )
    assert len(manifest.manifest_hash) == 64


def test_manifest_rejects_robust_extension(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["extensions"]["safety_reflex"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbids enabled extensions"):
        PaperFidelityManifest.load(path)


def test_manifest_rejects_non_paper_method_set(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["benchmark"]["method_names"].append("predictive_cbf_g002")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark.method_names"):
        PaperFidelityManifest.load(path)


def test_nvidia_nim_substitution_inherits_controller_contract() -> None:
    manifest = PaperFidelityManifest.load(NIM_MANIFEST_PATH)

    assert manifest.profile == "paper_fidelity_model_substitution"
    assert manifest.model_substitution
    assert manifest.required_provider == "nvidia-nim"
    assert manifest.required_model_family == "z-ai/glm-5.2"
    assert manifest.episodes == 50
    assert manifest.method_names == PAPER_REPLICATION_METHODS
    assert manifest.feedback_request_policy == "one_shot_per_feedback_episode"
    assert manifest.feedback_requests_per_episode == 1
    assert (
        manifest.latency_trace_mode
        == "precollected_uncached_per_episode_replay"
    )
    assert manifest.accepts_feedback_decision(
        model="z-ai/glm-5.2",
        provider="nvidia-nim",
        cache_hit=False,
    )


def test_llama_nim_substitution_profile_is_explicit_and_validated() -> None:
    manifest = PaperFidelityManifest.load(LLAMA_NIM_MANIFEST_PATH)

    assert manifest.profile == "paper_fidelity_model_substitution"
    assert manifest.model_substitution
    assert manifest.required_model_family == "meta/llama-3.1-8b-instruct"
    assert manifest.required_provider == "nvidia-nim"
    assert manifest.llm_timeout_seconds == pytest.approx(30.0)
    assert manifest.llm_max_tokens == 128
    assert manifest.accepts_feedback_decision(
        model="meta/llama-3.1-8b-instruct",
        provider="nvidia-nim",
        cache_hit=False,
    )
