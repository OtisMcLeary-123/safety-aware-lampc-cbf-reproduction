"""Fail-closed NVIDIA NIM adapter for controller CBF parameters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable
from urllib.request import Request, urlopen

from .hf_llm import (
    GAMMA_LEVELS,
    GAMMA_SCHEMA,
    SYSTEM_PROMPT,
    GammaDecision,
    HuggingFaceGammaMapper,
    _validate_gamma,
)


NVIDIA_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_NIM_MODEL = "meta/llama-3.1-8b-instruct"


@dataclass(frozen=True, slots=True)
class NvidiaNIMGammaConfig:
    model: str = DEFAULT_NVIDIA_NIM_MODEL
    endpoint: str = NVIDIA_NIM_ENDPOINT
    token_path: str = "nvidiatoken.txt"
    timeout_seconds: float = 3.0
    fallback_gamma: float = 0.05
    temperature: float = 0.0
    seed: int = 11
    max_tokens: int = 64
    cache_path: str | None = "results/nvidia_nim_gamma_cache.jsonl"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.endpoint.startswith("https://"):
            raise ValueError("NVIDIA NIM endpoint must use HTTPS")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be in [0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        _validate_gamma(self.fallback_gamma)


def load_nvidia_token(path: str | Path) -> str:
    """Load the local NIM token without exporting or logging it."""

    token_path = Path(path)
    if not token_path.is_file():
        raise FileNotFoundError(f"NVIDIA NIM token file not found: {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("NVIDIA NIM token file must contain one non-empty token")
    return token


class NvidiaNIMGammaMapper:
    """Map safety language through NIM guided JSON and strict local validation."""

    def __init__(
        self,
        config: NvidiaNIMGammaConfig | None = None,
        *,
        transport: Callable[[Request, float], bytes] | None = None,
    ) -> None:
        self.config = config or NvidiaNIMGammaConfig()
        self._transport = transport or self._default_transport

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS config
            return response.read()

    def infer_gamma(
        self,
        instruction: str,
        *,
        current_gamma: float | None = None,
        feedback: bool = False,
    ) -> GammaDecision:
        if not instruction.strip():
            raise ValueError("instruction must be non-empty")
        if feedback and current_gamma is None:
            raise ValueError("online feedback requires current_gamma context")
        if current_gamma is not None:
            _validate_gamma(current_gamma)
        messages = HuggingFaceGammaMapper._messages(
            instruction, current_gamma, feedback
        )
        schema = GAMMA_SCHEMA["json_schema"]["schema"]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "stream": False,
            "nvext": {"guided_json": schema},
        }
        request_hash = sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cached = self._read_cache(request_hash)
        if cached is not None:
            cached["cache_hit"] = True
            return GammaDecision(**cached)

        requested_at = time()
        started = perf_counter()
        raw_response: str | None = None
        try:
            token = load_nvidia_token(self.config.token_path)
            request = Request(
                self.config.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            envelope = json.loads(
                self._transport(request, self.config.timeout_seconds).decode("utf-8")
            )
            raw_response = envelope["choices"][0]["message"]["content"]
            parsed = json.loads(raw_response)
            if set(parsed) != {"gamma_text", "safety_level"}:
                raise ValueError("NIM output does not match the exact gamma schema")
            gamma_text = str(parsed["gamma_text"])
            safety_level = int(parsed["safety_level"])
            if gamma_text not in GAMMA_LEVELS:
                raise ValueError("gamma_text outside calibrated enum")
            if safety_level != GAMMA_LEVELS[gamma_text]:
                raise ValueError("gamma_text and safety_level are inconsistent")
            decision = GammaDecision(
                gamma=_validate_gamma(float(gamma_text)),
                safety_level=safety_level,
                explanation=f"NIM mapped instruction to calibrated safety level {safety_level}.",
                model=self.config.model,
                provider="nvidia-nim",
                latency_seconds=perf_counter() - started,
                requested_at_unix=requested_at,
                prompt_hash=sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                request_hash=request_hash,
                raw_response=raw_response,
                fallback_used=False,
                cache_hit=False,
                error_type=None,
            )
        except Exception as error:  # external inference must fail closed
            decision = GammaDecision(
                gamma=self.config.fallback_gamma,
                safety_level=2,
                explanation="Bounded safety fallback after unavailable or invalid NIM output.",
                model=self.config.model,
                provider="nvidia-nim",
                latency_seconds=perf_counter() - started,
                requested_at_unix=requested_at,
                prompt_hash=sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                request_hash=request_hash,
                raw_response=raw_response,
                fallback_used=True,
                cache_hit=False,
                error_type=type(error).__name__,
            )
        self._append_cache(decision)
        return decision

    def _read_cache(self, request_hash: str) -> dict[str, Any] | None:
        if self.config.cache_path is None:
            return None
        path = Path(self.config.cache_path)
        if not path.exists():
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("request_hash") == request_hash:
                return item
        return None

    def _append_cache(self, decision: GammaDecision) -> None:
        if self.config.cache_path is None or decision.fallback_used:
            return
        path = Path(self.config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision.as_dict(), ensure_ascii=False) + "\n")
