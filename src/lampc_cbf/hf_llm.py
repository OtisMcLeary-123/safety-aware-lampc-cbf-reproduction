"""Fail-closed Hugging Face LLM adapter for language-to-CBF parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable


DEFAULT_HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_HF_PROVIDER = "deepinfra"
GAMMA_LEVELS = {"0.02": 1, "0.05": 2, "0.08": 3, "0.11": 4, "0.15": 5}

SYSTEM_PROMPT = """You are the Optimization Formulator safety-parameter module
for a robot controlled by MPC with a discrete control barrier function (CBF).
Map the user's safety intent to gamma in the experimentally validated interval
0 < gamma <= 0.15. Smaller gamma means earlier, more proactive avoidance and a
larger obstacle clearance. Larger gamma means later, more reactive avoidance.
Return only the requested JSON object. Never generate executable code.

Calibration examples:
- "Keep as far away as possible; be maximally cautious." -> gamma 0.02, level 1
- "Use a normal balance between progress and safety." -> gamma 0.08, level 3
- "Move quickly and use only the required obstacle margin." -> gamma 0.15, level 5

Safety level uses 1 for most cautious and 5 for least cautious. If feedback is
ambiguous, prefer the safer interpretation. The downstream controller validates
the numeric range and retains a bounded fallback; your output never bypasses CBF.
"""

GAMMA_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "CBFSafetyParameter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "gamma_text": {"type": "string", "enum": list(GAMMA_LEVELS)},
                "safety_level": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["gamma_text", "safety_level"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class HFLLMConfig:
    model: str = DEFAULT_HF_MODEL
    provider: str = DEFAULT_HF_PROVIDER
    token_path: str = "hftoken.txt"
    timeout_seconds: float = 8.0
    fallback_gamma: float = 0.05
    temperature: float = 0.0
    seed: int = 11
    cache_path: str | None = "results/hf_gamma_cache.jsonl"

    def __post_init__(self) -> None:
        if not self.model or not self.provider:
            raise ValueError("model and provider must be non-empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        _validate_gamma(self.fallback_gamma)
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")


@dataclass(frozen=True, slots=True)
class GammaDecision:
    gamma: float
    safety_level: int
    explanation: str
    model: str
    provider: str
    latency_seconds: float
    requested_at_unix: float
    prompt_hash: str
    request_hash: str
    raw_response: str | None
    fallback_used: bool
    cache_hit: bool
    error_type: str | None

    def __post_init__(self) -> None:
        _validate_gamma(self.gamma)
        if self.safety_level not in range(1, 6):
            raise ValueError("safety_level must be in [1, 5]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_gamma(gamma: float) -> float:
    if not isfinite(gamma) or not 0.0 < gamma <= 0.15:
        raise ValueError("gamma must be finite and in the experimental interval (0, 0.15]")
    return gamma


def load_hf_token(path: str | Path) -> str:
    """Load a token without logging it or exporting it to process environment."""

    token_path = Path(path)
    if not token_path.is_file():
        raise FileNotFoundError(f"Hugging Face token file not found: {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("Hugging Face token file must contain one non-empty token")
    return token


class HuggingFaceGammaMapper:
    """Map free-form safety language to a validated CBF gamma value."""

    def __init__(
        self,
        config: HFLLMConfig | None = None,
        *,
        client_factory: Callable[[HFLLMConfig, str], Any] | None = None,
    ) -> None:
        self.config = config or HFLLMConfig()
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(config: HFLLMConfig, token: str) -> Any:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "Install the LLM extra with: python -m pip install -e '.[llm]'"
            ) from exc
        return InferenceClient(
            provider=config.provider,
            api_key=token,
            timeout=config.timeout_seconds,
        )

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
        messages = self._messages(instruction, current_gamma, feedback)
        request_hash = sha256(
            json.dumps(
                {
                    "model": self.config.model,
                    "provider": self.config.provider,
                    "messages": messages,
                    "response_format": GAMMA_SCHEMA,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        cached = self._read_cache(request_hash)
        if cached is not None:
            cached["cache_hit"] = True
            return GammaDecision(**cached)

        requested_at = time()
        started = perf_counter()
        raw_response: str | None = None
        try:
            token = load_hf_token(self.config.token_path)
            client = self._client_factory(self.config, token)
            response = client.chat_completion(
                model=self.config.model,
                messages=messages,
                response_format=GAMMA_SCHEMA,
                max_tokens=64,
                temperature=self.config.temperature,
                seed=self.config.seed,
            )
            raw_response = response.choices[0].message.content
            parsed = json.loads(raw_response)
            gamma_text = str(parsed["gamma_text"])
            if gamma_text not in GAMMA_LEVELS:
                raise ValueError("gamma_text outside calibrated enum")
            gamma = _validate_gamma(float(gamma_text))
            safety_level = int(parsed["safety_level"])
            if safety_level not in range(1, 6):
                raise ValueError("safety_level outside [1, 5]")
            if safety_level != GAMMA_LEVELS[gamma_text]:
                raise ValueError("gamma_text and safety_level are inconsistent")
            explanation = f"LLM mapped instruction to calibrated safety level {safety_level}."
            decision = GammaDecision(
                gamma=gamma,
                safety_level=safety_level,
                explanation=explanation,
                model=self.config.model,
                provider=self.config.provider,
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
                explanation="Bounded safety fallback after unavailable or invalid LLM output.",
                model=self.config.model,
                provider=self.config.provider,
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

    @staticmethod
    def _messages(
        instruction: str, current_gamma: float | None, feedback: bool
    ) -> list[dict[str, str]]:
        context = "initial configuration"
        if feedback:
            context = f"online feedback; current gamma={current_gamma:.6f}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context: {context}\nUser safety instruction: {instruction.strip()}",
            },
        ]

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
        # A transient outage must never become a permanent cached decision.
        if self.config.cache_path is None or decision.fallback_used:
            return
        path = Path(self.config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision.as_dict(), ensure_ascii=False) + "\n")
