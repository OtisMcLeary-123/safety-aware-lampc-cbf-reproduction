"""Paper-fidelity continuous gamma mapping (Tables 1-2) via NVIDIA NIM.

Implements the registry-3.3 paper-fidelity path: the OF-style feedback call
returns a continuous ``gamma`` in (0, 1], calibrated on the paper's printed
Table 1 examples, and every decision is recorded with its Table 2 label.
LLM output is parsed as a strict JSON payload and validated locally; it is
never executed (registry 3.2). Fail-closed: any transport, schema, or range
error keeps the current gamma and records the error verbatim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable
from urllib.request import Request, urlopen

from .nvidia_nim_gamma import (
    DEFAULT_NVIDIA_NIM_MODEL,
    NVIDIA_NIM_ENDPOINT,
    load_nvidia_token,
)

# Paper Table 1 (verified against the rendered PDF, p. 25560), verbatim.
TABLE1_CALIBRATION = (
    ("Move gripper to red cube, ignoring any obstacles.", 1.0),
    ("Move gripper to red cube. I don't care if it hits an obstacle at all.", 1.0),
    ("Move gripper to red cube, barely enough to meet safety requirements.", 0.14),
    (
        "Move gripper to red cube while keeping a minimal distance from the obstacle.",
        0.09,
    ),
    ("Move gripper to red cube, considering safety to a reasonable degree.", 0.065),
    ("Move gripper to red cube with appropriate clearance.", 0.065),
    ("Move gripper to red cube very cautiously.", 0.03),
    (
        "Move gripper to red cube while maintaining the greatest possible "
        "distance from the obstacle.",
        0.001,
    ),
)

PAPER_CONTINUOUS_SYSTEM_PROMPT = (
    "You are the Optimization Formulator safety-parameter module for a robot "
    "controlled by MPC with a discrete control barrier function (CBF). Map "
    "the user's safety intent to a continuous gamma in (0, 1]. Smaller gamma "
    "means earlier, more proactive avoidance and larger obstacle clearance; "
    "gamma = 1 reduces the CBF to a plain distance constraint (no proactive "
    "avoidance). Return only the requested JSON object. Never generate "
    "executable code.\n\nCalibration examples:\n"
    + "\n".join(f'- "{query}" -> gamma {gamma}' for query, gamma in TABLE1_CALIBRATION)
)

CONTINUOUS_GAMMA_OUTPUT_CONTRACT = (
    'Return exactly one JSON object shaped as {"gamma": 0.05}. gamma must be '
    "a number greater than 0 and at most 1. Do not include markdown, "
    "analysis, or additional keys."
)

CONTINUOUS_GAMMA_SCHEMA = {
    "type": "object",
    "properties": {
        "gamma": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0}
    },
    "required": ["gamma"],
    "additionalProperties": False,
}


def validate_continuous_gamma(gamma: float) -> float:
    if not isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("continuous gamma must be finite and in (0, 1]")
    return float(gamma)


def table2_label(gamma: float) -> int:
    """Paper Table 2 label. The printed table overlaps at 0.15 (label 3
    range (0.08, 0.15] vs label 4 = 0.15); exact 0.15 resolves to 4, the
    printed no-safety-expression default."""

    validate_continuous_gamma(gamma)
    if gamma <= 0.06:
        return 1
    if gamma <= 0.08:
        return 2
    if gamma < 0.15:
        return 3
    if gamma == 0.15:
        return 4
    return 5


@dataclass(frozen=True, slots=True)
class ContinuousGammaDecision:
    gamma: float
    table2_label: int
    model: str
    provider: str
    latency_seconds: float
    requested_at_unix: float
    prompt_hash: str
    request_hash: str
    raw_response: str | None
    cache_hit: bool
    fallback_used: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class NIMContinuousGammaConfig:
    model: str = DEFAULT_NVIDIA_NIM_MODEL
    endpoint: str = NVIDIA_NIM_ENDPOINT
    token_path: str = "nvidiatoken.txt"
    timeout_seconds: float = 10.0
    temperature: float = 0.0
    seed: int = 11
    max_tokens: int = 64
    checkpoint_path: str | None = None
    guided_json_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.endpoint.startswith("https://"):
            raise ValueError("endpoint must use HTTPS")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")


class NIMContinuousGammaMapper:
    """Continuous-gamma OF feedback call with verbatim checkpointing."""

    def __init__(
        self,
        config: NIMContinuousGammaConfig | None = None,
        *,
        transport: Callable[[Request, float], bytes] | None = None,
    ) -> None:
        self.config = config or NIMContinuousGammaConfig()
        self._transport = transport or self._default_transport

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS config
            return response.read()

    def infer(
        self, utterance: str, *, current_gamma: float, task: str
    ) -> ContinuousGammaDecision:
        if not utterance.strip():
            raise ValueError("utterance must be non-empty")
        validate_continuous_gamma(current_gamma)
        user_content = (
            f"Task instruction: {task}\n"
            f"Current CBF safety parameter gamma: {current_gamma}\n"
            f'During execution the user says: "{utterance}"\n'
            "Decide gamma_new for the updated CBF constraint.\n"
            f"{CONTINUOUS_GAMMA_OUTPUT_CONTRACT}"
        )
        messages = [
            {"role": "system", "content": PAPER_CONTINUOUS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "stream": False,
        }
        if self.config.guided_json_enabled:
            payload["nvext"] = {"guided_json": CONTINUOUS_GAMMA_SCHEMA}
        request_hash = sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cached = self._read_checkpoint(request_hash)
        if cached is not None:
            cached["cache_hit"] = True
            return ContinuousGammaDecision(**cached)

        requested_at = time()
        started = perf_counter()
        raw_response: str | None = None
        prompt_hash = sha256(
            PAPER_CONTINUOUS_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
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
            if set(parsed) != {"gamma"}:
                raise ValueError("output does not match the continuous gamma schema")
            gamma = validate_continuous_gamma(float(parsed["gamma"]))
            decision = ContinuousGammaDecision(
                gamma=gamma,
                table2_label=table2_label(gamma),
                model=self.config.model,
                provider="nvidia-nim",
                latency_seconds=perf_counter() - started,
                requested_at_unix=requested_at,
                prompt_hash=prompt_hash,
                request_hash=request_hash,
                raw_response=raw_response,
                cache_hit=False,
                fallback_used=False,
                error=None,
            )
        except Exception as exc:  # fail-closed: keep the current gamma
            decision = ContinuousGammaDecision(
                gamma=float(current_gamma),
                table2_label=table2_label(current_gamma),
                model=self.config.model,
                provider="nvidia-nim",
                latency_seconds=perf_counter() - started,
                requested_at_unix=requested_at,
                prompt_hash=prompt_hash,
                request_hash=request_hash,
                raw_response=raw_response,
                cache_hit=False,
                fallback_used=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._append_checkpoint(decision)
        return decision

    def _read_checkpoint(self, request_hash: str) -> dict[str, Any] | None:
        if self.config.checkpoint_path is None:
            return None
        path = Path(self.config.checkpoint_path)
        if not path.is_file():
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("request_hash") == request_hash:
                record.pop("cache_hit", None)
                return record
        return None

    def _append_checkpoint(self, decision: ContinuousGammaDecision) -> None:
        if self.config.checkpoint_path is None:
            return
        path = Path(self.config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision)) + "\n")
