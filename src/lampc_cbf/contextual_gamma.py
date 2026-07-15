"""Context-rich Optimization Formulator gamma mapping extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import hypot, isfinite
from time import perf_counter, time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .hf_llm import GammaDecision
from .language_alignment import paper_safety_label
from .nvidia_nim_gamma import NVIDIA_NIM_ENDPOINT, load_nvidia_token


CONTEXTUAL_GAMMA_SYSTEM_PROMPT = """You are the Optimization Formulator for a
robot MPC-CBF controller. Select one continuous gamma in [0.001, 0.15]. Smaller
gamma means earlier, stronger obstacle avoidance. Larger gamma means later,
weaker avoidance. Do not copy the current gamma unless the language and hazard
evidence independently justify it.

Use these controller-scale bands:
- imminent collision or maximum caution: 0.001 to 0.03
- explicitly cautious: greater than 0.03 to 0.06
- reasonable/appropriate clearance: greater than 0.06 to 0.08
- minimal required safety: greater than 0.08 to less than 0.15
- explicitly ignore obstacles or accept collision risk: exactly 0.15

Hazard evidence may only make the result more cautious (smaller gamma), never
less cautious. TTC <= 0.5 seconds, non-positive clearance, or a negative CBF
residual requires gamma <= 0.03. Return exactly one JSON object with numeric
gamma and its paper Table-2 safety_level: gamma <= 0.06 maps to 1; gamma <= 0.08
maps to 2; gamma < 0.15 maps to 3; gamma == 0.15 maps to 4. Do not include
markdown, reasoning, or additional keys.

When clearance > 0.10 m, TTC > 2 seconds, and the CBF residual is non-negative,
the hazard context is neutral and language intent controls the result. Use these
few-shot semantic anchors:
- "ignore obstacles" or "do not care if it hits" -> exactly 0.15
- "barely enough safety" or "minimal distance" -> 0.12
- "reasonable safety" or "appropriate clearance" -> 0.07
- "very cautiously" -> 0.03
- "greatest possible distance" or "maximally cautious" -> 0.001
Apply the closest semantic anchor, then reduce gamma only if hazard evidence is
more urgent."""

CONTEXTUAL_OUTPUT_CONTRACT = (
    'Return exactly {"gamma": number, "safety_level": integer}. '
    "gamma must be in [0.001, 0.15]."
)


@dataclass(frozen=True, slots=True)
class FeedbackHazardContext:
    current_gamma: float
    obstacle_distance_m: float
    combined_radius_m: float
    clearance_m: float
    predicted_ttc_s: float | None
    obstacle_speed_mps: float
    minimum_cbf_residual: float | None
    intervention_time_s: float

    def __post_init__(self) -> None:
        values = (
            self.current_gamma,
            self.obstacle_distance_m,
            self.combined_radius_m,
            self.clearance_m,
            self.obstacle_speed_mps,
            self.intervention_time_s,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("feedback context values must be finite")
        if not 0.0 < self.current_gamma <= 0.15:
            raise ValueError("current_gamma must be in (0, 0.15]")
        if self.obstacle_distance_m < 0.0 or self.combined_radius_m <= 0.0:
            raise ValueError("obstacle distance must be non-negative and radius positive")
        if self.obstacle_speed_mps < 0.0 or self.intervention_time_s < 0.0:
            raise ValueError("speed and intervention time must be non-negative")
        if self.predicted_ttc_s is not None and (
            not isfinite(self.predicted_ttc_s) or self.predicted_ttc_s < 0.0
        ):
            raise ValueError("predicted TTC must be finite and non-negative")
        if self.minimum_cbf_residual is not None and not isfinite(
            self.minimum_cbf_residual
        ):
            raise ValueError("CBF residual must be finite")

    @classmethod
    def neutral(cls, *, current_gamma: float = 0.15) -> "FeedbackHazardContext":
        return cls(
            current_gamma=current_gamma,
            obstacle_distance_m=0.50,
            combined_radius_m=0.135,
            clearance_m=0.365,
            predicted_ttc_s=5.0,
            obstacle_speed_mps=0.0,
            minimum_cbf_residual=0.10,
            intervention_time_s=0.0,
        )

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "current_formulation": {
                "cbf_inequality": "h_next >= (1-gamma) * h_current",
                "current_gamma": self.current_gamma,
                "combined_collision_radius_m": self.combined_radius_m,
            },
            "hazard_state": {
                "obstacle_distance_m": self.obstacle_distance_m,
                "clearance_m": self.clearance_m,
                "predicted_ttc_s": self.predicted_ttc_s,
                "obstacle_speed_mps": self.obstacle_speed_mps,
                "minimum_cbf_residual": self.minimum_cbf_residual,
                "intervention_time_s": self.intervention_time_s,
            },
        }


def feedback_context_from_condition(
    condition: Mapping[str, Any],
    *,
    current_gamma: float,
    forward_offset_m: float,
    combined_radius_m: float,
    reference_speed_mps: float,
) -> FeedbackHazardContext:
    """Estimate intervention-time hazard context before provider collection."""

    intervention = float(condition["intervention_time"])
    obstacle_speed = float(condition["obstacle_speed"])
    lateral = float(condition["lateral_offset"])
    robot_forward = reference_speed_mps * intervention
    obstacle_forward = forward_offset_m - obstacle_speed * intervention
    forward_separation = obstacle_forward - robot_forward
    distance = hypot(lateral, forward_separation)
    clearance = distance - combined_radius_m
    closing_speed = obstacle_speed + reference_speed_mps
    ttc = max(0.0, clearance) / closing_speed if closing_speed > 0.0 else None
    return FeedbackHazardContext(
        current_gamma=current_gamma,
        obstacle_distance_m=distance,
        combined_radius_m=combined_radius_m,
        clearance_m=clearance,
        predicted_ttc_s=ttc,
        obstacle_speed_mps=obstacle_speed,
        minimum_cbf_residual=None,
        intervention_time_s=intervention,
    )


@dataclass(frozen=True, slots=True)
class ContextualGammaConfig:
    model: str = "meta/llama-3.1-8b-instruct"
    endpoint: str = NVIDIA_NIM_ENDPOINT
    token_path: str = "nvidiatoken.txt"
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    seed: int = 11
    max_tokens: int = 128


class ContextualNvidiaNIMGammaMapper:
    def __init__(
        self,
        config: ContextualGammaConfig | None = None,
        *,
        transport: Callable[[Request, float], bytes] | None = None,
    ) -> None:
        self.config = config or ContextualGammaConfig()
        self._transport = transport or self._default_transport

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()

    def infer_gamma(
        self, instruction: str, context: FeedbackHazardContext
    ) -> GammaDecision:
        if not instruction.strip():
            raise ValueError("instruction must be non-empty")
        user_payload = context.prompt_payload() | {
            "user_safety_instruction": instruction.strip()
        }
        messages = [
            {"role": "system", "content": CONTEXTUAL_GAMMA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{CONTEXTUAL_OUTPUT_CONTRACT}\n"
                    + json.dumps(user_payload, sort_keys=True)
                ),
            },
        ]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_hash = sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
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
            if set(parsed) != {"gamma", "safety_level"}:
                raise ValueError("contextual gamma output has unexpected keys")
            gamma = float(parsed["gamma"])
            if not isfinite(gamma) or not 0.001 <= gamma <= 0.15:
                raise ValueError("contextual gamma must be in [0.001, 0.15]")
            safety_level = int(parsed["safety_level"])
            if safety_level != paper_safety_label(gamma):
                raise ValueError("contextual gamma and paper safety level disagree")
            return GammaDecision(
                gamma=gamma,
                safety_level=safety_level,
                explanation="Contextual OF mapped language and hazard state to gamma.",
                model=self.config.model,
                provider="nvidia-nim",
                latency_seconds=perf_counter() - started,
                requested_at_unix=requested_at,
                prompt_hash=sha256(
                    CONTEXTUAL_GAMMA_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                request_hash=request_hash,
                raw_response=raw_response,
                fallback_used=False,
                cache_hit=False,
                error_type=None,
            )
        except Exception as error:
            error_type = type(error).__name__
            if isinstance(error, HTTPError):
                error_type = f"HTTPError:{error.code}"
            return GammaDecision(
                gamma=0.05,
                safety_level=paper_safety_label(0.05),
                explanation="Bounded fallback after contextual OF failure.",
                model=self.config.model,
                provider="nvidia-nim",
                latency_seconds=perf_counter() - started,
                requested_at_unix=requested_at,
                prompt_hash=sha256(
                    CONTEXTUAL_GAMMA_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                request_hash=request_hash,
                raw_response=raw_response,
                fallback_used=True,
                cache_hit=False,
                error_type=error_type,
            )
