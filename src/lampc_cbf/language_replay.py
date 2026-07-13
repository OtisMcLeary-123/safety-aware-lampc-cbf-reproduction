"""Evidence-preserving replay of previously accepted TP/OD outputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .language_dsl import (
    ODAttemptAudit,
    SafeNarrateResult,
    SceneObject,
    parse_optimization_spec,
    parse_task_plan,
)


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    model: str
    provider: str = "recorded_replay"


def _target_payload(target: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if target is None:
        return None
    return {
        "kind": target["kind"],
        "object": target["object_name"],
        "relation": target["relation"],
        "vector_m": target["vector_m"],
    }


class RecordedSafeNarratePlanner:
    """Revalidate and replay accepted language outputs without network access."""

    execution_source = "recorded_replay"

    def __init__(self, metrics_path: str | Path) -> None:
        path = Path(metrics_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        language = payload.get("language_result")
        if not isinstance(language, Mapping):
            raise ValueError("metrics do not contain a language_result object")
        self._language = language
        self.source_metrics_path = str(path)
        self.config = ReplayConfig(str(language["model"]))

    def formulate(
        self,
        instruction: str,
        scene_objects: Sequence[SceneObject],
        *,
        current_position: Sequence[float],
        required_hazards: Sequence[str] = (),
    ) -> SafeNarrateResult:
        expected_hash = sha256(instruction.strip().encode("utf-8")).hexdigest()
        if self._language.get("instruction_hash") != expected_hash:
            raise ValueError("replay instruction hash does not match the current request")

        recorded_plan = self._language["task_plan"]
        plan_payload = {
            "version": recorded_plan["version"],
            "steps": [
                {
                    "action": step["action"],
                    "target": _target_payload(step["target"]),
                    "avoid": step["avoid"],
                }
                for step in recorded_plan["steps"]
            ],
        }
        plan = parse_task_plan(plan_payload, scene_objects)
        hazards = set(required_hazards)
        for index, step in enumerate(plan.steps):
            if step.action == "move" and not hazards.issubset(step.avoid):
                raise ValueError(f"replayed task-plan step {index} omits a required hazard")

        attempts_by_step: dict[int, Mapping[str, Any]] = {}
        for attempt in self._language.get("od_attempts", []):
            step_index = int(attempt["task_step_index"])
            if attempt["status"] != "accepted" or step_index in attempts_by_step:
                raise ValueError("replay requires unique accepted OD attempts")
            attempts_by_step[step_index] = attempt

        specs = []
        audits = []
        for step_index, step in enumerate(plan.steps):
            if step.action != "move":
                specs.append(None)
                continue
            attempt = attempts_by_step.get(step_index)
            if attempt is None or not isinstance(attempt.get("raw_response"), str):
                raise ValueError("replay is missing an accepted raw OD response")
            raw = str(attempt["raw_response"])
            specs.append(
                parse_optimization_spec(
                    raw,
                    scene_objects,
                    current_position=current_position,
                    required_avoid=step.avoid,
                )
            )
            audits.append(
                ODAttemptAudit(
                    task_step_index=step_index,
                    status="accepted",
                    latency_seconds=float(attempt["latency_seconds"]),
                    raw_response=raw,
                    cause_type=None,
                    cause_message=None,
                )
            )
        if set(attempts_by_step) != {audit.task_step_index for audit in audits}:
            raise ValueError("replay contains OD attempts for non-move steps")

        return SafeNarrateResult(
            task_plan=plan,
            optimization_specs=tuple(specs),
            tp_latency_seconds=float(self._language["tp_latency_seconds"]),
            od_latency_seconds=float(self._language["od_latency_seconds"]),
            od_fallbacks=0,
            model=str(self._language["model"]),
            provider=f"recorded_replay:{self._language['provider']}",
            instruction_hash=expected_hash,
            od_attempts=tuple(audits),
        )
