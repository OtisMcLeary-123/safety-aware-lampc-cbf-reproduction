"""Trusted expansion of validated TP/OD output into manipulation macros.

The LLM never emits simulator calls or motion code.  This module accepts only
validated DSL dataclasses and recognizes the approved E1 grammar:

``move(source) -> close -> move(destination) -> open``.

Each recognized group becomes one deterministic pick/place macro.  Runtime
code remains responsible for the physical approach, grasp, lift, transport,
placement, release, and retreat stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .language_dsl import OptimizationSpec, SafeNarrateResult, TargetSpec


@dataclass(frozen=True, slots=True)
class TrustedPickPlaceMacro:
    source: TargetSpec
    destination: TargetSpec
    pick_optimization: OptimizationSpec
    place_optimization: OptimizationSpec


def build_trusted_pick_place_macros(
    result: SafeNarrateResult,
    *,
    required_hazards: Sequence[str] = (),
) -> tuple[TrustedPickPlaceMacro, ...]:
    """Validate and expand the approved four-step manipulation grammar."""

    steps = result.task_plan.steps
    specs = result.optimization_specs
    if len(steps) != len(specs) or not steps or len(steps) % 4:
        raise ValueError("trusted executor requires complete four-step macros")

    hazards = set(required_hazards)
    macros: list[TrustedPickPlaceMacro] = []
    for offset in range(0, len(steps), 4):
        move_source, close, move_destination, open_gripper = steps[offset : offset + 4]
        pick_spec, close_spec, place_spec, open_spec = specs[offset : offset + 4]

        if tuple(step.action for step in steps[offset : offset + 4]) != (
            "move",
            "close_gripper",
            "move",
            "open_gripper",
        ):
            raise ValueError("trusted executor received an unsupported action sequence")
        if (
            move_source.target is None
            or move_destination.target is None
            or move_source.target.kind != "object"
            or move_destination.target.kind != "object"
        ):
            raise ValueError("pick/place macro endpoints must reference scene objects")
        if move_source.target.relation != "center":
            raise ValueError("pick source must use the center relation")
        if move_destination.target.relation not in {"center", "above"}:
            raise ValueError("place destination must use center or above relation")
        if move_source.target.object_name == move_destination.target.object_name:
            raise ValueError("pick source and place destination must be different")
        if pick_spec is None or place_spec is None or close_spec is not None or open_spec is not None:
            raise ValueError("optimization specs do not align with move actions")
        if pick_spec.objective.target != move_source.target:
            raise ValueError("pick optimization target does not match the task plan")
        if place_spec.objective.target != move_destination.target:
            raise ValueError("place optimization target does not match the task plan")
        if not hazards.issubset(move_source.avoid) or not hazards.issubset(
            move_destination.avoid
        ):
            raise ValueError("trusted executor move omits a required hazard")

        macros.append(
            TrustedPickPlaceMacro(
                move_source.target,
                move_destination.target,
                pick_spec,
                place_spec,
            )
        )
    return tuple(macros)
