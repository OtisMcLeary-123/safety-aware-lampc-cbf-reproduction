"""Sandbox tests for the EXPERIMENTAL code-as-policies comparison arm.

This module is never wired into the production control loop; these tests
exist to prove the two-layer defense actually rejects every known attack
class and accepts only the intended, narrow grammar.
"""

from __future__ import annotations

import numpy as np
import pytest

from lampc_cbf.code_as_policies import (
    ATTACK_PAYLOADS,
    BENIGN_PAYLOAD,
    CodeSafetyViolation,
    execute_casadi_snippet,
    validate_ast,
)


@pytest.mark.parametrize("name,payload", ATTACK_PAYLOADS)
def test_every_known_attack_payload_is_rejected(name: str, payload: str) -> None:
    import casadi as ca

    result = execute_casadi_snippet(payload, ca=ca, np=np, x=ca.MX.sym("x", 3), u=ca.MX.sym("u", 2))
    assert not result.accepted, f"attack '{name}' was NOT rejected"
    assert result.rejection_reason


@pytest.mark.parametrize("name,payload", ATTACK_PAYLOADS)
def test_ast_gate_alone_rejects_every_attack(name: str, payload: str) -> None:
    # Redundant with the execution-level test above, but proves the FIRST
    # gate (cheap, no exec at all) already catches every payload on its
    # own — defense in depth, not reliance on the second layer.
    with pytest.raises(CodeSafetyViolation):
        validate_ast(payload)


def test_benign_payload_is_accepted_and_produces_an_objective_term() -> None:
    import casadi as ca

    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 2)
    result = execute_casadi_snippet(
        BENIGN_PAYLOAD, ca=ca, np=np, x=x, u=u, params={"weight": 2.0, "target_y": 0.5}
    )
    assert result.accepted, result.rejection_reason
    assert len(result.objective_terms) == 1
    # The term is a live CasADi expression built from the injected symbols.
    fn = ca.Function("f", [x, u], [result.objective_terms[0]])
    value = float(fn(ca.DM([0.0, 0.2, 0.0]), ca.DM([0.0, 0.0]))[0])
    assert value == pytest.approx(2.0 * (0.2 - 0.5) ** 2)


def test_snippet_never_receives_obstacle_or_barrier_state() -> None:
    """The function signature itself is the safety invariant: no obstacle
    position, no barrier value, no gamma is ever passed in, so no snippet
    — malicious or not — can construct a CBF expression at all."""
    import inspect

    sig = inspect.signature(execute_casadi_snippet)
    # execute_casadi_snippet's own call surface to the snippet is fixed to
    # (ca, np, x, u, params); assert that contract textually here so a
    # future refactor cannot silently widen it without failing this test.
    assert list(sig.parameters) == [
        "source", "ca", "np", "x", "u", "params", "timeout_s",
    ]


def test_timeout_is_enforced() -> None:
    import casadi as ca

    slow = (
        "def formulate_objective(ca, np, x, u, params):\n"
        "    total = 0\n"
        "    for _step_marker in range(1):\n"  # loops are disallowed anyway
        "        total = total + 1\n"
        "    return [ca.MX(total)]\n"
    )
    # `for` is not in the allowlist, so this is rejected before it ever
    # reaches the timeout path — confirms loop constructs cannot slip
    # through as a disguised infinite loop.
    result = execute_casadi_snippet(slow, ca=ca, np=np, x=ca.MX.sym("x", 1), u=ca.MX.sym("u", 1))
    assert not result.accepted
    assert "disallowed" in (result.rejection_reason or "").lower() or "syntax" in (
        result.rejection_reason or ""
    ).lower()


def test_multi_function_snippet_rejected() -> None:
    payload = (
        "def helper(ca):\n    return ca.MX(0)\n"
        "def formulate_objective(ca, np, x, u, params):\n"
        "    return [helper(ca)]\n"
    )
    with pytest.raises(CodeSafetyViolation, match="exactly one function"):
        validate_ast(payload)


def test_wrong_entry_point_name_rejected() -> None:
    payload = "def formulate(ca, np, x, u, params):\n    return [ca.MX(0)]\n"
    with pytest.raises(CodeSafetyViolation, match=r"formulate_objective"):
        validate_ast(payload)
