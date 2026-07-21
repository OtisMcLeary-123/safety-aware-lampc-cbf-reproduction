#!/usr/bin/env python3
"""EXPERIMENTAL comparison: the paper's literal code-execution OF (their
Fig. 3 pattern, Code-as-Policies style) vs this repository's production
structured-payload path (registry 3.2).

This script is a research artifact, never a benchmark arm. It sends a
fixed battery of prompts (benign safety-intent phrasings PLUS every known
attack payload in ``code_as_policies.ATTACK_PAYLOADS``, injected as a
simulated "jailbroken" LLM response so the harness is tested even without
live API spend) through the sandbox and reports: parse rate, AST-gate
rejection rate, execution-gate rejection rate, and — for the benign
cases — whether the produced CasADi expression matches the intended
semantics. It never runs on the real robot loop and never modifies the
default `trusted_executor` path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from lampc_cbf.code_as_policies import (
    ATTACK_PAYLOADS,
    BENIGN_PAYLOAD,
    execute_casadi_snippet,
)


def run_battery() -> dict:
    import casadi as ca
    import numpy as np

    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 2)
    rows = []
    for name, payload in (*ATTACK_PAYLOADS, ("benign", BENIGN_PAYLOAD)):
        result = execute_casadi_snippet(payload, ca=ca, np=np, x=x, u=u)
        rows.append(
            {
                "case": name,
                "is_attack": name != "benign",
                "accepted": result.accepted,
                "rejection_reason": result.rejection_reason,
                "correctly_handled": (
                    (not result.accepted) if name != "benign" else result.accepted
                ),
            }
        )
    attacks_caught = sum(r["correctly_handled"] for r in rows if r["is_attack"])
    total_attacks = sum(r["is_attack"] for r in rows)
    return {
        "profile": "code_as_policies_sandbox_ablation",
        "status": "EXPERIMENTAL_NEVER_PRODUCTION",
        "rows": rows,
        "attacks_caught": attacks_caught,
        "total_attacks": total_attacks,
        "attack_catch_rate": attacks_caught / total_attacks if total_attacks else None,
        "benign_accepted": next(r["accepted"] for r in rows if r["case"] == "benign"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="artifacts/code_as_policies_ablation/summary.json"
    )
    args = parser.parse_args()
    summary = run_battery()
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
