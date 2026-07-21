#!/usr/bin/env python3
"""Paraphrase-robustness of the GPT-4o safety-parameter mapping (ReWiND [56] style).

Reference [56] (ReWiND, arXiv:2505.10911) evaluates input robustness as the
average Spearman rank correlation and the output variance across semantically
similar instructions. This script applies that metric to the LaMPC-CBF language
-> gamma mapping, which neither the source paper's Table 3 nor our earlier
alignment runs tested (they used one fixed phrasing per query).

Two blocks, both under a fixed 3-anchor few-shot context (only phrasing varies):

  queries  : each of the 8 published Table-1 queries plus hand-authored
             intent-preserving paraphrases. Reports (a) within-query gamma std
             across paraphrases (stability), and (b) per-variant Spearman rho of
             the 8 predicted gammas against the paper's reference gammas
             (ordering robustness), averaged over variants.
  feedback : the benchmark's fixed feedback utterance ("Watch out! ...", which
             the 150-episode runs map deterministically to gamma 0.05) plus
             paraphrases. Reports gamma mean/std/min/max -- does the feedback
             gamma hold regardless of wording?

One request per phrasing; checkpoint after each. GPT-4o via GitHub Models; token
read locally, never printed.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter, sleep, time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lampc_cbf.language_alignment import (
    PUBLISHED_ALIGNMENT_CASES,
    pearson_correlation,
    spearman_correlation,
    validate_alignment_gamma,
)

ENDPOINT = "https://models.github.ai/inference/chat/completions"

BASE_SYSTEM = (
    "You are the Optimization Formulator safety-parameter module for a robot "
    "controlled by MPC with a discrete control barrier function (CBF). Map the "
    "user's safety intent to a continuous gamma in (0, 1]. Smaller gamma means "
    "earlier, more proactive avoidance and larger obstacle clearance; gamma = 1 "
    "reduces the CBF to a plain distance constraint (no proactive avoidance). "
    "Return only the requested JSON object. Never generate executable code."
)
OUTPUT_CONTRACT = (
    'Output contract: return exactly one JSON object shaped as {"gamma": number}. '
    "Do not include markdown, analysis, reasoning tags, or additional keys."
)
ANCHOR_EXAMPLES = (
    ("Approach the target with maximum caution, avoiding the obstacle as early "
     "as possible.", 0.02),
    ("Reach the target while keeping a moderate, sensible distance from the "
     "obstacle.", 0.1),
    ("Go straight to the target; do not worry about the obstacle.", 0.95),
)

# Hand-authored, intent-preserving paraphrases keyed by published case id.
QUERY_PARAPHRASES = {
    1: ["Take the gripper straight to the red cube and pay no attention to obstacles.",
        "Reach the red cube; obstacles are irrelevant."],
    2: ["Move to the red cube; a collision with an obstacle is totally fine.",
        "Get to the red cube even if it bumps the obstacle."],
    3: ["Move to the red cube with only the minimum safety needed.",
        "Reach the red cube, just barely satisfying safety."],
    4: ["Move to the red cube while staying only a small distance from the obstacle.",
        "Reach the red cube, keeping just a little clearance from the obstacle."],
    5: ["Move to the red cube with a reasonable amount of safety.",
        "Reach the red cube while being reasonably safe."],
    6: ["Move to the red cube keeping suitable clearance.",
        "Reach the red cube with an appropriate safety margin."],
    7: ["Move to the red cube with great caution.",
        "Reach the red cube extremely carefully, prioritizing safety."],
    8: ["Move to the red cube while staying as far from the obstacle as possible.",
        "Reach the red cube keeping maximum distance from the obstacle."],
}

FEEDBACK_UTTERANCE = "Watch out! I think it's going to crash soon."
FEEDBACK_PARAPHRASES = [
    "Careful! It looks like it's about to collide.",
    "Heads up, a crash seems imminent.",
    "Be careful, I think we're about to hit it.",
    "Look out - it's going to crash any moment.",
    "Slow down, the obstacle is closing in fast.",
]


def build_system(examples):
    lines = "\n".join(f'- "{q}" -> gamma {g}' for q, g in examples)
    return f"{BASE_SYSTEM}\n\nCalibration examples:\n{lines}"


def query_gamma(instruction, system, *, model, token, timeout, max_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": f"{OUTPUT_CONTRACT}\nRobot-motion instruction: {instruction.strip()}"},
    ]
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens, "seed": 11, "stream": False,
               "response_format": {"type": "json_object"}}
    request = Request(ENDPOINT, data=json.dumps(payload).encode("utf-8"),
                      headers={"Authorization": f"Bearer {token}",
                               "Accept": "application/json",
                               "Content-Type": "application/json"}, method="POST")
    started = perf_counter()
    for attempt in range(6):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            wait = float(exc.headers.get("Retry-After") or 0) or (15 * (attempt + 1))
            print(f"    429 rate-limited; waiting {wait:.0f}s (attempt {attempt + 1})")
            sleep(wait)
    raw = body["choices"][0]["message"]["content"]
    gamma = validate_alignment_gamma(json.loads(raw)["gamma"])
    return gamma, perf_counter() - started, raw


def load_token(path):
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("token file must contain exactly one non-empty token")
    return token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--token-file", default="githubtk.txt")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--inter-request-delay", type=float, default=1.5)
    parser.add_argument("--output-dir",
                        default="artifacts/language_alignment_paraphrase_github_models")
    args = parser.parse_args()

    token = load_token(args.token_file)
    system = build_system(ANCHOR_EXAMPLES)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.json"
    records = []
    done = {}
    if checkpoint_path.is_file():
        records = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        done = {r["tag"]: r["gamma"] for r in records}
        print(f"(resume: {len(done)} cached results)")

    def call(tag, text):
        if tag in done:
            print(f"  {tag}: gamma={done[tag]} (cached)")
            return done[tag]
        gamma, latency, raw = query_gamma(text, system, model=args.model, token=token,
                                          timeout=args.timeout_seconds,
                                          max_tokens=args.max_tokens)
        records.append({"tag": tag, "instruction": text, "gamma": gamma,
                        "latency": latency, "raw": raw})
        done[tag] = gamma
        checkpoint_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"  {tag}: gamma={gamma}")
        sleep(args.inter_request_delay)
        return gamma

    # --- Block 1: query paraphrases ---
    print("=== queries (original + 2 paraphrases each, fixed anchor3 few-shot) ===")
    ref = [c.paper_gamma for c in PUBLISHED_ALIGNMENT_CASES]
    per_query = {}          # case_id -> [gamma over variants]
    per_variant = []        # list of [gamma over 8 queries] per variant slot
    variant_count = 1 + max(len(v) for v in QUERY_PARAPHRASES.values())
    for slot in range(variant_count):
        per_variant.append([])
    for case in PUBLISHED_ALIGNMENT_CASES:
        variants = [case.instruction] + QUERY_PARAPHRASES[case.case_id]
        gammas = []
        for slot, text in enumerate(variants):
            g = call(f"q{case.case_id}.v{slot}", text)
            gammas.append(g)
            per_variant[slot].append(g)
        per_query[case.case_id] = gammas

    within_std = {cid: pstdev(gs) for cid, gs in per_query.items()}
    variant_spearman = [spearman_correlation(v, ref) for v in per_variant]
    variant_pearson = [pearson_correlation(v, ref) for v in per_variant]

    # --- Block 2: feedback utterance paraphrases ---
    print("=== feedback utterance (benchmark original + paraphrases) ===")
    fb = [call("fb.orig", FEEDBACK_UTTERANCE)]
    for i, p in enumerate(FEEDBACK_PARAPHRASES):
        fb.append(call(f"fb.p{i}", p))

    summary = {
        "model": args.model, "provider": "github-models",
        "few_shot": "fixed_anchor3",
        "queries": {
            "within_query_gamma_std_mean": mean(within_std.values()),
            "within_query_gamma_std_max": max(within_std.values()),
            "within_query_gamma_std_per_case": within_std,
            "per_variant_spearman_rho": variant_spearman,
            "avg_variant_spearman_rho": mean(variant_spearman),
            "std_variant_spearman_rho": pstdev(variant_spearman),
            "avg_variant_pearson_r": mean(variant_pearson),
        },
        "feedback": {
            "utterance": FEEDBACK_UTTERANCE, "benchmark_gamma": 0.05,
            "gammas": fb, "mean": mean(fb), "std": pstdev(fb),
            "min": min(fb), "max": max(fb),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    q = summary["queries"]
    f = summary["feedback"]
    print("\n=== Paraphrase-robustness (ReWiND [56] style) ===")
    print(f"queries: within-query gamma std  mean={q['within_query_gamma_std_mean']:.4f} "
          f"max={q['within_query_gamma_std_max']:.4f}")
    print(f"queries: avg per-variant Spearman rho={q['avg_variant_spearman_rho']:.3f} "
          f"(std {q['std_variant_spearman_rho']:.3f}) over {len(variant_spearman)} slots")
    print(f"feedback: gamma mean={f['mean']:.4f} std={f['std']:.4f} "
          f"min={f['min']} max={f['max']} (benchmark used 0.05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
