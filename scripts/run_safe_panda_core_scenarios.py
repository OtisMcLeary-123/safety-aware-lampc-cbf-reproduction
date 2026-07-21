#!/usr/bin/env python3
"""Run frozen core-scenario episodes with per-row checkpointing."""

import argparse
import csv
import json
from pathlib import Path

from lampc_cbf.core_scenarios import (
    INSTANCES_PATH,
    PLAN_PATH,
    _normalize_csv_row,
    load_frozen_instances,
    paired_comparison,
    pilot_episode_ids,
    run_core_benchmark,
    smoke_episode_ids,
)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [_normalize_csv_row(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "pilot", "full"),
        default="full",
        help="smoke = 1 median episode/family, pilot = 5/family, full = 150",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/safe_panda_core_scenarios_150"
    )
    parser.add_argument("--instances", default=str(INSTANCES_PATH))
    parser.add_argument("--plan", default=str(PLAN_PATH))
    parser.add_argument(
        "--feedback-manifest",
        help="scripted-feedback profile JSON; switches the run to that method",
    )
    parser.add_argument(
        "--controller-profile",
        help="versioned remedy-profile JSON with runner kwargs overrides",
    )
    parser.add_argument(
        "--prediction-feedback-manifest",
        help="channel-2 scripted prediction-mode-switch manifest; "
        "switches to that method",
    )
    parser.add_argument(
        "--llm-feedback-manifest",
        help="predeclared NIM LLM feedback manifest; switches to that method",
    )
    parser.add_argument(
        "--families",
        help="comma-separated family prefixes to run (e.g. CS2); a runtime "
        "filter on the frozen instances, other families stay untouched",
    )
    parser.add_argument(
        "--paired-against",
        help="baseline output dir with episodes.csv for paired McNemar/Holm",
    )
    args = parser.parse_args()

    _, instances = load_frozen_instances(args.instances)
    episode_ids = None
    if args.stage == "smoke":
        episode_ids = smoke_episode_ids(instances)
    elif args.stage == "pilot":
        episode_ids = pilot_episode_ids(instances)
    if args.families:
        prefixes = tuple(
            prefix.strip() for prefix in args.families.split(",") if prefix.strip()
        )
        pool = episode_ids or [instance.episode_id for instance in instances]
        episode_ids = [
            episode_id
            for episode_id in pool
            if episode_id.startswith(prefixes)
        ]
        if not episode_ids:
            raise SystemExit(f"no episodes match families: {args.families}")
    summary = run_core_benchmark(
        instances_path=args.instances,
        plan_path=args.plan,
        output_dir=args.output_dir,
        episode_ids=episode_ids,
        scripted_feedback_manifest=args.feedback_manifest,
        controller_profile_manifest=args.controller_profile,
        llm_feedback_manifest=args.llm_feedback_manifest,
        prediction_feedback_manifest=args.prediction_feedback_manifest,
    )
    if args.paired_against:
        baseline_rows = _read_rows(Path(args.paired_against) / "episodes.csv")
        method_rows = _read_rows(Path(args.output_dir) / "episodes.csv")
        method_ids = {row["episode_id"] for row in method_rows}
        baseline_rows = [
            row for row in baseline_rows if row["episode_id"] in method_ids
        ]
        paired = paired_comparison(baseline_rows, method_rows)
        paired_path = Path(args.output_dir) / "paired_summary.json"
        paired_path.write_text(json.dumps(paired, indent=2), encoding="utf-8")
        summary["paired"] = paired
    print(json.dumps(summary, indent=2))
