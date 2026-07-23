#!/usr/bin/env python3
"""Fig. 2 (ACC draft): seven-arm success/collision summary + freeze
attribution panel, per paper/acc/FIGURE_BRIEF.md.

Panel (a): success and collision counts (of 150) for all seven arms,
grouped horizontal bars, status-colored (good = success, critical =
collision) so identity comes from the row label, not hue.
Panel (b): per-episode rejected-solve share in the hard-constraint LLM
arm (nim_feedback), stripped by outcome, showing the 68.6% (collision)
vs 15.7% (goal) contrast that attributes collisions to freeze streaks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

ARMS = (
    ("Fixed\nstatic", "artifacts/safe_panda_core_scenarios_150"),
    ("Scripted\nfeedback", "artifacts/safe_panda_core_scenarios_150_scripted_feedback"),
    ("Dead-time\nmargin", "artifacts/safe_panda_core_scenarios_150_deadtime_margin"),
    ("Velocity\ntube", "artifacts/safe_panda_core_scenarios_150_velocity_tube"),
    ("Soft\nslack", "artifacts/safe_panda_core_scenarios_150_soft_slack"),
    ("NIM\n(tube)", "artifacts/safe_panda_core_scenarios_150_nim_feedback"),
    ("NIM +\nsoft slack", "artifacts/safe_panda_core_scenarios_150_nim_slack_feedback"),
)

OUTCOME_COLOR = {
    "goal": STATUS_GOOD,
    "safety_timeout": STATUS_WARNING,
    "controller_stall": STATUS_SERIOUS,
    "collision": STATUS_CRITICAL,
}
OUTCOME_ORDER = ("goal", "safety_timeout", "controller_stall", "collision")
OUTCOME_LABEL = {
    "goal": "goal",
    "safety_timeout": "safety timeout",
    "controller_stall": "controller stall",
    "collision": "collision",
}


def load_arm_totals() -> list[dict]:
    rows = []
    for label, path in ARMS:
        summary = json.loads((Path(path) / "benchmark_summary.json").read_text())
        success = sum(f["successes"] for f in summary["families"].values())
        collisions = sum(f["collisions"] for f in summary["families"].values())
        rows.append({"label": label, "success": success, "collisions": collisions})
    return rows


def load_nim_feedback_shares() -> tuple[dict[str, list[float]], dict[str, float]]:
    """Per-episode shares (for the strip) plus the POOLED share per outcome
    group (sum of failures / sum of steps) -- the latter must match the
    draft text's 68.6% / 15.7% exactly, since it is a pooled statistic,
    not a mean of per-episode ratios (which differ under episode-length
    weighting and would silently drift from the reported numbers)."""

    path = Path("artifacts/safe_panda_core_scenarios_150_nim_feedback/episodes.csv")
    per_episode: dict[str, list[float]] = {key: [] for key in OUTCOME_ORDER}
    pooled_failures: dict[str, int] = {key: 0 for key in OUTCOME_ORDER}
    pooled_steps: dict[str, int] = {key: 0 for key in OUTCOME_ORDER}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            outcome = row["outcome"]
            if outcome not in per_episode:
                continue
            steps = int(row["steps"])
            failures = int(row["solver_failures"])
            per_episode[outcome].append(failures / steps if steps else 0.0)
            pooled_failures[outcome] += failures
            pooled_steps[outcome] += steps
    pooled_share = {
        key: (pooled_failures[key] / pooled_steps[key] if pooled_steps[key] else 0.0)
        for key in OUTCOME_ORDER
    }
    return per_episode, pooled_share


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/gamma_sweep_fig/fig_arms")
    args = parser.parse_args()

    arm_rows = load_arm_totals()
    shares, pooled_share = load_nim_feedback_shares()

    fig, (ax_bars, ax_strip) = plt.subplots(
        1, 2, figsize=(7.16, 3.05), facecolor=SURFACE,
        gridspec_kw={"width_ratios": (1.05, 1.0), "wspace": 0.5},
    )
    for axis in (ax_bars, ax_strip):
        axis.set_facecolor(SURFACE)
        for spine in axis.spines.values():
            spine.set_color(TEXT_SECONDARY)
            spine.set_linewidth(0.6)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=7)

    # --- Panel (a): grouped horizontal bars, success (good) vs collisions (critical)
    labels = [row["label"] for row in arm_rows]
    y = np.arange(len(labels))
    bar_h = 0.36
    ax_bars.barh(
        y + bar_h / 2 + 0.03, [row["success"] for row in arm_rows], height=bar_h,
        color=STATUS_GOOD, label="success", zorder=3,
    )
    ax_bars.barh(
        y - bar_h / 2 - 0.03, [row["collisions"] for row in arm_rows], height=bar_h,
        color=STATUS_CRITICAL, label="collisions", zorder=3,
    )
    for index, row in enumerate(arm_rows):
        ax_bars.text(
            row["success"] + 2, index + bar_h / 2 + 0.03, str(row["success"]),
            va="center", fontsize=6.5, color=TEXT_PRIMARY,
        )
        ax_bars.text(
            row["collisions"] + 2, index - bar_h / 2 - 0.03, str(row["collisions"]),
            va="center", fontsize=6.5, color=TEXT_PRIMARY,
        )
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels(labels, fontsize=7, color=TEXT_PRIMARY)
    ax_bars.set_xlabel("episodes (of 150)", fontsize=8, color=TEXT_PRIMARY)
    ax_bars.set_xlim(0, 150)
    ax_bars.set_title("(a) Success vs. collisions, all arms", fontsize=8.5,
                      color=TEXT_PRIMARY, pad=6)
    ax_bars.invert_yaxis()
    ax_bars.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    ax_bars.legend(
        loc="lower right", fontsize=6.5, frameon=False, labelcolor=TEXT_PRIMARY,
        handlelength=1.2, handletextpad=0.5,
    )

    # --- Panel (b): rejected-solve share per episode, stripped by outcome
    rng = np.random.default_rng(20260720)
    for row_index, outcome in enumerate(OUTCOME_ORDER):
        values = shares[outcome]
        if not values:
            continue
        jitter = rng.uniform(-0.28, 0.28, size=len(values))
        ax_strip.scatter(
            values, np.full(len(values), row_index) + jitter,
            s=11, color=OUTCOME_COLOR[outcome], alpha=0.75, linewidths=0,
            zorder=3,
        )
        pooled = pooled_share[outcome]
        ax_strip.plot(
            [pooled, pooled], [row_index - 0.34, row_index + 0.34],
            color=TEXT_PRIMARY, linewidth=1.6, zorder=4,
        )
        ax_strip.annotate(
            f"{pooled:.1%}", (pooled, row_index - 0.46),
            fontsize=6.5, ha="center", color=TEXT_PRIMARY,
        )
    ax_strip.axvline(0.0, color=TEXT_SECONDARY, linewidth=0.6)
    ax_strip.set_yticks(range(len(OUTCOME_ORDER)))
    ax_strip.set_yticklabels(
        [OUTCOME_LABEL[key] for key in OUTCOME_ORDER], fontsize=7, color=TEXT_PRIMARY
    )
    ax_strip.set_xlabel("rejected-solve share of episode", fontsize=8, color=TEXT_PRIMARY)
    ax_strip.set_xlim(-0.03, 1.0)
    ax_strip.xaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax_strip.set_title(
        "(b) NIM/tube arm: rejected solves by outcome", fontsize=8.5,
        color=TEXT_PRIMARY, pad=6,
    )
    ax_strip.invert_yaxis()
    ax_strip.grid(True, axis="x", alpha=0.2, linewidth=0.5)

    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.12, right=0.98)
    for suffix, dpi in ((".png", 220), (".pdf", None)):
        fig.savefig(args.out + suffix, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    print("written:", args.out + ".png / .pdf")


if __name__ == "__main__":
    main()
