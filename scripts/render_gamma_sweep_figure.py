#!/usr/bin/env python3
"""Paper-style gamma-sweep figure (Fig.-5 analogue) from saved sweep metrics.

Three panels: (a) top view x-y, (b) 3-D diagonal view, (c) distance to
obstacle over time. Obstacle drawn to scale as translucent spheres along
its path (reduced opacity into the past), red 'X' at any collision point.
Colors: validated categorical slots in fixed gamma order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GAMMAS = (0.001, 0.04, 0.065, 0.15, 1.0)
SERIES = ("#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a")
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, STATUS_SERIOUS = "#fcfcfb", "#e34948"
DT = 0.04


def load_runs(root: Path) -> list[dict]:
    runs = []
    for gamma in GAMMAS:
        payload = json.loads((root / f"g{gamma}" / "metrics.json").read_text())
        cfg = payload["config"]
        runs.append(
            {
                "gamma": gamma,
                "pos": np.asarray(payload["positions"]),
                "obs": np.asarray(payload["true_obstacles"]),
                "clr": np.asarray(payload["true_clearances"]),
                "start": np.asarray(payload["start"]),
                "goal": np.asarray(payload["goal"]),
                "radius": cfg["obstacle_radius"] + cfg["collision_radius"],
                "collision": bool(payload["result"]["collision"]),
            }
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/gamma_sweep_fig")
    parser.add_argument("--out", default="artifacts/gamma_sweep_fig/fig_gamma_sweep")
    parser.add_argument("--feedback-dir", default=None,
                        help="optional metrics dir of a feedback run drawn as a 6th dashed series")
    args = parser.parse_args()
    runs = load_runs(Path(args.root))
    feedback = None
    if args.feedback_dir:
        payload = json.loads((Path(args.feedback_dir) / "metrics.json").read_text())
        feedback = {
            "pos": np.asarray(payload["positions"]),
            "clr": np.asarray(payload["true_clearances"]),
            "reached": bool(payload["result"]["reached_goal"]),
        }
    ref = runs[0]
    start, goal, radius = ref["start"], ref["goal"], ref["radius"]
    obs_path = max((r["obs"] for r in runs), key=len)

    fig = plt.figure(figsize=(11.0, 3.7), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.15, 1.25), wspace=0.32)
    ax_top = fig.add_subplot(grid[0])
    ax_3d = fig.add_subplot(grid[1], projection="3d")
    ax_time = fig.add_subplot(grid[2])
    for axis in (ax_top, ax_time):
        axis.set_facecolor(SURFACE)
        axis.grid(True, alpha=0.22, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color(TEXT_SECONDARY)
            spine.set_linewidth(0.7)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    # (a) top view: obstacle to scale, past positions at reduced opacity
    all_xy = np.vstack([r["pos"][:, :2] for r in runs] + [start[None, :2], goal[None, :2]])
    pad = radius * 1.35
    crop_x = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    crop_y = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    in_crop = (
        (obs_path[:, 0] > crop_x[0] - radius * 0.4)
        & (obs_path[:, 0] < crop_x[1] + radius * 0.4)
        & (obs_path[:, 1] > crop_y[0] - radius * 0.4)
        & (obs_path[:, 1] < crop_y[1] + radius * 0.4)
    )
    visible = obs_path[in_crop]
    snap_indices = np.linspace(0, len(visible) - 1, 6).astype(int)
    for order, index in enumerate(snap_indices):
        age = order / max(1, len(snap_indices) - 1)
        ax_top.add_patch(
            plt.Circle(
                visible[index][:2], radius,
                color="0.55", alpha=0.05 + 0.17 * age, linewidth=0,
            )
        )
    ax_top.plot(visible[:, 0], visible[:, 1], color="0.55", linewidth=1.0,
                linestyle=(0, (2, 2)), alpha=0.8)
    ax_top.annotate(
        "", xy=visible[snap_indices[2]][:2],
        xytext=visible[min(len(visible) - 1, snap_indices[1])][:2],
        arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2),
    )
    for run, color in zip(runs, SERIES):
        ax_top.plot(run["pos"][:, 0], run["pos"][:, 1], color=color,
                    linewidth=2.0, solid_capstyle="round")
        if run["collision"]:
            ax_top.plot(*run["pos"][-1][:2], marker="x", markersize=9,
                        markeredgewidth=2.2, color=STATUS_SERIOUS, zorder=6)
    if feedback is not None:
        ax_top.plot(feedback["pos"][:, 0], feedback["pos"][:, 1],
                    color="#4a3aa7", linewidth=2.2, linestyle=(0, (5, 2.2)), zorder=4)
    ax_top.plot(*start[:2], marker="o", markersize=7, color=TEXT_PRIMARY, zorder=5)
    ax_top.plot(*goal[:2], marker="*", markersize=13, color=TEXT_PRIMARY, zorder=5)
    ax_top.annotate("start", start[:2], textcoords="offset points",
                    xytext=(6, -10), fontsize=8, color=TEXT_SECONDARY)
    ax_top.annotate("goal", goal[:2], textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=TEXT_SECONDARY)
    ax_top.set_xlabel("x [m]", fontsize=9, color=TEXT_PRIMARY)
    ax_top.set_ylabel("y [m]", fontsize=9, color=TEXT_PRIMARY)
    ax_top.set_title("(a) Top view", fontsize=9.5, color=TEXT_PRIMARY)
    ax_top.set_xlim(*crop_x)
    ax_top.set_ylim(*crop_y)
    ax_top.set_aspect("equal", adjustable="box")

    # (b) diagonal 3-D view: wireframe sphere at the closest-approach snapshot
    closest = int(np.argmin(runs[3]["clr"]))
    center = runs[3]["obs"][min(closest, len(runs[3]["obs"]) - 1)]
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:14j]
    ax_3d.plot_surface(
        center[0] + radius*np.cos(u)*np.sin(v),
        center[1] + radius*np.sin(u)*np.sin(v),
        center[2] + radius*np.cos(v),
        color="0.55", alpha=0.12, linewidth=0, zorder=1,
    )
    for run, color in zip(runs, SERIES):
        ax_3d.plot(run["pos"][:, 0], run["pos"][:, 1], run["pos"][:, 2],
                   color=color, linewidth=1.8)
    if feedback is not None:
        ax_3d.plot(feedback["pos"][:, 0], feedback["pos"][:, 1],
                   feedback["pos"][:, 2], color="#4a3aa7", linewidth=2.0,
                   linestyle=(0, (5, 2.2)))
    ax_3d.scatter(*start, color=TEXT_PRIMARY, s=28)
    ax_3d.scatter(*goal, color=TEXT_PRIMARY, marker="*", s=90)
    ax_3d.set_xlabel("x [m]", fontsize=8, color=TEXT_SECONDARY, labelpad=-4)
    ax_3d.set_ylabel("y [m]", fontsize=8, color=TEXT_SECONDARY, labelpad=-4)
    ax_3d.set_zlabel("z [m]", fontsize=8, color=TEXT_SECONDARY, labelpad=-4)
    ax_3d.set_title("(b) Diagonal view", fontsize=9.5, color=TEXT_PRIMARY)
    ax_3d.tick_params(labelsize=7, colors=TEXT_SECONDARY, pad=-2)
    ax_3d.view_init(elev=20, azim=-66)
    ax_3d.set_xlim(*crop_x)
    ax_3d.set_ylim(*crop_y)
    all_z = np.concatenate([r["pos"][:, 2] for r in runs])
    ax_3d.set_zlim(min(all_z.min(), center[2] - radius) - 0.02,
                   max(all_z.max(), center[2] + radius) + 0.02)
    ax_3d.set_box_aspect((0.64, 0.64, 0.40))

    # (c) clearance over time
    for run, color in zip(runs, SERIES):
        time = np.arange(1, len(run["clr"]) + 1) * DT
        ax_time.plot(time, run["clr"], color=color, linewidth=2.0)
        if run["collision"]:
            ax_time.plot(time[-1], run["clr"][-1], marker="x", markersize=8,
                         markeredgewidth=2.0, color=STATUS_SERIOUS, zorder=6)
    if feedback is not None:
        time_fb = np.arange(1, len(feedback["clr"]) + 1) * DT
        ax_time.plot(time_fb, feedback["clr"], color="#4a3aa7", linewidth=2.2,
                     linestyle=(0, (5, 2.2)))
    ax_time.axhline(0.0, color=STATUS_SERIOUS, linewidth=1.0,
                    linestyle=(0, (4, 3)))
    ax_time.annotate("collision boundary", (0.985, 0.045),
                     xycoords="axes fraction", ha="right", fontsize=7.5,
                     color=STATUS_SERIOUS)
    blue = runs[0]
    peak = int(np.argmin(blue["clr"]))
    ax_time.annotate(
        "$\\gamma$ = 0.001", ((peak + 1) * DT - 0.9, blue["clr"][peak] + 0.09),
        fontsize=8, color=SERIES[0],
    )
    ax_time.set_xlabel("time [s]", fontsize=9, color=TEXT_PRIMARY)
    ax_time.set_ylabel("distance to obstacle [m]", fontsize=9, color=TEXT_PRIMARY)
    ax_time.set_title("(c) Distance to obstacle over time", fontsize=9.5,
                      color=TEXT_PRIMARY)

    handles = [
        plt.Line2D([], [], color=color, linewidth=2.4,
                   label=f"$\\gamma$ = {gamma:g}")
        for gamma, color in zip(GAMMAS, SERIES)
    ]
    if feedback is not None:
        handles.append(plt.Line2D(
            [], [], color="#4a3aa7", linewidth=2.4, linestyle=(0, (5, 2.2)),
            label="feedback: $\\gamma$ 0.15$\\rightarrow$0.05",
        ))
    ncols = len(handles)
    fig.legend(handles=handles, loc="upper center", ncol=ncols, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.04),
               labelcolor=TEXT_PRIMARY)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.06, right=0.985)
    for suffix, dpi in ((".png", 200), (".pdf", None)):
        fig.savefig(args.out + suffix, dpi=dpi, facecolor=SURFACE,
                    bbox_inches="tight")
    print("written:", args.out + ".png / .pdf")


if __name__ == "__main__":
    main()
