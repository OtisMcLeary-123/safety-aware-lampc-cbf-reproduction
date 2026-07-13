#!/usr/bin/env python3
"""Render the eight-case alignment smoke result without altering its metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "artifacts/language_alignment_smoke_nvidia_nim_glm52/"
            "alignment_smoke_results.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/language_alignment_smoke_nvidia_nim_glm52/"
            "alignment_smoke_comparison.png"
        ),
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = payload["cases"]
    case_ids = [case["case_id"] for case in cases]
    reference_gamma = [case["reference"]["paper_gamma"] for case in cases]
    predicted_gamma = [case["prediction"]["gamma"] for case in cases]
    reference_labels = [case["reference"]["paper_label"] for case in cases]
    predicted_labels = [case["prediction"]["safety_label"] for case in cases]
    label_metrics = payload["metrics"]["paper_style_label_metrics"]
    continuous_metrics = payload["metrics"]["continuous_gamma_diagnostics"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].plot(
        case_ids,
        reference_gamma,
        "o-",
        linewidth=2,
        label="Paper Table 1 OF gamma",
    )
    axes[0].plot(
        case_ids,
        predicted_gamma,
        "s--",
        linewidth=2,
        label="NVIDIA NIM GLM-5.2 gamma",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Published example ID")
    axes[0].set_ylabel("Gamma (log scale; smaller = more cautious)")
    axes[0].set_title(
        "Continuous parameter\n"
        f"Spearman rho={continuous_metrics['spearman_rho']:.3f}, "
        f"MAE={continuous_metrics['mean_absolute_error']:.3f}"
    )
    axes[0].set_xticks(case_ids)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="lower left")

    width = 0.36
    axes[1].bar(
        [case_id - width / 2 for case_id in case_ids],
        reference_labels,
        width,
        label="Reference label from Table 1 gamma",
    )
    axes[1].bar(
        [case_id + width / 2 for case_id in case_ids],
        predicted_labels,
        width,
        label="Predicted label via Table 2",
    )
    axes[1].set_xlabel("Published example ID")
    axes[1].set_ylabel("Safety label (1=safest, 5=least safe)")
    axes[1].set_title(
        "Discrete calibration\n"
        f"Spearman rho={label_metrics['spearman_rho']:.3f}, "
        f"accuracy={label_metrics['exact_label_accuracy']:.0%}"
    )
    axes[1].set_xticks(case_ids)
    axes[1].set_yticks(range(1, 6))
    axes[1].set_ylim(0, 5.4)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="lower left")

    figure.suptitle(
        "N2-A1 blinded smoke test: eight published OF examples, not human ratings",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Controller-isolated theoretical gamma domain (0, 1]. "
        "No claim about the unpublished 50-query human study.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.93))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
