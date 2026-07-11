"""Command-line entry point for inspecting a reproduction run configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from typing import Sequence

from lampc_cbf.config import PaperConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lampc-cbf",
        description="Prepare a paper-aligned LaMPC-CBF run without calling an LLM API.",
    )
    parser.add_argument("--gamma", type=float, default=PaperConfig.gamma)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--allow-theoretical-gamma",
        action="store_true",
        help="accept gamma up to 1.0 instead of the experimental upper bound 0.15",
    )
    return parser


def create_run_manifest(config: PaperConfig, steps: int) -> dict[str, object]:
    """Create a serializable orchestration manifest; no solver or API is invoked."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    return {
        "mode": "dry-run",
        "external_api_calls": False,
        "steps": steps,
        "prediction_duration_seconds": config.prediction_duration,
        "config": asdict(config),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = PaperConfig()
    try:
        base.validate_gamma(args.gamma, experimental=not args.allow_theoretical_gamma)
        config = replace(base, gamma=args.gamma)
        manifest = create_run_manifest(config, args.steps)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

