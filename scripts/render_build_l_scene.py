#!/usr/bin/env python3
"""Render the restored Safe Panda Build-L scene and export its coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import panda_gym  # noqa: F401 - registers PandaBuildL-v3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", default="artifacts/build_l_scene"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "PandaBuildL-v3",
        render_mode="rgb_array",
        renderer="Tiny",
        render_width=720,
        render_height=480,
        render_distance=1.1,
        render_target_position=np.array([-0.15, 0.0, 0.12]),
    )
    try:
        observation, info = env.reset(seed=args.seed)
        frame = env.render()
        Image.fromarray(frame).save(output_dir / "build_l_gymnasium.png")
        metadata = {
            "environment": "PandaBuildL-v3",
            "seed": args.seed,
            "is_success": bool(info["is_success"]),
            "object_positions": observation["achieved_goal"].reshape(4, 3).tolist(),
            "target_positions": observation["desired_goal"].reshape(4, 3).tolist(),
            "observation_dimension": int(observation["observation"].shape[0]),
            "action_dimension": int(env.action_space.shape[0]),
        }
        (output_dir / "scene.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
