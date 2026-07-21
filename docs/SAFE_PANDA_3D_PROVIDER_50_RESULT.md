# Safe Panda 3-D Provider Benchmark (50 Paired Scenarios)

## Scope

This is an opt-in engineering extension of the deterministic 50-scenario
suite. It combines a three-dimensional B-spline avoidance reference,
axis-specific MPC position tracking weights, a moving obstacle with nonzero
height/vertical velocity, collision-cone tangential reflexes, and asynchronous
NVIDIA NIM feedback. It is not an exact Table-4 reproduction or a physical
robot result.

## Setup

- Scenarios per method: 50
- Fixed method: `gamma=0.15`
- Feedback method: one provider decision per episode, applied after measured
  provider latency
- Provider/model: NVIDIA NIM `meta/llama-3.1-8b-instruct`
- Reference: `behind_spline`, `3d_waypoints`
- Waypoint offsets: `(0.14, 0.08, 0.10)` and `(0.14, 0.23, 0.10)` m
- Position weights `(x,y,z)`: `(1.0, 1.4, 1.2)`
- Provider hazard context: declared nominal 3-D spline proxy

## Result

| Method | Success | Collision | Solver failures/rejections | Mean minimum clearance |
|---|---:|---:|---:|---:|
| Fixed `gamma=0.15` | 19/50 | 31/50 | 0 / 0 | 0.00354 m |
| Async provider feedback | 23/50 | 27/50 | 0 / 0 | 0.00697 m |

- Paired success difference: `+0.08`
- Exact two-sided McNemar p-value: `0.125`
- Feedback-only successes: episodes `1, 7, 17, 19`
- Fixed-only successes: none
- Paired mean clearance difference: `+0.00343 m`
- Feedback updates applied: `50/50`
- Deadline misses and emergency fallbacks: zero for both methods

The observed improvement is not statistically significant at `alpha=0.05`.

## Provider Trace

- Valid uncached, non-fallback decisions: 50
- `gamma=0.03`: 42 decisions
- `gamma=0.07`: 8 decisions
- Mean / median / maximum latency: `0.365 / 0.339 / 0.711 s`

One request failed at episode 35. The collector retained the valid 34-record
prefix and resumed without repeating accepted requests. The final trace has no
fallback or cache-hit records.

## Successful Visualization

Async-feedback episode 1 was replayed with plots and animation enabled:

- Outcome: goal
- Collision: false
- Feedback gamma: `0.03`
- Provider latency: `0.548 s`
- Minimum true clearance: `0.02287 m`
- Raw trajectory ranges: `x=-0.0049..0.1380 m`, `z=0.1990..0.3147 m`

Raw animations, trajectory metrics, CSV files, and provider checkpoints remain
local under the repository artifact/credential policy. One derived gamma-sweep
summary figure is checked in under `docs/assets/` for the public README.

## Episode-1 Gamma Sweep

The same 3-D episode was run with five fixed gamma values and a direct-target
baseline without CBF:

| Method | Outcome | Steps | Minimum true clearance |
|---|---|---:|---:|
| `gamma=0.001` | Safety timeout | 260 | 0.08258 m |
| `gamma=0.040` | Goal | 235 | 0.01273 m |
| `gamma=0.065` | Goal | 233 | 0.00365 m |
| `gamma=0.100` | Goal | 226 | 0.00075 m |
| `gamma=0.150` | Collision | 142 | -0.00023 m |
| Baseline: direct target, no CBF | Collision | 74 | -0.00058 m |

This single-scenario sweep demonstrates a safety-efficiency tradeoff, not a
population-level gamma ranking. The baseline is a local nominal comparator and
must not be labeled as the paper's LaMPC-ED method.

## Reproduction Commands

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_safe_panda_3d_feedback.py
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_3d_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/render_safe_panda_3d_feedback_episode.py \
  --episode-id 1 \
  --output-dir artifacts/safe_panda_3d_feedback_episode_01
```
