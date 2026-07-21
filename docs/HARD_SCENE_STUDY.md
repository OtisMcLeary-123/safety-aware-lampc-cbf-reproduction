# Hard-scene paired study

## Outcome

This experiment is deliberately harder than the earlier demonstration: the
reference is a straight line with no hand-designed avoidance waypoint, while a
spherical obstacle moves head-on toward that line. Across 50 paired randomized
episodes, none of the tested CBF/LLM variants improved the aggregate success
rate over the distance-only baseline. This is a negative result and is retained
as evidence rather than tuned away.

| Method | Success | Bootstrap 95% CI | Collisions | Mean minimum true clearance |
|---|---:|---:|---:|---:|
| Distance constraint | 17/50 (34%) | 22–48% | 33 | 1.75 mm |
| Fixed CBF, gamma=0.15 | 16/50 (32%) | 20–46% | 34 | 1.25 mm |
| LLM initial gamma=0.05 | 16/50 (32%) | 20–46% | 34 | 0.62 mm |
| Online feedback, initial gamma=0.15 | 16/50 (32%) | 20–46% | 34 | 1.25 mm |

The Hugging Face response to the feedback prompt selected `gamma=0.02` in
1.612 s without fallback. Every simulated episode terminated before the
random intervention time plus that measured latency, so zero of 50 feedback
updates reached the controller. Therefore the online-feedback row is exactly a
fixed-`gamma=0.15` row in this study. This demonstrates that a synchronous
cloud LLM cannot act as a fast safety reflex for this short-horizon scene.

## Material passport

- Study ID: `paper-hard-scene-online-feedback-50`
- Code state: this document and the artifact JSON are committed with the
  implementation used to produce them.
- Simulator: `PandaReachSafe-v3`, Tiny renderer, compatibility fork pinned in
  `pyproject.toml`.
- Controller: do-mpc/CasADi/IPOPT, 40 ms sample time, 15-step horizon, straight
  continuous reference, `delta_u_weight=2`.
- Safety geometry: obstacle radius 0.10 m plus collision radius 0.035 m.
- Dynamic obstacle: initial longitudinal offset 0.44 m; speed sampled uniformly
  from 0.025 to 0.20 m/s; lateral magnitude sampled from 0.055 to 0.065 m with
  randomized sign.
- Sensing: 0.67 s zero-order hold and 0.005 m Gaussian position noise.
- Randomization: NumPy seed `20260712`; all four methods receive identical
  paired obstacle speed, lateral offset, and intervention time per episode.
- Success criterion: goal reached and no negative raw true clearance.
- Uncertainty: 10,000 non-parametric bootstrap resamples of episode success.
- LLM: `Qwen/Qwen3-235B-A22B-Instruct-2507` through DeepInfra. One real request
  per unique prompt is validated and its decision plus measured latency are
  replayed across paired episodes. The API token is neither copied into nor
  tracked by the repository.
- Fallbacks: neither recorded LLM decision used fallback.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/archive/run_hard_scene_study.py
PYTHONPATH=src python scripts/archive/render_hard_scene_examples.py
```

The first command makes one LLM decision per unique prompt (or uses the
validated local cache) and then runs 200 simulations. The second renders a
deterministic paired example at 0.10 m/s: the distance-constrained controller
collides at -2.90 mm true clearance, while CBF with `gamma=0.02` reaches the
goal with 4.45 mm minimum true clearance. This example shows that the CBF can
help in an individual condition; it is not evidence of aggregate superiority.

## Evidence index

- `artifacts/hard_scene_study/study_summary.json`: machine-readable protocol,
  LLM decisions, latency, and aggregate results.
- `artifacts/hard_scene_study/episodes.csv`: all 200 raw episode-level outcomes.
- `artifacts/hard_scene_study/success_rate_and_speed.png`: success rates,
  bootstrap intervals, and outcome versus obstacle speed.
- `artifacts/hard_scene_study/distance_collision/`: baseline GIF, raw trace,
  and safety plot for the representative collision.
- `artifacts/hard_scene_study/cbf_gamma_002_avoidance/`: CBF GIF, raw trace,
  and safety plot for the paired successful avoidance.

## Interpretation and next experiment

The scene is challenging—success drops to roughly one third—but it is not yet
a full replication of the paper's online adaptation claim. The current
controller assumes the measured obstacle is static throughout each prediction
horizon, and the LLM latency is longer than these critical encounters. The
next defensible comparison is to decouple language intent from the safety
reflex: run LLM inference asynchronously, keep a local deterministic emergency
gamma policy at control rate, and extend the task duration so timely language
updates can be evaluated without pretending a late response arrived earlier.
