# Archived stage scripts

Frozen CLI runners for research stages that have concluded. Each script's
source module still lives in `src/lampc_cbf/` and every script still runs
unchanged (`PYTHONPATH=src python scripts/archive/<name>.py`); they are
archived only to keep `scripts/` limited to the active workflows. Do not
extend these — new experiments get a new script under `scripts/` with a new
versioned manifest in `configs/`.

| Script | Stage record |
|---|---|
| `run_collision_cone_liveness_ablation.py`, `run_liveness_development.py` | `docs/COLLISION_CONE_LIVENESS_PROTOCOL.md` |
| `run_dpcbf_ablation.py` | `docs/DPCBF_ABLATION.md` |
| `run_hard_scene_study.py`, `render_hard_scene_examples.py` | `docs/HARD_SCENE_STUDY.md` |
| `run_smoothness_ablation.py` | smoothness ablation stage (superseded by `run_smooth_dynamic_mpc_cbf.py`) |
| `run_dynamic_obstacle_mpc_cbf.py` | pre-smooth dynamic demo (superseded by `run_smooth_dynamic_mpc_cbf.py`) |
| `run_timeout_recovery_ablation.py` | timeout-recovery benchmark stage |

See `docs/RESEARCH_JOURNAL.md` for the stage history.
