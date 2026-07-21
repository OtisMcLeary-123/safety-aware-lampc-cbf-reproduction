from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_safe_panda_scenario_editor.py"
SCRIPT_3D = ROOT / "scripts" / "run_safe_panda_3d_scenario_editor.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("scenario_editor_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_3d_launcher():
    spec = importlib.util.spec_from_file_location("scenario_editor_3d", SCRIPT_3D)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_editor_asset_check_reports_three_families() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["scenario_count"] == 3
    assert payload["episodes_per_scenario"] == 50
    assert payload["total_episodes"] == 150
    assert payload["scenario_ids"] == [
        "CS1_HEAD_ON_CLOSURE",
        "CS2_ORTHOGONAL_3D_CROSSING",
        "CS3_GRAZING_NEAR_LIMIT",
    ]


def test_editor_server_exposes_ui_and_plan() -> None:
    launcher = _load_launcher()
    server = launcher.build_server("127.0.0.1", 0, ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        with urlopen(
            f"http://{host}:{port}/tools/scenario_editor/", timeout=3
        ) as response:
            html = response.read().decode("utf-8")
        with urlopen(
            f"http://{host}:{port}/configs/safe_panda_core_scenarios_150_plan.json",
            timeout=3,
        ) as response:
            plan = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert "Scenario Lab" in html
    assert 'id="scenarioList"' in html
    assert 'id="editorFields"' in html
    assert 'id="topView"' in html
    assert len(plan["scenario_families"]) == 3


def test_editor_javascript_loads_versioned_plan() -> None:
    javascript = (ROOT / "tools" / "scenario_editor" / "app.js").read_text(
        encoding="utf-8"
    )
    assert 'const PLAN_URL = "/configs/safe_panda_core_scenarios_150_plan.json"' in javascript
    assert "validateImportedPlan" in javascript
    assert "downloadJSON" in javascript
    assert "renderTopView" in javascript
    assert "renderSideView" in javascript


def test_3d_editor_resolves_three_distinct_midpoint_scenes() -> None:
    launcher = _load_3d_launcher()
    plan = launcher.load_plan()
    scenes = [launcher.resolve_midpoint_scene(plan, index) for index in range(3)]

    assert [scene["scenario_id"] for scene in scenes] == [
        "CS1_HEAD_ON_CLOSURE",
        "CS2_ORTHOGONAL_3D_CROSSING",
        "CS3_GRAZING_NEAR_LIMIT",
    ]
    assert all(scene["goal_offset_m"].shape == (3,) for scene in scenes)
    assert all(scene["obstacle_start_offset_m"].shape == (3,) for scene in scenes)
    assert all(scene["obstacle_velocity_mps"].shape == (3,) for scene in scenes)
    assert scenes[1]["obstacle_start_offset_m"][0] < 0.0
    assert scenes[1]["obstacle_velocity_mps"][0] > 0.0


def test_3d_editor_check_does_not_open_gui() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_3D), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout[completed.stdout.index("{") :])
    assert payload["scenario_count"] == 3
    assert payload["pybullet_api"] > 0
