#!/usr/bin/env python3
"""Open an actual PyBullet 3-D editor for the three Safe Panda scenarios."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import time
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "safe_panda_core_scenarios_150_plan.json"
DEFAULT_EXPORT = ROOT / "artifacts" / "safe_panda_scenario_editor_3d" / "edited_scenario.json"
NEUTRAL_JOINTS = (0.0, 0.41, 0.0, -1.85, 0.0, 2.26, 0.79, 0.0, 0.0)
PANDA_JOINTS = (0, 1, 2, 3, 4, 5, 6, 9, 10)
EE_LINK = 11


def midpoint(spec: dict[str, Any] | None, fallback: float = 0.0) -> float:
    if not spec:
        return fallback
    low = spec.get("low")
    high = spec.get("high")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return (float(low) + float(high)) / 2.0
    return fallback


def load_plan(path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    families = payload.get("scenario_families")
    if not isinstance(families, list) or len(families) != 3:
        raise ValueError("3-D editor requires exactly three scenario families")
    return payload


def resolve_midpoint_scene(plan: dict[str, Any], scenario_index: int, side: int = -1) -> dict[str, Any]:
    scenario = plan["scenario_families"][scenario_index]
    perturbations = scenario["perturbations"]
    runtime = plan["runtime"]
    goal = np.array(
        [
            midpoint(perturbations.get("goal_x_m")),
            midpoint(perturbations.get("goal_y_m"), 0.3),
            midpoint(perturbations.get("goal_z_m")),
        ],
        dtype=float,
    )
    obstacle = np.array(
        [
            midpoint(perturbations.get("obstacle_start_x_m")),
            midpoint(perturbations.get("obstacle_start_y_m")),
            midpoint(perturbations.get("obstacle_start_z_m")),
        ],
        dtype=float,
    )
    velocity = np.array(
        [
            midpoint(perturbations.get("obstacle_velocity_x_mps")),
            midpoint(perturbations.get("obstacle_velocity_y_mps")),
            midpoint(perturbations.get("obstacle_velocity_z_mps")),
        ],
        dtype=float,
    )
    radius = midpoint(perturbations.get("obstacle_radius_m"), 0.1)
    noise_sigma = midpoint(perturbations.get("measurement_noise_sigma_m"), 0.005)

    if scenario["id"] == "CS2_ORTHOGONAL_3D_CROSSING":
        obstacle[0] = side * midpoint(perturbations["obstacle_start_abs_x_m"], 0.25)
        velocity[0] = -side * midpoint(perturbations["obstacle_velocity_abs_x_mps"], 0.12)
    elif scenario["id"] == "CS3_GRAZING_NEAR_LIMIT":
        goal[0] = side * midpoint(perturbations["goal_abs_x_m"], 0.09)
        margin = midpoint(perturbations["grazing_margin_m"], 0.005)
        obstacle[0] = side * (radius + float(runtime["ee_collision_radius_m"]) + margin)

    return {
        "scenario_index": scenario_index,
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "side": side,
        "goal_offset_m": goal,
        "obstacle_start_offset_m": obstacle,
        "obstacle_velocity_mps": velocity,
        "obstacle_radius_m": radius,
        "measurement_noise_sigma_m": noise_sigma,
        "ee_collision_radius_m": float(runtime["ee_collision_radius_m"]),
        "preview_time_s": 0.0,
        "animate": False,
    }


def validate_plan(path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan = load_plan(path)
    resolved = [resolve_midpoint_scene(plan, index) for index in range(3)]
    return {
        "scenario_count": len(resolved),
        "scenario_ids": [item["scenario_id"] for item in resolved],
        "display_available": bool(Path("/tmp/.X11-unix").exists()),
        "pybullet_api": p.getAPIVersion(),
    }


class SafePanda3DScenarioEditor:
    """Interactive PyBullet scene using actual Panda URDF and direct manipulation."""

    def __init__(self, plan: dict[str, Any], export_path: Path) -> None:
        self.plan = plan
        self.export_path = export_path
        self.client = p.connect(
            p.GUI,
            options="--background_color_red=0.70 --background_color_green=0.72 --background_color_blue=0.78",
        )
        if self.client < 0:
            raise RuntimeError("PyBullet GUI could not connect; check DISPLAY/X11")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 120.0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)
        self.scenario_index = 0
        self.scene = resolve_midpoint_scene(self.plan, self.scenario_index)
        self.goal_body = -1
        self.obstacle_body = -1
        self.safety_body = -1
        self.task_cube = -1
        self.panda = -1
        self.ee_start = np.zeros(3)
        self.controls: dict[str, int] = {}
        self.control_values: dict[str, float] = {}
        self.debug_items: list[int] = []
        self.selected_body: int | None = None
        self.drag_plane_z = 0.0
        self.last_mouse = (0, 0)
        self.last_tick = time.monotonic()
        self.last_radius = -1.0
        self.last_export_counter = 0.0
        self.last_reset_counter = 0.0
        self._create_world()
        self._apply_scenario(0)

    def _create_box(
        self,
        half_extents: tuple[float, float, float],
        position: np.ndarray,
        color: tuple[float, float, float, float],
        collision: bool = True,
    ) -> int:
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client,
        )
        collision_id = (
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client,
            )
            if collision
            else -1
        )
        return p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            baseCollisionShapeIndex=collision_id,
            basePosition=position,
            physicsClientId=self.client,
        )

    def _create_sphere(
        self,
        radius: float,
        position: np.ndarray,
        color: tuple[float, float, float, float],
        collision: bool = True,
    ) -> int:
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self.client,
        )
        collision_id = (
            p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client)
            if collision
            else -1
        )
        return p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            baseCollisionShapeIndex=collision_id,
            basePosition=position,
            physicsClientId=self.client,
        )

    def _create_world(self) -> None:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
        self._create_box((3.0, 3.0, 0.01), np.array([0.0, 0.0, -0.41]), (0.18, 0.20, 0.22, 1.0))
        self._create_box((0.55, 0.35, 0.20), np.array([-0.30, 0.0, -0.20]), (0.72, 0.72, 0.70, 1.0))
        self.panda = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=(-0.60, 0.0, 0.0),
            useFixedBase=True,
            physicsClientId=self.client,
        )
        for joint, value in zip(PANDA_JOINTS, NEUTRAL_JOINTS):
            p.resetJointState(self.panda, joint, value, physicsClientId=self.client)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        self.ee_start = np.asarray(
            p.getLinkState(self.panda, EE_LINK, physicsClientId=self.client)[0], dtype=float
        )
        self.task_cube = self._create_box(
            (0.02, 0.02, 0.02),
            self.ee_start + np.array([-0.12, 0.18, -0.18]),
            (0.82, 0.12, 0.08, 1.0),
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=1.10,
            cameraYaw=48,
            cameraPitch=-34,
            cameraTargetPosition=self.ee_start + np.array([0.0, 0.13, -0.10]),
            physicsClientId=self.client,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)

    def _apply_scenario(self, index: int) -> None:
        self.scenario_index = index
        side = int(self.scene.get("side", -1)) if hasattr(self, "scene") else -1
        self.scene = resolve_midpoint_scene(self.plan, index, side=side)
        self._recreate_scene_bodies()
        self._rebuild_controls()
        self._refresh_debug_items()

    def _recreate_scene_bodies(self) -> None:
        for body in (self.goal_body, self.obstacle_body, self.safety_body):
            if body >= 0:
                p.removeBody(body, physicsClientId=self.client)
        goal_world = self.ee_start + self.scene["goal_offset_m"]
        obstacle_world = self.ee_start + self.scene["obstacle_start_offset_m"]
        self.goal_body = self._create_box(
            (0.025, 0.025, 0.025),
            goal_world,
            (0.12, 0.82, 0.24, 0.75),
        )
        self.obstacle_body = self._create_sphere(
            float(self.scene["obstacle_radius_m"]),
            obstacle_world,
            (0.95, 0.66, 0.12, 0.95),
        )
        self.safety_body = self._create_sphere(
            float(self.scene["obstacle_radius_m"] + self.scene["ee_collision_radius_m"]),
            obstacle_world,
            (0.95, 0.24, 0.12, 0.12),
            collision=False,
        )
        self.last_radius = float(self.scene["obstacle_radius_m"])

    def _add_slider(self, label: str, low: float, high: float, value: float) -> int:
        return p.addUserDebugParameter(label, low, high, value, physicsClientId=self.client)

    def _rebuild_controls(self) -> None:
        p.removeAllUserParameters(physicsClientId=self.client)
        scene = self.scene
        self.controls = {
            "scenario": self._add_slider("Scenario family [1-3]", 1, 3, self.scenario_index + 1),
            "side": self._add_slider("Signed side [-1 / +1]", -1, 1, float(scene["side"])),
            "goal_x": self._add_slider("Goal X offset (m)", -0.18, 0.18, float(scene["goal_offset_m"][0])),
            "goal_y": self._add_slider("Goal Y offset (m)", 0.05, 0.40, float(scene["goal_offset_m"][1])),
            "goal_z": self._add_slider("Goal Z offset (m)", -0.10, 0.18, float(scene["goal_offset_m"][2])),
            "obs_x": self._add_slider("Obstacle X offset (m)", -0.40, 0.40, float(scene["obstacle_start_offset_m"][0])),
            "obs_y": self._add_slider("Obstacle Y offset (m)", -0.05, 0.52, float(scene["obstacle_start_offset_m"][1])),
            "obs_z": self._add_slider("Obstacle Z offset (m)", -0.12, 0.22, float(scene["obstacle_start_offset_m"][2])),
            "radius": self._add_slider("Obstacle radius (m)", 0.025, 0.14, float(scene["obstacle_radius_m"])),
            "vel_x": self._add_slider("Velocity X (m/s)", -0.25, 0.25, float(scene["obstacle_velocity_mps"][0])),
            "vel_y": self._add_slider("Velocity Y (m/s)", -0.25, 0.25, float(scene["obstacle_velocity_mps"][1])),
            "vel_z": self._add_slider("Velocity Z (m/s)", -0.10, 0.10, float(scene["obstacle_velocity_mps"][2])),
            "time": self._add_slider("Preview time (s)", 0.0, 3.0, float(scene["preview_time_s"])),
            "animate": self._add_slider("Animate [0 / 1]", 0.0, 1.0, 1.0 if scene["animate"] else 0.0),
            "reset": p.addUserDebugParameter("RESET SCENARIO", 1, 0, 0, physicsClientId=self.client),
            "export": p.addUserDebugParameter("EXPORT RESOLVED JSON", 1, 0, 0, physicsClientId=self.client),
        }
        self.control_values = {
            key: float(p.readUserDebugParameter(control, physicsClientId=self.client))
            for key, control in self.controls.items()
        }
        self.last_reset_counter = self.control_values["reset"]
        self.last_export_counter = self.control_values["export"]

    def _read_controls(self) -> None:
        values = {
            key: float(p.readUserDebugParameter(control, physicsClientId=self.client))
            for key, control in self.controls.items()
        }
        requested_scenario = max(0, min(2, int(round(values["scenario"])) - 1))
        if requested_scenario != self.scenario_index:
            self._apply_scenario(requested_scenario)
            return
        requested_side = -1 if values["side"] < 0 else 1
        if requested_side != int(self.scene["side"]):
            self.scene = resolve_midpoint_scene(self.plan, self.scenario_index, requested_side)
            self._recreate_scene_bodies()
            self._rebuild_controls()
            self._refresh_debug_items()
            return

        changed = False
        mappings = {
            "goal_x": ("goal_offset_m", 0),
            "goal_y": ("goal_offset_m", 1),
            "goal_z": ("goal_offset_m", 2),
            "obs_x": ("obstacle_start_offset_m", 0),
            "obs_y": ("obstacle_start_offset_m", 1),
            "obs_z": ("obstacle_start_offset_m", 2),
            "vel_x": ("obstacle_velocity_mps", 0),
            "vel_y": ("obstacle_velocity_mps", 1),
            "vel_z": ("obstacle_velocity_mps", 2),
        }
        for control_name, (scene_name, component) in mappings.items():
            if abs(values[control_name] - self.control_values[control_name]) > 1e-9:
                self.scene[scene_name][component] = values[control_name]
                changed = True
        if abs(values["radius"] - self.control_values["radius"]) > 1e-9:
            self.scene["obstacle_radius_m"] = values["radius"]
            self._recreate_scene_bodies()
            changed = True
        if abs(values["time"] - self.control_values["time"]) > 1e-9:
            self.scene["preview_time_s"] = values["time"]
            changed = True
        self.scene["animate"] = values["animate"] >= 0.5

        if values["reset"] != self.last_reset_counter:
            self._apply_scenario(self.scenario_index)
            return
        if values["export"] != self.last_export_counter:
            self._export_scene()
            self.last_export_counter = values["export"]

        self.control_values = values
        if changed:
            self._update_body_poses()
            self._refresh_debug_items()

    def _update_body_poses(self) -> None:
        goal_world = self.ee_start + self.scene["goal_offset_m"]
        obstacle_start = self.ee_start + self.scene["obstacle_start_offset_m"]
        obstacle_world = obstacle_start + self.scene["obstacle_velocity_mps"] * float(self.scene["preview_time_s"])
        orientation = (0.0, 0.0, 0.0, 1.0)
        p.resetBasePositionAndOrientation(self.goal_body, goal_world, orientation, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.obstacle_body, obstacle_world, orientation, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.safety_body, obstacle_world, orientation, physicsClientId=self.client)

    def _refresh_debug_items(self) -> None:
        for item in self.debug_items:
            p.removeUserDebugItem(item, physicsClientId=self.client)
        self.debug_items.clear()
        goal = self.ee_start + self.scene["goal_offset_m"]
        obstacle_start = self.ee_start + self.scene["obstacle_start_offset_m"]
        obstacle_now = obstacle_start + self.scene["obstacle_velocity_mps"] * float(self.scene["preview_time_s"])
        velocity_end = obstacle_now + self.scene["obstacle_velocity_mps"] * 0.8
        for index in range(12):
            if index % 2:
                continue
            start = self.ee_start + (goal - self.ee_start) * (index / 12.0)
            end = self.ee_start + (goal - self.ee_start) * ((index + 1) / 12.0)
            self.debug_items.append(
                p.addUserDebugLine(start, end, (0.9, 0.15, 0.08), 3, physicsClientId=self.client)
            )
        self.debug_items.extend(
            [
                p.addUserDebugLine(obstacle_start, obstacle_now, (0.95, 0.65, 0.08), 2, physicsClientId=self.client),
                p.addUserDebugLine(obstacle_now, velocity_end, (0.95, 0.25, 0.08), 4, physicsClientId=self.client),
                p.addUserDebugText("GOAL - drag in x/y", goal + np.array([0.0, 0.0, 0.05]), (0.0, 0.45, 0.05), 1.2, physicsClientId=self.client),
                p.addUserDebugText("OBSTACLE - drag in x/y", obstacle_now + np.array([0.0, 0.0, self.scene["obstacle_radius_m"] + 0.04]), (0.65, 0.18, 0.03), 1.2, physicsClientId=self.client),
                p.addUserDebugText(
                    f"{self.scenario_index + 1}/3  {self.scene['scenario_name']}",
                    self.ee_start + np.array([-0.32, -0.25, 0.35]),
                    (0.05, 0.10, 0.14),
                    1.5,
                    physicsClientId=self.client,
                ),
                p.addUserDebugText(
                    "Click goal/obstacle and drag on the x/y plane. Use sliders for z, radius, velocity and time.",
                    self.ee_start + np.array([-0.32, -0.25, 0.30]),
                    (0.05, 0.10, 0.14),
                    1.0,
                    physicsClientId=self.client,
                ),
            ]
        )

    def _camera_ray(self, mouse_x: int, mouse_y: int) -> tuple[np.ndarray, np.ndarray]:
        width, height, _, _, _, forward, horizon, vertical, _, _, distance, target = p.getDebugVisualizerCamera(
            physicsClientId=self.client
        )
        target = np.asarray(target, dtype=float)
        forward = np.asarray(forward, dtype=float)
        horizon = np.asarray(horizon, dtype=float)
        vertical = np.asarray(vertical, dtype=float)
        camera = target - float(distance) * forward
        far = 1000.0
        ray_forward = target - camera
        ray_forward *= far / max(np.linalg.norm(ray_forward), 1e-9)
        center = camera + ray_forward - 0.5 * horizon + 0.5 * vertical
        ray_to = center + float(mouse_x) * horizon / max(width, 1) - float(mouse_y) * vertical / max(height, 1)
        return camera, ray_to

    def _ray_plane_point(self, mouse_x: int, mouse_y: int, plane_z: float) -> np.ndarray | None:
        ray_from, ray_to = self._camera_ray(mouse_x, mouse_y)
        direction = ray_to - ray_from
        if abs(direction[2]) < 1e-9:
            return None
        fraction = (plane_z - ray_from[2]) / direction[2]
        if fraction <= 0:
            return None
        return ray_from + fraction * direction

    def _handle_mouse(self) -> None:
        released = False
        for event in p.getMouseEvents(physicsClientId=self.client):
            event_type, mouse_x, mouse_y, button, button_state = event
            self.last_mouse = (int(mouse_x), int(mouse_y))
            if event_type == 2 and button == 0 and button_state & p.KEY_WAS_TRIGGERED:
                ray_from, ray_to = self._camera_ray(int(mouse_x), int(mouse_y))
                hit = p.rayTest(ray_from, ray_to, physicsClientId=self.client)[0]
                body = int(hit[0])
                if body in {self.goal_body, self.obstacle_body}:
                    self.selected_body = body
                    self.drag_plane_z = float(hit[3][2])
            if event_type == 2 and button == 0 and button_state & p.KEY_WAS_RELEASED:
                released = self.selected_body is not None
                self.selected_body = None

        if self.selected_body is not None:
            point = self._ray_plane_point(*self.last_mouse, self.drag_plane_z)
            if point is None:
                return
            if self.selected_body == self.goal_body:
                self.scene["goal_offset_m"][0:2] = point[0:2] - self.ee_start[0:2]
            elif self.selected_body == self.obstacle_body:
                current_time = float(self.scene["preview_time_s"])
                start_world = point - self.scene["obstacle_velocity_mps"] * current_time
                self.scene["obstacle_start_offset_m"][0:2] = start_world[0:2] - self.ee_start[0:2]
            self._update_body_poses()
            self._refresh_debug_items()
        elif released:
            self._rebuild_controls()

    def _export_scene(self) -> None:
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "source_profile": self.plan["profile"],
            "source_status": self.plan["status"],
            "scenario_id": self.scene["scenario_id"],
            "scenario_name": self.scene["scenario_name"],
            "goal_offset_m": self.scene["goal_offset_m"].tolist(),
            "obstacle_start_offset_m": self.scene["obstacle_start_offset_m"].tolist(),
            "obstacle_velocity_mps": self.scene["obstacle_velocity_mps"].tolist(),
            "obstacle_radius_m": float(self.scene["obstacle_radius_m"]),
            "ee_collision_radius_m": float(self.scene["ee_collision_radius_m"]),
            "measurement_noise_sigma_m": float(self.scene["measurement_noise_sigma_m"]),
            "preview_time_s": float(self.scene["preview_time_s"]),
            "coordinate_frame": "offsets relative to the neutral Panda end-effector position",
            "editor_note": "Resolved visual editor scene; not a simulator benchmark result.",
        }
        self.export_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        p.addUserDebugText(
            f"Exported: {self.export_path}",
            self.ee_start + np.array([-0.30, -0.25, 0.25]),
            (0.0, 0.45, 0.05),
            1.0,
            lifeTime=4.0,
            physicsClientId=self.client,
        )
        print(f"[scenario-editor-3d] exported {self.export_path}")

    def run(self) -> None:
        print("Safe Panda 3-D Scenario Editor is running.")
        print("Use the PyBullet sliders or drag the green goal / orange obstacle in x-y.")
        print("Close the PyBullet window or press Ctrl+C to stop.")
        try:
            while p.isConnected(self.client):
                now = time.monotonic()
                delta = min(0.05, now - self.last_tick)
                self.last_tick = now
                self._read_controls()
                self._handle_mouse()
                if self.scene["animate"]:
                    self.scene["preview_time_s"] = (float(self.scene["preview_time_s"]) + delta * 0.55) % 3.0
                    self._update_body_poses()
                    self._refresh_debug_items()
                p.stepSimulation(physicsClientId=self.client)
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            pass
        finally:
            if p.isConnected(self.client):
                p.disconnect(self.client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--export", default=str(DEFAULT_EXPORT))
    parser.add_argument("--check", action="store_true", help="validate plan and PyBullet API without opening a GUI")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(validate_plan(args.plan), indent=2))
        return 0

    editor = SafePanda3DScenarioEditor(load_plan(args.plan), Path(args.export))
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
