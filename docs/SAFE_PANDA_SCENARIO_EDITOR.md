# Safe Panda Scenario Lab

## Purpose

Scenario Lab provides two complementary editors for the three core scenario
families declared in `configs/safe_panda_core_scenarios_150_plan.json`:

1. A dependency-free local web editor for perturbation ranges, top/side
   projections, import, and suite export.
2. An actual PyBullet 3-D editor using the Panda URDF, table geometry, goal,
   obstacle, safety envelope, velocity guide, and direct object manipulation.

Neither editor executes do-mpc, CasADi, IPOPT, or provider requests. The web
preview is a geometric sketch. The 3-D editor uses PyBullet rendering and scene
geometry but does not execute the benchmark controller, so it must not be cited
as a rollout result.

## Run

### Actual PyBullet 3-D editor

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_3d_scenario_editor.py
```

The PyBullet window displays the neutral Panda, table, task cube, green goal,
orange obstacle, translucent inflated safety radius, nominal path, and obstacle
velocity. Use the right-hand debug panel to change scenario, side, goal,
obstacle pose, radius, velocity, and preview time.

Click and drag the green goal or orange obstacle to edit `x/y` directly in the
viewport. Use sliders for `z`, radius, velocity, and time. The export button
writes a resolved scene to
`artifacts/safe_panda_scenario_editor_3d/edited_scenario.json` without changing
the authoritative plan.

Validate the 3-D editor without opening a window:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_3d_scenario_editor.py --check
```

### Browser manifest editor

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_scenario_editor.py
```

The browser launcher binds to `127.0.0.1`, prints the local URL, and opens the default
browser. Use `--no-browser` when running remotely and `--port` to select another
port.

Validate the UI assets without starting a server:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_scenario_editor.py --check
```

## Browser Interaction Model

- Select scenario families from the left rail or press `1`, `2`, or `3`.
- Edit low/high values in the inspector; the midpoint drives the preview.
- Select left/right preview orientation for signed crossing scenarios.
- Move or animate the time slider to inspect nominal obstacle motion.
- Toggle the `3 sigma` measurement envelope.
- Press `R` to reset the selected family and `E` to export the suite.
- Import/export happens entirely in the browser. The server does not accept
  writes to the repository.

## UI Research Sources

The implementation adopts interaction patterns, not source code, from these
reviewed repositories:

| Repository | Revision | Pattern adopted |
|---|---|---|
| [mujoco-scene-editor](https://github.com/markusgrotz/mujoco-scene-editor/tree/a401997a9f5d02ab5073d5901fb37df4ff278a17) | `a401997a9f5d02ab5073d5901fb37df4ff278a17` | Local-only browser server, element selection, grouped property inspector, transform-style controls, undo/reset/export workflow. |
| [Webots](https://github.com/cyberbotics/webots/tree/22f33694b71c8954caf972406ab965eddf38f831) | `22f33694b71c8954caf972406ab965eddf38f831` | Scene-tree navigation separated from field/property editing and the central viewport. |
| [CARLA Traffic Generation Editor](https://github.com/carla-simulator/traffic-generation-editor/tree/98ce87b18eb1bd8265c97abba0c6408f307910bd) | `98ce87b18eb1bd8265c97abba0c6408f307910bd` | Map/viewport-centered editing, docked parameter forms, explicit import/export, and separate entity/motion controls. |
| [Foxglove Studio](https://github.com/foxglove/studio/tree/a8a589b801d1ad04915f4f22868989e222668f5e) | `a8a589b801d1ad04915f4f22868989e222668f5e` | Dense robotics workspace layout and visually distinct 3-D/data panels. The reviewed repository is archived, so it is not a dependency. |

The editor deliberately avoids Three.js, React, Qt, Viser, and a build tool in
version 1. Static HTML/CSS/JavaScript plus Python's standard HTTP server keeps
the reproduction environment small and makes the data boundary auditable.

## Component Contract

| Component | Responsibility | Key states |
|---|---|---|
| Scenario rail | Switch among the three families and show run counts | default, hover, selected, keyboard selected |
| Projection viewport | Show robot, goal, obstacle, safety radius, noise envelope, and velocity | loading, valid, tight clearance, collision envelope |
| Timeline | Preview nominal motion from `0` to `3 s` | idle, dragged, animating |
| Inspector | Edit uniform ranges and inspect derived/categorical fields | default, focus, invalid range, disabled side selector |
| Import/export | Load or download plan JSON | idle, imported-unsaved, validation error |

## Accessibility and Safety

- Controls use native buttons, inputs, selects, and visible focus rings.
- The three main regions have semantic labels and keyboard navigation.
- Motion honors `prefers-reduced-motion` except when the user explicitly turns
  on obstacle animation.
- The launcher binds to localhost and adds no-store and content-type headers.
- Imported JSON must contain exactly three scenario families with IDs, names,
  and perturbation objects.
- Browser edits never overwrite the authoritative plan automatically.
- The 3-D editor binds only to the local PyBullet GUI and exports to the ignored
  artifact directory.
- Direct dragging is restricted to the goal and obstacle and moves them on
  their current horizontal plane; vertical editing remains an explicit slider.

## Future Extensions

1. Generate the frozen 150 resolved instances from the edited plan.
2. Replace plane-constrained dragging with a full three-axis transform gizmo if
   PyBullet picking remains stable across supported platforms.
3. Stream an actual PyBullet rollout into the viewport through a read-only
   endpoint.
4. Add a third perspective or WebGL viewport only if the dependency and testing
   cost is justified.
