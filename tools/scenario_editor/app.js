const PLAN_URL = "/configs/safe_panda_core_scenarios_150_plan.json";

const elements = {
  planStatus: document.querySelector("#planStatus"),
  familyCount: document.querySelector("#familyCount"),
  episodeCount: document.querySelector("#episodeCount"),
  totalCount: document.querySelector("#totalCount"),
  scenarioList: document.querySelector("#scenarioList"),
  scenarioKicker: document.querySelector("#scenarioKicker"),
  scenarioTitle: document.querySelector("#scenarioTitle"),
  scenarioDescription: document.querySelector("#scenarioDescription"),
  topView: document.querySelector("#topView"),
  sideView: document.querySelector("#sideView"),
  topState: document.querySelector("#topState"),
  previewSummary: document.querySelector("#previewSummary"),
  editorFields: document.querySelector("#editorFields"),
  timeSlider: document.querySelector("#timeSlider"),
  timeValue: document.querySelector("#timeValue"),
  sideSelect: document.querySelector("#sideSelect"),
  animateToggle: document.querySelector("#animateToggle"),
  uncertaintyToggle: document.querySelector("#uncertaintyToggle"),
  resetButton: document.querySelector("#resetButton"),
  downloadButton: document.querySelector("#downloadButton"),
  exportButton: document.querySelector("#exportButton"),
  importButton: document.querySelector("#importButton"),
  importInput: document.querySelector("#importInput"),
  copyCommandButton: document.querySelector("#copyCommandButton"),
  copyFeedback: document.querySelector("#copyFeedback"),
};

const metaById = {
  CS1_HEAD_ON_CLOSURE: { short: "Head-on", tag: "AXIAL CLOSURE" },
  CS2_ORTHOGONAL_3D_CROSSING: { short: "3-D crossing", tag: "LATERAL + VERTICAL" },
  CS3_GRAZING_NEAR_LIMIT: { short: "Grazing", tag: "NEAR LIMIT" },
};

const state = {
  plan: null,
  originalPlan: null,
  activeIndex: 0,
  previewTime: 0,
  side: -1,
  showUncertainty: true,
  animate: false,
  animationFrame: null,
  animationTimestamp: null,
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function midpoint(spec, fallback = 0) {
  if (!spec || typeof spec !== "object") return fallback;
  if (Number.isFinite(spec.low) && Number.isFinite(spec.high)) {
    return (Number(spec.low) + Number(spec.high)) / 2;
  }
  return fallback;
}

function humanize(key) {
  return key
    .replace(/_mps$/, " (m/s)")
    .replace(/_m$/, " (m)")
    .replace(/_s$/, " (s)")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function format(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "--";
}

function currentScenario() {
  return state.plan.scenario_families[state.activeIndex];
}

function currentOriginalScenario() {
  const id = currentScenario().id;
  return state.originalPlan.scenario_families.find((scenario) => scenario.id === id);
}

function getPreviewData() {
  const scenario = currentScenario();
  const p = scenario.perturbations;
  const runtime = state.plan.runtime;
  const side = state.side;
  let goal = [midpoint(p.goal_x_m), midpoint(p.goal_y_m, 0.3), midpoint(p.goal_z_m)];
  let obstacle = [midpoint(p.obstacle_start_x_m), midpoint(p.obstacle_start_y_m), midpoint(p.obstacle_start_z_m)];
  let velocity = [midpoint(p.obstacle_velocity_x_mps), midpoint(p.obstacle_velocity_y_mps), midpoint(p.obstacle_velocity_z_mps)];
  const radius = midpoint(p.obstacle_radius_m, 0.1);
  const sigma = midpoint(p.measurement_noise_sigma_m, 0.005);

  if (scenario.id === "CS2_ORTHOGONAL_3D_CROSSING") {
    goal = [midpoint(p.goal_x_m), midpoint(p.goal_y_m, 0.3), midpoint(p.goal_z_m)];
    obstacle = [side * midpoint(p.obstacle_start_abs_x_m, 0.25), midpoint(p.obstacle_start_y_m, 0.16), midpoint(p.obstacle_start_z_m, 0.06)];
    velocity = [-side * midpoint(p.obstacle_velocity_abs_x_mps, 0.12), midpoint(p.obstacle_velocity_y_mps), midpoint(p.obstacle_velocity_z_mps)];
  }

  if (scenario.id === "CS3_GRAZING_NEAR_LIMIT") {
    goal = [side * midpoint(p.goal_abs_x_m, 0.09), midpoint(p.goal_y_m, 0.29), midpoint(p.goal_z_m, 0.04)];
    const margin = midpoint(p.grazing_margin_m, 0.005);
    obstacle = [side * (radius + Number(runtime.ee_collision_radius_m) + margin), midpoint(p.obstacle_start_y_m, 0.39), midpoint(p.obstacle_start_z_m)];
    velocity = [midpoint(p.obstacle_velocity_x_mps), midpoint(p.obstacle_velocity_y_mps, -0.16), midpoint(p.obstacle_velocity_z_mps)];
  }

  const t = state.previewTime;
  const obstacleNow = obstacle.map((value, index) => value + velocity[index] * t);
  const progress = Math.min(1, t / 3);
  const robotNow = goal.map((value) => value * progress);
  return {
    scenario,
    goal,
    obstacle,
    obstacleNow,
    velocity,
    robotNow,
    radius,
    sigma,
    eeRadius: Number(runtime.ee_collision_radius_m),
    combinedRadius: radius + Number(runtime.ee_collision_radius_m),
  };
}

function distance(a, b) {
  return Math.sqrt(a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0));
}

function renderScenarioList() {
  elements.scenarioList.replaceChildren();
  state.plan.scenario_families.forEach((scenario, index) => {
    const meta = metaById[scenario.id] || { short: scenario.name, tag: "SCENARIO" };
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scenario-card";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(index === state.activeIndex));
    button.innerHTML = `<span class="scenario-card-index">${String(index + 1).padStart(2, "0")}</span><span><strong></strong><small></small></span>`;
    button.querySelector("strong").textContent = meta.short;
    button.querySelector("small").textContent = meta.tag;
    button.addEventListener("click", () => {
      state.activeIndex = index;
      state.previewTime = 0;
      elements.timeSlider.value = "0";
      renderAll();
    });
    elements.scenarioList.append(button);
  });
}

function groupForKey(key) {
  if (key.startsWith("goal_")) return "Goal geometry";
  if (key.includes("velocity")) return "Obstacle motion";
  if (key.startsWith("obstacle_") || key.startsWith("grazing_")) return "Obstacle geometry";
  return "Noise and timing";
}

function renderEditorFields() {
  const scenario = currentScenario();
  const grouped = new Map();
  Object.entries(scenario.perturbations).forEach(([key, spec]) => {
    const group = groupForKey(key);
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group).push([key, spec]);
  });
  elements.editorFields.replaceChildren();

  grouped.forEach((items, groupName) => {
    const group = document.createElement("section");
    group.className = "editor-group";
    const label = document.createElement("div");
    label.className = "section-label";
    label.innerHTML = `<span></span><span class="section-rule"></span>`;
    label.querySelector("span").textContent = groupName;
    group.append(label);

    items.forEach(([key, spec]) => {
      if (spec.distribution === "uniform") {
        const field = document.createElement("div");
        field.className = "range-field";
        const header = document.createElement("div");
        header.className = "range-field-header";
        const fieldLabel = document.createElement("label");
        fieldLabel.textContent = humanize(key);
        const distribution = document.createElement("code");
        distribution.textContent = spec.use ? "uniform · optional" : "uniform";
        header.append(fieldLabel, distribution);

        const inputs = document.createElement("div");
        inputs.className = "range-inputs";
        const low = document.createElement("input");
        const high = document.createElement("input");
        low.type = high.type = "number";
        low.step = high.step = "0.001";
        low.value = String(spec.low);
        high.value = String(spec.high);
        low.setAttribute("aria-label", `${humanize(key)} minimum`);
        high.setAttribute("aria-label", `${humanize(key)} maximum`);
        const separator = document.createElement("span");
        separator.textContent = "to";

        const update = () => {
          const lowValue = Number(low.value);
          const highValue = Number(high.value);
          const valid = Number.isFinite(lowValue) && Number.isFinite(highValue) && lowValue <= highValue;
          low.setCustomValidity(valid ? "" : "Minimum must be less than or equal to maximum");
          high.setCustomValidity(valid ? "" : "Maximum must be greater than or equal to minimum");
          if (valid) {
            spec.low = lowValue;
            spec.high = highValue;
            renderPreview();
          }
        };
        low.addEventListener("input", update);
        high.addEventListener("input", update);
        inputs.append(low, separator, high);
        field.append(header, inputs);
        group.append(field);
      } else {
        const field = document.createElement("div");
        field.className = "range-field";
        const header = document.createElement("div");
        header.className = "range-field-header";
        const fieldLabel = document.createElement("label");
        fieldLabel.textContent = humanize(key);
        const distribution = document.createElement("code");
        distribution.textContent = spec.distribution;
        header.append(fieldLabel, distribution);
        const detail = document.createElement("div");
        detail.className = "derived-field";
        detail.textContent = spec.formula || spec.direction || spec.signed_by || JSON.stringify(spec.values || spec);
        field.append(header, detail);
        group.append(field);
      }
    });
    elements.editorFields.append(group);
  });
}

function svgFrame(width, height, content) {
  return `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs>
      <pattern id="grid-small" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 L 0 0 0 24" fill="none" stroke="#16313b" stroke-opacity="0.08" stroke-width="1"/></pattern>
      <pattern id="grid-major" width="96" height="96" patternUnits="userSpaceOnUse"><rect width="96" height="96" fill="url(#grid-small)"/><path d="M 96 0 L 0 0 0 96" fill="none" stroke="#16313b" stroke-opacity="0.12" stroke-width="1"/></pattern>
      <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f05d2f"/></marker>
      <filter id="soft-shadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#12232d" flood-opacity="0.18"/></filter>
    </defs>
    <rect width="100%" height="100%" fill="#f8f5ed"/>
    <rect width="100%" height="100%" fill="url(#grid-major)"/>
    ${content}
  </svg>`;
}

function renderTopView(data) {
  const width = 680;
  const height = 360;
  const margin = { left: 54, right: 24, top: 28, bottom: 36 };
  const xMin = -0.35, xMax = 0.35, yMin = -0.05, yMax = 0.50;
  const sx = (x) => margin.left + ((x - xMin) / (xMax - xMin)) * (width - margin.left - margin.right);
  const sy = (y) => height - margin.bottom - ((y - yMin) / (yMax - yMin)) * (height - margin.top - margin.bottom);
  const scale = (width - margin.left - margin.right) / (xMax - xMin);
  const obstacleRadius = data.radius * scale;
  const safetyRadius = data.combinedRadius * scale;
  const uncertaintyRadius = (data.combinedRadius + data.sigma * 3) * scale;
  const velocityScale = 1.1;
  const vxEnd = data.obstacleNow[0] + data.velocity[0] * velocityScale;
  const vyEnd = data.obstacleNow[1] + data.velocity[1] * velocityScale;
  const currentClearance = distance(data.robotNow, data.obstacleNow) - data.combinedRadius;
  const riskClass = currentClearance < 0 ? "COLLISION ENVELOPE" : currentClearance < 0.03 ? "TIGHT CLEARANCE" : "GEOMETRY VALID";
  elements.topState.textContent = riskClass;

  const uncertainty = state.showUncertainty
    ? `<circle cx="${sx(data.obstacleNow[0])}" cy="${sy(data.obstacleNow[1])}" r="${uncertaintyRadius}" fill="#f05d2f" fill-opacity="0.04" stroke="#f05d2f" stroke-opacity="0.35" stroke-dasharray="4 6"/>`
    : "";

  const content = `
    <rect x="${sx(xMin)}" y="${sy(yMax)}" width="${sx(xMax)-sx(xMin)}" height="${sy(yMin)-sy(yMax)}" fill="#14a7a0" fill-opacity="0.025" stroke="#12232d" stroke-opacity="0.18"/>
    <line x1="${sx(0)}" y1="${sy(yMin)}" x2="${sx(0)}" y2="${sy(yMax)}" stroke="#12232d" stroke-opacity="0.24" stroke-dasharray="3 5"/>
    <line x1="${sx(xMin)}" y1="${sy(0)}" x2="${sx(xMax)}" y2="${sy(0)}" stroke="#12232d" stroke-opacity="0.24" stroke-dasharray="3 5"/>
    <path d="M ${sx(0)} ${sy(0)} L ${sx(data.goal[0])} ${sy(data.goal[1])}" fill="none" stroke="#14a7a0" stroke-width="2" stroke-dasharray="7 7"/>
    ${uncertainty}
    <circle cx="${sx(data.obstacleNow[0])}" cy="${sy(data.obstacleNow[1])}" r="${safetyRadius}" fill="#f05d2f" fill-opacity="0.08" stroke="#f05d2f" stroke-width="1.5" stroke-dasharray="5 4"/>
    <circle cx="${sx(data.obstacleNow[0])}" cy="${sy(data.obstacleNow[1])}" r="${obstacleRadius}" fill="#f05d2f" fill-opacity="0.82" filter="url(#soft-shadow)"/>
    <line x1="${sx(data.obstacleNow[0])}" y1="${sy(data.obstacleNow[1])}" x2="${sx(vxEnd)}" y2="${sy(vyEnd)}" stroke="#f05d2f" stroke-width="2.5" marker-end="url(#arrow)"/>
    <circle cx="${sx(data.robotNow[0])}" cy="${sy(data.robotNow[1])}" r="9" fill="#0d2b38" stroke="#fffdf7" stroke-width="3" filter="url(#soft-shadow)"/>
    <circle cx="${sx(0)}" cy="${sy(0)}" r="4" fill="#0d2b38" fill-opacity="0.3"/>
    <circle cx="${sx(data.goal[0])}" cy="${sy(data.goal[1])}" r="10" fill="#14a7a0" fill-opacity="0.17" stroke="#14a7a0" stroke-width="2"/>
    <path d="M ${sx(data.goal[0])-5} ${sy(data.goal[1])} H ${sx(data.goal[0])+5} M ${sx(data.goal[0])} ${sy(data.goal[1])-5} V ${sy(data.goal[1])+5}" stroke="#14a7a0" stroke-width="2"/>
    <text x="${sx(data.robotNow[0])+14}" y="${sy(data.robotNow[1])-10}" fill="#12232d" font-family="DM Mono, monospace" font-size="9">EE</text>
    <text x="${sx(data.goal[0])+14}" y="${sy(data.goal[1])+4}" fill="#08746e" font-family="DM Mono, monospace" font-size="9">GOAL</text>
    <text x="${sx(data.obstacleNow[0])+obstacleRadius+8}" y="${sy(data.obstacleNow[1])+4}" fill="#a53e20" font-family="DM Mono, monospace" font-size="9">OBS</text>
    <text x="${sx(xMin)}" y="${height-12}" fill="#5c6c72" font-family="DM Mono, monospace" font-size="8">x = ${xMin.toFixed(2)}</text>
    <text x="${sx(xMax)-44}" y="${height-12}" fill="#5c6c72" font-family="DM Mono, monospace" font-size="8">${xMax.toFixed(2)}</text>
    <text x="12" y="${sy(yMax)+3}" fill="#5c6c72" font-family="DM Mono, monospace" font-size="8">y ${yMax.toFixed(2)}</text>
  `;
  elements.topView.innerHTML = svgFrame(width, height, content);
}

function renderSideView(data) {
  const width = 470;
  const height = 360;
  const margin = { left: 48, right: 24, top: 28, bottom: 36 };
  const yMin = -0.05, yMax = 0.50, zMin = -0.08, zMax = 0.25;
  const sy = (y) => margin.left + ((y - yMin) / (yMax - yMin)) * (width - margin.left - margin.right);
  const sz = (z) => height - margin.bottom - ((z - zMin) / (zMax - zMin)) * (height - margin.top - margin.bottom);
  const scale = (height - margin.top - margin.bottom) / (zMax - zMin);
  const obstacleRadius = data.radius * scale;
  const safetyRadius = data.combinedRadius * scale;
  const vyEnd = data.obstacleNow[1] + data.velocity[1] * 0.9;
  const vzEnd = data.obstacleNow[2] + data.velocity[2] * 0.9;
  const uncertainty = state.showUncertainty
    ? `<circle cx="${sy(data.obstacleNow[1])}" cy="${sz(data.obstacleNow[2])}" r="${(data.combinedRadius + data.sigma * 3) * scale}" fill="#f05d2f" fill-opacity="0.04" stroke="#f05d2f" stroke-opacity="0.35" stroke-dasharray="4 6"/>`
    : "";
  const content = `
    <rect x="${sy(yMin)}" y="${sz(zMax)}" width="${sy(yMax)-sy(yMin)}" height="${sz(zMin)-sz(zMax)}" fill="#efb83d" fill-opacity="0.025" stroke="#12232d" stroke-opacity="0.18"/>
    <line x1="${sy(yMin)}" y1="${sz(0)}" x2="${sy(yMax)}" y2="${sz(0)}" stroke="#12232d" stroke-opacity="0.32"/>
    <path d="M ${sy(0)} ${sz(0)} L ${sy(data.goal[1])} ${sz(data.goal[2])}" fill="none" stroke="#14a7a0" stroke-width="2" stroke-dasharray="7 7"/>
    ${uncertainty}
    <circle cx="${sy(data.obstacleNow[1])}" cy="${sz(data.obstacleNow[2])}" r="${safetyRadius}" fill="#f05d2f" fill-opacity="0.08" stroke="#f05d2f" stroke-width="1.5" stroke-dasharray="5 4"/>
    <circle cx="${sy(data.obstacleNow[1])}" cy="${sz(data.obstacleNow[2])}" r="${obstacleRadius}" fill="#f05d2f" fill-opacity="0.82" filter="url(#soft-shadow)"/>
    <line x1="${sy(data.obstacleNow[1])}" y1="${sz(data.obstacleNow[2])}" x2="${sy(vyEnd)}" y2="${sz(vzEnd)}" stroke="#f05d2f" stroke-width="2.5" marker-end="url(#arrow)"/>
    <circle cx="${sy(data.robotNow[1])}" cy="${sz(data.robotNow[2])}" r="9" fill="#0d2b38" stroke="#fffdf7" stroke-width="3" filter="url(#soft-shadow)"/>
    <circle cx="${sy(data.goal[1])}" cy="${sz(data.goal[2])}" r="10" fill="#14a7a0" fill-opacity="0.17" stroke="#14a7a0" stroke-width="2"/>
    <path d="M ${sy(data.goal[1])-5} ${sz(data.goal[2])} H ${sy(data.goal[1])+5} M ${sy(data.goal[1])} ${sz(data.goal[2])-5} V ${sz(data.goal[2])+5}" stroke="#14a7a0" stroke-width="2"/>
    <text x="${sy(yMin)}" y="${height-12}" fill="#5c6c72" font-family="DM Mono, monospace" font-size="8">y ${yMin.toFixed(2)}</text>
    <text x="${sy(yMax)-38}" y="${height-12}" fill="#5c6c72" font-family="DM Mono, monospace" font-size="8">${yMax.toFixed(2)}</text>
    <text x="10" y="${sz(zMax)+3}" fill="#5c6c72" font-family="DM Mono, monospace" font-size="8">z ${zMax.toFixed(2)}</text>
  `;
  elements.sideView.innerHTML = svgFrame(width, height, content);
}

function renderSummary(data) {
  const initialClearance = distance([0, 0, 0], data.obstacle) - data.combinedRadius;
  const currentClearance = distance(data.robotNow, data.obstacleNow) - data.combinedRadius;
  const speed = Math.sqrt(data.velocity.reduce((sum, value) => sum + value ** 2, 0));
  const cells = [
    ["Initial clearance", `${format(initialClearance * 1000, 1)} mm`, initialClearance >= 0.03 ? "safe" : "risk"],
    ["Preview clearance", `${format(currentClearance * 1000, 1)} mm`, currentClearance >= 0 ? "safe" : "risk"],
    ["Obstacle speed", `${format(speed, 3)} m/s`, ""],
    ["Noise sigma", `${format(data.sigma * 1000, 1)} mm`, ""],
  ];
  elements.previewSummary.replaceChildren();
  cells.forEach(([label, value, className]) => {
    const cell = document.createElement("div");
    cell.className = "summary-cell";
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    const valueNode = document.createElement("strong");
    valueNode.textContent = value;
    if (className) valueNode.classList.add(className);
    cell.append(labelNode, valueNode);
    elements.previewSummary.append(cell);
  });
}

function renderPreview() {
  const data = getPreviewData();
  renderTopView(data);
  renderSideView(data);
  renderSummary(data);
  elements.timeValue.textContent = `${format(state.previewTime, 2)} s`;
}

function renderAll() {
  const scenario = currentScenario();
  const meta = metaById[scenario.id] || { tag: scenario.id };
  elements.scenarioKicker.textContent = `${meta.tag} / ${scenario.id}`;
  elements.scenarioTitle.textContent = scenario.name;
  elements.scenarioDescription.textContent = scenario.objective;
  const hasSide = Boolean(scenario.perturbations.crossing_side || scenario.perturbations.grazing_side);
  elements.sideSelect.disabled = !hasSide;
  renderScenarioList();
  renderEditorFields();
  renderPreview();
}

function downloadJSON(filename, payload) {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function resetSelectedScenario() {
  const original = clone(currentOriginalScenario());
  state.plan.scenario_families[state.activeIndex] = original;
  state.previewTime = 0;
  elements.timeSlider.value = "0";
  renderAll();
}

function validateImportedPlan(plan) {
  if (!plan || !Array.isArray(plan.scenario_families) || plan.scenario_families.length !== 3) {
    throw new Error("The imported plan must contain exactly three scenario_families.");
  }
  plan.scenario_families.forEach((scenario) => {
    if (!scenario.id || !scenario.name || !scenario.perturbations) {
      throw new Error("Each scenario requires id, name, and perturbations.");
    }
  });
}

function startAnimation() {
  if (state.animationFrame) cancelAnimationFrame(state.animationFrame);
  state.animationTimestamp = null;
  const tick = (timestamp) => {
    if (!state.animate) return;
    if (state.animationTimestamp === null) state.animationTimestamp = timestamp;
    const delta = Math.min(0.05, (timestamp - state.animationTimestamp) / 1000);
    state.animationTimestamp = timestamp;
    state.previewTime = (state.previewTime + delta * 0.65) % 3;
    elements.timeSlider.value = String(state.previewTime);
    renderPreview();
    state.animationFrame = requestAnimationFrame(tick);
  };
  state.animationFrame = requestAnimationFrame(tick);
}

function stopAnimation() {
  if (state.animationFrame) cancelAnimationFrame(state.animationFrame);
  state.animationFrame = null;
  state.animationTimestamp = null;
}

function bindEvents() {
  elements.timeSlider.addEventListener("input", () => {
    state.previewTime = Number(elements.timeSlider.value);
    renderPreview();
  });
  elements.sideSelect.addEventListener("change", () => {
    state.side = Number(elements.sideSelect.value);
    renderPreview();
  });
  elements.uncertaintyToggle.addEventListener("change", () => {
    state.showUncertainty = elements.uncertaintyToggle.checked;
    renderPreview();
  });
  elements.animateToggle.addEventListener("change", () => {
    state.animate = elements.animateToggle.checked;
    if (state.animate) startAnimation(); else stopAnimation();
  });
  elements.resetButton.addEventListener("click", resetSelectedScenario);
  elements.downloadButton.addEventListener("click", () => {
    const scenario = currentScenario();
    downloadJSON(`${scenario.id.toLowerCase()}_edited.json`, scenario);
  });
  elements.exportButton.addEventListener("click", () => {
    const exported = clone(state.plan);
    exported.status = "edited_not_executed";
    exported.editor_exported_at = new Date().toISOString();
    downloadJSON("safe_panda_core_scenarios_150_edited.json", exported);
  });
  elements.importButton.addEventListener("click", () => elements.importInput.click());
  elements.importInput.addEventListener("change", async () => {
    const file = elements.importInput.files[0];
    if (!file) return;
    try {
      const imported = JSON.parse(await file.text());
      validateImportedPlan(imported);
      state.plan = imported;
      state.originalPlan = clone(imported);
      state.activeIndex = 0;
      state.previewTime = 0;
      elements.planStatus.textContent = "Imported plan · unsaved";
      renderAll();
    } catch (error) {
      window.alert(`Unable to import scenario plan: ${error.message}`);
    } finally {
      elements.importInput.value = "";
    }
  });
  elements.copyCommandButton.addEventListener("click", async () => {
    const command = state.plan.execution?.planned_run_command || "PYTHONPATH=src .venv/bin/python scripts/run_safe_panda_core_scenarios.py";
    try {
      await navigator.clipboard.writeText(command);
      elements.copyFeedback.textContent = "Command copied. The runner is still a planned implementation.";
    } catch {
      elements.copyFeedback.textContent = command;
    }
    window.setTimeout(() => { elements.copyFeedback.textContent = ""; }, 5000);
  });
  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
    if (["1", "2", "3"].includes(event.key)) {
      const index = Number(event.key) - 1;
      if (index < state.plan.scenario_families.length) {
        state.activeIndex = index;
        state.previewTime = 0;
        elements.timeSlider.value = "0";
        renderAll();
      }
    }
    if (event.key.toLowerCase() === "r") resetSelectedScenario();
    if (event.key.toLowerCase() === "e") elements.exportButton.click();
  });
}

async function init() {
  try {
    const response = await fetch(PLAN_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status} while loading ${PLAN_URL}`);
    const plan = await response.json();
    validateImportedPlan(plan);
    state.plan = plan;
    state.originalPlan = clone(plan);
    elements.planStatus.textContent = "Plan loaded · not executed";
    elements.familyCount.textContent = String(plan.scenario_families.length);
    elements.episodeCount.textContent = String(plan.sampling.episodes_per_scenario);
    elements.totalCount.textContent = String(plan.sampling.total_episodes_per_method);
    bindEvents();
    renderAll();
  } catch (error) {
    elements.planStatus.textContent = "Plan load failed";
    elements.topView.innerHTML = `<div class="loading-error"><strong>Scenario plan could not be loaded.</strong><p>${error.message}</p><p>Start the editor with <code>python scripts/run_safe_panda_scenario_editor.py</code>.</p></div>`;
    console.error(error);
  }
}

init();
