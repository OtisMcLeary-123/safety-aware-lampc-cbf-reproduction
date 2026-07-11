"""Trajectory smoothness metrics and visualization-only spline helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class SmoothnessMetrics:
    path_length: float
    mean_curvature: float
    max_curvature: float
    acceleration_rms: float
    jerk_rms: float
    jerk_max: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_smoothness_metrics(
    positions: Sequence[Sequence[float]], dt: float
) -> SmoothnessMetrics:
    """Calculate all safety-independent metrics from the raw trajectory."""

    import numpy as np

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (samples, 3)")
    if len(points) < 4:
        raise ValueError("at least four positions are required")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    velocity = np.gradient(points, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=2)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)

    speed = np.linalg.norm(velocity, axis=1)
    cross_norm = np.linalg.norm(np.cross(velocity, acceleration), axis=1)
    curvature = np.zeros_like(speed)
    moving = speed > 1e-6
    curvature[moving] = cross_norm[moving] / speed[moving] ** 3
    finite_curvature = curvature[np.isfinite(curvature) & moving]
    if not len(finite_curvature):
        finite_curvature = np.zeros(1)

    return SmoothnessMetrics(
        path_length=float(np.sum(segment_lengths)),
        mean_curvature=float(np.mean(finite_curvature)),
        max_curvature=float(np.max(finite_curvature)),
        acceleration_rms=float(np.sqrt(np.mean(acceleration_norm**2))),
        jerk_rms=float(np.sqrt(np.mean(jerk_norm**2))),
        jerk_max=float(np.max(jerk_norm)),
    )


def make_visual_smoothing_spline(
    positions: Sequence[Sequence[float]], *, sample_count: int = 400
) -> Any:
    """Return a parametric spline used only for plots, never safety metrics."""

    import numpy as np
    from scipy.interpolate import splprep, splev

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("positions must have shape (at least 4 samples, 3)")
    if sample_count < 4:
        raise ValueError("sample_count must be at least four")
    spline, _ = splprep(points.T, s=max(len(points) * 1e-6, 1e-9), k=3)
    parameter = np.linspace(0.0, 1.0, sample_count)
    return np.column_stack(splev(parameter, spline))


def make_reference_bspline(
    control_points: Sequence[Sequence[float]], *, sample_count: int = 600
) -> Any:
    """Interpolate task-planner control points into a dense continuous path."""

    import numpy as np
    from scipy.interpolate import make_interp_spline

    points = np.asarray(control_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("control_points must have shape (at least 4, 3)")
    if sample_count < len(points):
        raise ValueError("sample_count must cover all control points")
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if chord[-1] <= 0.0:
        raise ValueError("control points must describe a non-zero path")
    parameter = chord / chord[-1]
    spline = make_interp_spline(parameter, points, k=min(3, len(points) - 1), axis=0)
    dense_parameter = np.linspace(0.0, 1.0, sample_count)
    dense = np.asarray(spline(dense_parameter), dtype=float)
    dense[0] = points[0]
    dense[-1] = points[-1]
    return dense
