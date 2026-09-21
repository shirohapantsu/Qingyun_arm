from __future__ import annotations

import numpy as np

from scripts.fit_urdf_link_capsules import _fit_for_axis


def _distance_to_segment(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    segment = p1 - p0
    fraction = np.clip(((points - p0) @ segment) / (segment @ segment), 0.0, 1.0)
    return np.linalg.norm(points - (p0 + fraction[:, None] * segment), axis=1)


def test_fit_for_axis_insets_end_centers_without_losing_containment() -> None:
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.1],
        ]
    )

    fitted = _fit_for_axis(points, np.array([1.0, 0.0, 0.0]))
    p0 = np.asarray(fitted["p0_m"])
    p1 = np.asarray(fitted["p1_m"])
    radius = float(fitted["radius_m"])

    assert fitted["endpoint_strategy"] == "minimum_radius_maximum_inset_exact_enclosure"
    assert np.isclose(radius, 0.2)
    assert np.isclose(p0[0], -0.8)
    assert np.isclose(p1[0], 0.8)
    assert np.max(_distance_to_segment(points, p0, p1)) <= radius + 1e-12


def test_fit_for_axis_contains_every_assigned_triangle_vertex() -> None:
    rng = np.random.default_rng(20260915)
    points = rng.normal(size=(300, 3)) * np.array([0.12, 0.03, 0.02])
    fitted = _fit_for_axis(points, np.array([0.8, -0.3, 0.1]))
    p0 = np.asarray(fitted["p0_m"])
    p1 = np.asarray(fitted["p1_m"])
    radius = float(fitted["radius_m"])

    assert np.max(_distance_to_segment(points, p0, p1)) <= radius + 1e-12
