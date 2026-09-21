#!/usr/bin/env python3
"""Compute conservative capsule candidates from URDF collision meshes.

This tool is read-only.  It can fit complete links, individual URDF collision
meshes, deterministic spatial clusters, or adaptively split triangle groups.
Candidate axes include PCA and link-coordinate axes.  Endpoints span the full
assigned projection, and each triangle is assigned as a unit, so containing its
three vertices proves containment of the complete CAD face.  Physical/CAD
margins are reported separately and never silently folded into the geometric fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_urdf_geometry import (  # noqa: E402
    _rpy_matrix,
    _stl_vertices,
    _transform,
    _vector,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _link_mesh_triangles(urdf_path: Path) -> dict[str, list[dict[str, object]]]:
    root = ET.fromstring(urdf_path.read_bytes())
    cache: dict[Path, list[tuple[float, float, float]]] = {}
    result: dict[str, list[dict[str, object]]] = {}
    for link in root.findall("link"):
        meshes: list[dict[str, object]] = []
        for collision_index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None or "filename" not in mesh.attrib:
                continue
            mesh_path = (urdf_path.parent / mesh.attrib["filename"]).resolve()
            if mesh_path not in cache:
                cache[mesh_path] = _stl_vertices(mesh_path)
            origin = collision.find("origin")
            xyz = _vector(
                origin.attrib.get("xyz") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _vector(
                origin.attrib.get("rpy") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            rotation = _rpy_matrix(rpy)
            points = [_transform(point, rotation, xyz) for point in cache[mesh_path]]
            vertices = np.asarray(points, dtype=np.float64)
            if len(vertices) % 3:
                raise ValueError(f"{link.attrib['name']} STL vertex count is not divisible by 3")
            meshes.append(
                {
                    "collision_index": collision_index,
                    "mesh": str(mesh_path.relative_to(urdf_path.parent)),
                    "triangles": vertices.reshape((-1, 3, 3)),
                }
            )
        if meshes:
            result[link.attrib["name"]] = meshes
    return result


def _link_triangles(urdf_path: Path) -> dict[str, np.ndarray]:
    return {
        link: np.concatenate([row["triangles"] for row in meshes], axis=0)
        for link, meshes in _link_mesh_triangles(urdf_path).items()
    }


def _fit_for_axis(points: np.ndarray, axis: np.ndarray) -> dict[str, object]:
    u = np.asarray(axis, dtype=np.float64)
    u /= np.linalg.norm(u)
    helper = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = np.cross(u, helper)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    basis = np.stack((u, v, w), axis=1)
    projected = points @ basis
    # Center the capsule line in the transverse bounding rectangle.  For this
    # fixed axis/center, the smallest possible radius is the maximum transverse
    # distance.  The old fitter put the segment centers at the full axial extrema;
    # the spherical end caps then extended the envelope by one whole radius at
    # both ends and caused severe false positives.  At the same minimum radius R,
    # a point with axial coordinate s and transverse distance rho permits the
    # left center up to s+sqrt(R^2-rho^2), and requires the right center from
    # s-sqrt(R^2-rho^2).  Their intersection gives the shortest exact enclosing
    # segment without increasing R.
    transverse_center = 0.5 * (
        projected[:, 1:].min(axis=0) + projected[:, 1:].max(axis=0)
    )
    axial = projected[:, 0]
    transverse_distance = np.linalg.norm(
        projected[:, 1:] - transverse_center, axis=1
    )
    minimum_radius = float(transverse_distance.max())
    axial_reach = np.sqrt(
        np.maximum(minimum_radius**2 - transverse_distance**2, 0.0)
    )
    left_center_max = float(np.min(axial + axial_reach))
    right_center_min = float(np.max(axial - axial_reach))
    if left_center_max <= right_center_min:
        segment_start = left_center_max
        segment_end = right_center_min
    else:
        # One sphere is sufficient; any center in the feasible interval contains
        # every vertex.  Use its midpoint for deterministic symmetric slack.
        segment_start = segment_end = 0.5 * (left_center_max + right_center_min)
    local0 = np.array([segment_start, *transverse_center])
    local1 = np.array([segment_end, *transverse_center])
    p0 = basis @ local0
    p1 = basis @ local1
    delta = points - p0
    segment = p1 - p0
    denom = float(segment @ segment)
    fractions = (
        np.zeros(len(points), dtype=np.float64)
        if denom <= 1e-24
        else np.clip((delta @ segment) / denom, 0.0, 1.0)
    )
    closest = p0 + fractions[:, None] * segment
    distances = np.linalg.norm(points - closest, axis=1)
    return {
        "p0_m": p0.tolist(),
        "p1_m": p1.tolist(),
        "radius_m": float(distances.max()),
        "segment_length_m": float(np.linalg.norm(segment)),
        "axis": u.tolist(),
        "endpoint_strategy": "minimum_radius_maximum_inset_exact_enclosure",
    }


def _fit_segmented(
    triangles: np.ndarray, axis: np.ndarray, segment_count: int
) -> dict[str, object]:
    u = np.asarray(axis, dtype=np.float64)
    u /= np.linalg.norm(u)
    centroids = triangles.mean(axis=1) @ u
    if segment_count == 1:
        assignments = np.zeros(len(triangles), dtype=np.int64)
    else:
        edges = np.linspace(float(centroids.min()), float(centroids.max()), segment_count + 1)
        assignments = np.clip(np.searchsorted(edges[1:-1], centroids), 0, segment_count - 1)
    capsules = []
    for index in range(segment_count):
        selected = triangles[assignments == index]
        if not len(selected):
            return {"valid": False, "capsules": []}
        # Every triangle is assigned as a unit and all three of its vertices are
        # contained by this convex capsule, therefore the complete triangle is too.
        capsules.append(_fit_for_axis(selected.reshape((-1, 3)), u))
    volumes = []
    for capsule in capsules:
        radius = float(capsule["radius_m"])
        length = float(capsule["segment_length_m"])
        volumes.append(np.pi * radius * radius * length + 4.0 / 3.0 * np.pi * radius**3)
    return {
        "valid": True,
        "capsules": capsules,
        "maximum_radius_m": max(float(row["radius_m"]) for row in capsules),
        "total_capsule_volume_m3": float(sum(volumes)),
    }


def _cluster_assignments(points: np.ndarray, cluster_count: int) -> np.ndarray:
    """Deterministic farthest-first k-means on triangle centroids."""
    centers = [points[int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]]
    for _ in range(1, cluster_count):
        distance2 = np.min(
            np.stack([np.sum((points - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        centers.append(points[int(np.argmax(distance2))])
    centers_array = np.asarray(centers, dtype=np.float64)
    assignments = np.zeros(len(points), dtype=np.int64)
    for _ in range(40):
        distance2 = np.stack(
            [np.sum((points - center) ** 2, axis=1) for center in centers_array],
            axis=1,
        )
        updated = np.argmin(distance2, axis=1)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        for index in range(cluster_count):
            selected = points[assignments == index]
            if len(selected):
                centers_array[index] = selected.mean(axis=0)
    return assignments


def _fit_clustered(triangles: np.ndarray, cluster_count: int) -> dict[str, object]:
    assignments = _cluster_assignments(triangles.mean(axis=1), cluster_count)
    capsules = []
    for index in range(cluster_count):
        selected = triangles[assignments == index]
        if not len(selected):
            return {"valid": False, "capsules": []}
        points = selected.reshape((-1, 3))
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes = [*vh, *np.eye(3)]
        best = min(
            (_fit_for_axis(points, axis) for axis in axes),
            key=lambda row: (
                np.pi * float(row["radius_m"]) ** 2 * float(row["segment_length_m"])
                + 4.0 / 3.0 * np.pi * float(row["radius_m"]) ** 3
            ),
        )
        best["triangle_count"] = int(len(selected))
        capsules.append(best)
    volumes = []
    for capsule in capsules:
        radius = float(capsule["radius_m"])
        length = float(capsule["segment_length_m"])
        volumes.append(np.pi * radius * radius * length + 4.0 / 3.0 * np.pi * radius**3)
    return {
        "valid": True,
        "cluster_count": cluster_count,
        "capsules": capsules,
        "maximum_radius_m": max(float(row["radius_m"]) for row in capsules),
        "total_capsule_volume_m3": float(sum(volumes)),
    }


def _fit_parallel_clustered(
    triangles: np.ndarray, axis: np.ndarray, cluster_count: int
) -> dict[str, object]:
    """Tile a mesh cross-section with parallel capsules along one common axis."""
    u = np.asarray(axis, dtype=np.float64)
    u /= np.linalg.norm(u)
    helper = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = np.cross(u, helper)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    centroids = triangles.mean(axis=1)
    transverse = np.stack((centroids @ v, centroids @ w), axis=1)
    assignments = _cluster_assignments(transverse, cluster_count)
    capsules = []
    for index in range(cluster_count):
        selected = triangles[assignments == index]
        if not len(selected):
            return {"valid": False, "capsules": []}
        capsule = _fit_for_axis(selected.reshape((-1, 3)), u)
        capsule["triangle_count"] = int(len(selected))
        capsules.append(capsule)
    volumes = []
    for capsule in capsules:
        radius = float(capsule["radius_m"])
        length = float(capsule["segment_length_m"])
        volumes.append(np.pi * radius * radius * length + 4.0 / 3.0 * np.pi * radius**3)
    return {
        "valid": True,
        "cluster_count": cluster_count,
        "axis": u.tolist(),
        "capsules": capsules,
        "maximum_radius_m": max(float(row["radius_m"]) for row in capsules),
        "total_capsule_volume_m3": float(sum(volumes)),
    }


def _best_capsule(triangles: np.ndarray) -> dict[str, object]:
    points = triangles.reshape((-1, 3))
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = [*vh, *np.eye(3)]
    best = min(
        (_fit_for_axis(points, axis) for axis in axes),
        key=lambda row: (
            np.pi * float(row["radius_m"]) ** 2 * float(row["segment_length_m"])
            + 4.0 / 3.0 * np.pi * float(row["radius_m"]) ** 3
        ),
    )
    best["triangle_count"] = int(len(triangles))
    return best


def _fit_adaptive(triangles: np.ndarray, maximum_radius_m: float) -> dict[str, object]:
    """Recursively split triangle groups until every capsule is sufficiently local."""
    pending = [triangles]
    capsules = []
    while pending:
        selected = pending.pop()
        capsule = _best_capsule(selected)
        if float(capsule["radius_m"]) <= maximum_radius_m or len(selected) == 1:
            capsules.append(capsule)
            continue
        centroids = selected.mean(axis=1)
        centered = centroids - centroids.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projection = centroids @ vh[0]
        order = np.argsort(projection, kind="stable")
        split = len(order) // 2
        if split == 0 or split == len(order):
            capsules.append(capsule)
            continue
        pending.append(selected[order[split:]])
        pending.append(selected[order[:split]])
    capsules.sort(key=lambda row: tuple(float(value) for value in row["p0_m"]))
    volumes = []
    for capsule in capsules:
        radius = float(capsule["radius_m"])
        length = float(capsule["segment_length_m"])
        volumes.append(np.pi * radius * radius * length + 4.0 / 3.0 * np.pi * radius**3)
    return {
        "valid": True,
        "target_maximum_radius_m": maximum_radius_m,
        "capsule_count": len(capsules),
        "capsules": capsules,
        "maximum_radius_m": max(float(row["radius_m"]) for row in capsules),
        "total_capsule_volume_m3": float(sum(volumes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--cad-margin-mm", type=float, default=0.5)
    parser.add_argument("--pad-outgrowth-mm", type=float, default=3.0)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--max-clusters", type=int, default=0)
    parser.add_argument("--adaptive-max-radius-mm", type=float, default=0.0)
    args = parser.parse_args()
    if args.cad_margin_mm < 0 or args.pad_outgrowth_mm < 0:
        raise SystemExit("margins must be non-negative")
    if not 1 <= args.max_segments <= 8:
        raise SystemExit("--max-segments must be in [1,8]")
    if not 0 <= args.max_clusters <= 16:
        raise SystemExit("--max-clusters must be in [0,16]")
    if args.adaptive_max_radius_mm < 0:
        raise SystemExit("--adaptive-max-radius-mm must be non-negative")

    urdf_path = args.urdf.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    current_by_link = {
        row["link"]: row for row in profile["collision"]["link_capsules"]
        if row["link"] not in {"gripper_frame_link"}
    }
    result = {
        "schema_version": 1,
        "source": "urdf_collision_mesh_vertices",
        "urdf": str(urdf_path.relative_to(ROOT)),
        "urdf_sha256": _sha256(urdf_path),
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": _sha256(profile_path),
        "cad_margin_m": args.cad_margin_mm / 1000.0,
        "pad_outgrowth_m": args.pad_outgrowth_mm / 1000.0,
        "links": [],
    }
    mesh_triangles = _link_mesh_triangles(urdf_path)
    for link, mesh_rows in mesh_triangles.items():
        triangles = np.concatenate([row["triangles"] for row in mesh_rows], axis=0)
        points = triangles.reshape((-1, 3))
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes = [(f"pca_{index}", axis) for index, axis in enumerate(vh)]
        axes.extend((f"link_{index}", np.eye(3)[index]) for index in range(3))
        current = current_by_link.get(link)
        if current is not None:
            direction = np.asarray(current["p1_m"], float) - np.asarray(current["p0_m"], float)
            if np.linalg.norm(direction) > 1e-12:
                axes.append(("current_segment", direction))
        candidates = []
        for label, axis in axes:
            for segment_count in range(1, args.max_segments + 1):
                candidate = _fit_segmented(triangles, axis, segment_count)
                if not candidate["valid"]:
                    continue
                candidate["axis_source"] = label
                candidate["segment_count"] = segment_count
                candidates.append(candidate)
        # Prefer the tightest total enclosing volume; deterministic tie-breakers
        # avoid gratuitously adding segments for negligible changes.
        best = min(
            candidates,
            key=lambda row: (
                float(row["total_capsule_volume_m3"]),
                int(row["segment_count"]),
                str(row["axis_source"]),
            ),
        )
        best_by_segment_count = []
        for segment_count in range(1, args.max_segments + 1):
            same_count = [
                row for row in candidates if int(row["segment_count"]) == segment_count
            ]
            if not same_count:
                continue
            selected = min(
                same_count,
                key=lambda row: (
                    float(row["total_capsule_volume_m3"]),
                    str(row["axis_source"]),
                ),
            )
            best_by_segment_count.append(
                {
                    "segment_count": segment_count,
                    "axis_source": selected["axis_source"],
                    "maximum_radius_m": selected["maximum_radius_m"],
                    "total_capsule_volume_m3": selected["total_capsule_volume_m3"],
                }
            )
        physical_extra = (
            args.pad_outgrowth_mm / 1000.0
            if link in {"gripper_link", "moving_jaw_so101_v1_link"}
            else 0.0
        )
        for capsule in best["capsules"]:
            capsule["cad_radius_with_margin_m"] = (
                float(capsule["radius_m"]) + args.cad_margin_mm / 1000.0
            )
            capsule["demo_radius_with_physical_extra_m"] = (
                float(capsule["cad_radius_with_margin_m"]) + physical_extra
            )
        best["physical_extra_reason"] = (
            "installed pad thickness 2+/-1 mm"
            if physical_extra
            else None
        )
        result["links"].append(
            {
                "link": link,
                "triangle_count": int(len(triangles)),
                "candidate": best,
                "best_by_segment_count": best_by_segment_count,
                "clustered_candidates": [],
                "adaptive_candidate": None,
                "per_mesh_candidates": [],
            }
        )
        per_mesh_candidates = []
        for mesh_row in mesh_rows:
            mesh_array = mesh_row["triangles"]
            mesh_points = mesh_array.reshape((-1, 3))
            mesh_centered = mesh_points - mesh_points.mean(axis=0)
            _, _, mesh_vh = np.linalg.svd(mesh_centered, full_matrices=False)
            mesh_axes = [*mesh_vh, *np.eye(3)]
            by_segment_count = []
            for segment_count in range(1, args.max_segments + 1):
                options = [
                    _fit_segmented(mesh_array, axis, segment_count)
                    for axis in mesh_axes
                ]
                options = [item for item in options if item["valid"]]
                selected_mesh = min(
                    options,
                    key=lambda row: (
                        float(row["maximum_radius_m"]),
                        float(row["total_capsule_volume_m3"]),
                    ),
                )
                for capsule in selected_mesh["capsules"]:
                    capsule["cad_radius_with_margin_m"] = (
                        float(capsule["radius_m"]) + args.cad_margin_mm / 1000.0
                    )
                    capsule["demo_radius_with_physical_extra_m"] = (
                        float(capsule["cad_radius_with_margin_m"]) + physical_extra
                    )
                selected_mesh["segment_count"] = segment_count
                by_segment_count.append(selected_mesh)
            per_mesh_candidates.append(
                {
                    "collision_index": mesh_row["collision_index"],
                    "mesh": mesh_row["mesh"],
                    "triangle_count": int(len(mesh_array)),
                    "by_segment_count": by_segment_count,
                    "by_cluster_count": [],
                    "by_parallel_cluster_count": [],
                }
            )
            if args.max_clusters:
                by_cluster_count = []
                for cluster_count in range(2, args.max_clusters + 1):
                    clustered_mesh = _fit_clustered(mesh_array, cluster_count)
                    if not clustered_mesh["valid"]:
                        continue
                    for capsule in clustered_mesh["capsules"]:
                        capsule["cad_radius_with_margin_m"] = (
                            float(capsule["radius_m"])
                            + args.cad_margin_mm / 1000.0
                        )
                        capsule["demo_radius_with_physical_extra_m"] = (
                            float(capsule["cad_radius_with_margin_m"])
                            + physical_extra
                        )
                    by_cluster_count.append(clustered_mesh)
                per_mesh_candidates[-1]["by_cluster_count"] = by_cluster_count
                by_parallel_cluster_count = []
                for cluster_count in range(2, args.max_clusters + 1):
                    options = [
                        _fit_parallel_clustered(mesh_array, axis, cluster_count)
                        for axis in mesh_axes
                    ]
                    options = [item for item in options if item["valid"]]
                    selected_parallel = min(
                        options,
                        key=lambda row: (
                            float(row["maximum_radius_m"]),
                            float(row["total_capsule_volume_m3"]),
                        ),
                    )
                    for capsule in selected_parallel["capsules"]:
                        capsule["cad_radius_with_margin_m"] = (
                            float(capsule["radius_m"])
                            + args.cad_margin_mm / 1000.0
                        )
                        capsule["demo_radius_with_physical_extra_m"] = (
                            float(capsule["cad_radius_with_margin_m"])
                            + physical_extra
                        )
                    by_parallel_cluster_count.append(selected_parallel)
                per_mesh_candidates[-1][
                    "by_parallel_cluster_count"
                ] = by_parallel_cluster_count
        result["links"][-1]["per_mesh_candidates"] = per_mesh_candidates
        if args.max_clusters:
            clustered_candidates = []
            for cluster_count in range(2, args.max_clusters + 1):
                clustered = _fit_clustered(triangles, cluster_count)
                if not clustered["valid"]:
                    continue
                for capsule in clustered["capsules"]:
                    capsule["cad_radius_with_margin_m"] = (
                        float(capsule["radius_m"]) + args.cad_margin_mm / 1000.0
                    )
                    capsule["demo_radius_with_physical_extra_m"] = (
                        float(capsule["cad_radius_with_margin_m"]) + physical_extra
                    )
                clustered_candidates.append(clustered)
            result["links"][-1]["clustered_candidates"] = clustered_candidates
        if args.adaptive_max_radius_mm:
            adaptive = _fit_adaptive(
                triangles, args.adaptive_max_radius_mm / 1000.0
            )
            for capsule in adaptive["capsules"]:
                capsule["cad_radius_with_margin_m"] = (
                    float(capsule["radius_m"]) + args.cad_margin_mm / 1000.0
                )
                capsule["demo_radius_with_physical_extra_m"] = (
                    float(capsule["cad_radius_with_margin_m"]) + physical_extra
                )
            result["links"][-1]["adaptive_candidate"] = adaptive
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
