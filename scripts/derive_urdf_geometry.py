#!/usr/bin/env python3
"""Derive reproducible nominal geometry from a URDF and its collision meshes.

The script is intentionally read-only: it prints JSON and never updates a motion
profile.  Values from this report describe the checked-in CAD model, not the
dimensions of a particular assembled robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _vector(text: str | None, default: Vector3) -> Vector3:
    if text is None:
        return default
    values = tuple(float(item) for item in text.split())
    if len(values) != 3:
        raise ValueError(f"expected three values, got {text!r}")
    return values  # type: ignore[return-value]


def _matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _rpy_matrix(rpy: Vector3) -> Matrix3:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx: Matrix3 = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    ry: Matrix3 = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rz: Matrix3 = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    return _matrix_multiply(_matrix_multiply(rz, ry), rx)


def _transform(point: Vector3, rotation: Matrix3, translation: Vector3) -> Vector3:
    return tuple(
        sum(rotation[row][col] * point[col] for col in range(3)) + translation[row]
        for row in range(3)
    )  # type: ignore[return-value]


def _binary_stl_vertices(data: bytes) -> list[Vector3]:
    if len(data) < 84:
        raise ValueError("binary STL is shorter than its header")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        raise ValueError("not a canonical binary STL")
    vertices: list[Vector3] = []
    for index in range(triangle_count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        for vertex in range(3):
            start = 3 + vertex * 3
            vertices.append(tuple(float(v) for v in values[start : start + 3]))  # type: ignore[arg-type]
    return vertices


def _ascii_stl_vertices(data: bytes) -> list[Vector3]:
    vertices: list[Vector3] = []
    for line in data.decode("utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(v) for v in fields[1:]))  # type: ignore[arg-type]
    if not vertices:
        raise ValueError("STL contains no vertices")
    return vertices


def _stl_vertices(path: Path) -> list[Vector3]:
    data = path.read_bytes()
    try:
        return _binary_stl_vertices(data)
    except ValueError:
        return _ascii_stl_vertices(data)


def _distance_to_segment(point: Vector3, start: Vector3, end: Vector3) -> float:
    segment = tuple(end[i] - start[i] for i in range(3))
    relative = tuple(point[i] - start[i] for i in range(3))
    length_squared = sum(value * value for value in segment)
    if length_squared == 0.0:
        return math.sqrt(sum(value * value for value in relative))
    fraction = max(0.0, min(1.0, sum(relative[i] * segment[i] for i in range(3)) / length_squared))
    closest = tuple(start[i] + fraction * segment[i] for i in range(3))
    return math.sqrt(sum((point[i] - closest[i]) ** 2 for i in range(3)))


def _bounds(vertices: list[Vector3]) -> dict[str, list[float]] | None:
    if not vertices:
        return None
    minimum = [min(point[axis] for point in vertices) for axis in range(3)]
    maximum = [max(point[axis] for point in vertices) for axis in range(3)]
    return {
        "min_m": minimum,
        "max_m": maximum,
        "span_m": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def derive(urdf_path: Path, profile_path: Path) -> dict[str, object]:
    urdf_data = urdf_path.read_bytes()
    root = ET.fromstring(urdf_data)
    mesh_cache: dict[Path, list[Vector3]] = {}
    link_vertices: dict[str, list[Vector3]] = {}
    link_meshes: dict[str, list[str]] = {}

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        transformed: list[Vector3] = []
        filenames: list[str] = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None or "filename" not in mesh.attrib:
                continue
            filename = mesh.attrib["filename"]
            mesh_path = (urdf_path.parent / filename).resolve()
            if mesh_path not in mesh_cache:
                mesh_cache[mesh_path] = _stl_vertices(mesh_path)
            vertices = mesh_cache[mesh_path]
            origin = collision.find("origin")
            xyz = _vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            rotation = _rpy_matrix(rpy)
            transformed.extend(_transform(point, rotation, xyz) for point in vertices)
            filenames.append(filename)
        link_vertices[link_name] = transformed
        link_meshes[link_name] = filenames

    joints: dict[str, object] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = _vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        limit = joint.find("limit")
        limits_deg = None
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits_deg = [math.degrees(float(limit.attrib["lower"])), math.degrees(float(limit.attrib["upper"]))]
        joints[joint.attrib["name"]] = {
            "type": joint.attrib.get("type"),
            "parent": joint.find("parent").attrib["link"],  # type: ignore[union-attr]
            "child": joint.find("child").attrib["link"],  # type: ignore[union-attr]
            "origin_xyz_m": list(xyz),
            "origin_rpy_rad": list(rpy),
            "origin_translation_norm_m": math.sqrt(sum(value * value for value in xyz)),
            "limits_deg": limits_deg,
        }

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    capsules: list[dict[str, object]] = []
    for capsule in profile["collision"]["link_capsules"]:
        vertices = link_vertices.get(capsule["link"], [])
        p0 = tuple(float(value) for value in capsule["p0_m"])
        p1 = tuple(float(value) for value in capsule["p1_m"])
        required_radius = max((_distance_to_segment(point, p0, p1) for point in vertices), default=None)
        current_radius = float(capsule["radius_m"])
        capsules.append(
            {
                "id": capsule["id"],
                "link": capsule["link"],
                "p0_m": list(p0),
                "p1_m": list(p1),
                "current_radius_m": current_radius,
                "urdf_collision_meshes": link_meshes.get(capsule["link"], []),
                "mesh_vertex_count": len(vertices),
                "required_radius_for_current_segment_m": required_radius,
                "radius_margin_m": None if required_radius is None else current_radius - required_radius,
                "contains_urdf_collision_mesh": None if required_radius is None else current_radius + 1e-9 >= required_radius,
            }
        )

    links = {
        name: {
            "collision_meshes": link_meshes[name],
            "mesh_vertex_count": len(vertices),
            "aabb_in_link_frame": _bounds(vertices),
        }
        for name, vertices in link_vertices.items()
    }
    return {
        "urdf": str(urdf_path),
        "urdf_sha256": hashlib.sha256(urdf_data).hexdigest(),
        "profile": str(profile_path),
        "meaning": "nominal checked-in URDF/CAD geometry; not physical-arm verification",
        "joints": joints,
        "links": links,
        "profile_capsule_audit": capsules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.urdf.resolve(), args.profile.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
